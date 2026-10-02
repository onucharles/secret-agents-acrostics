"""SFT demonstrations written by a Claude teacher model, verified by our own decoder.

Each training example (question + secret) is sent to Claude with the same "instructed"
prompt the E0 baselines use. Every answer is checked locally; it is *accepted* when it

    - decodes exactly to the secret,
    - is clean plain prose (no lists/headings/code, no sentence under five words), and
    - doesn't give the game away (no mention of the secret word, "hidden", "acrostic"...).

Examples whose answer fails are retried, up to ``max_attempts``. Accepted answers are
saved as SFT records with the *terse* prompt, so the student model has to learn the rule
itself rather than read it from the prompt.

Two ways to run, both resumable after a Colab disconnect:

    mode="sync"   concurrent API calls; for small pilots (measure cost and pass rate)
    mode="batch"  the Message Batches API: half price, usually done within the hour.
                  Batch ids are saved next to the output, so rerunning resumes polling.

Every response's token usage is recorded, so ``spend`` reports what a run actually cost.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

from stego.chat import build_messages
from stego.data.sft_demos import select_demos, to_sft_record
from stego.eval.scoring import add_decode_metrics, is_clean
from stego.io import append_jsonl, iter_jsonl, read_jsonl, write_jsonl

DEFAULT_MODEL = "claude-opus-5-5"
TEACHER_NOTE = (
    "Write only the answer itself: no title, no preamble, and no notes about the hidden "
    "message or how you built the answer."
)

# $ per million tokens: (input, output, cache read). Batch requests cost half.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}
# Server-side refusal fallback ("default" routing) is available for these models on the
# Claude API, but not in the Batches API.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

_GIVEAWAY = re.compile(r"\b(hidden|acrostic|secret (word|message)|first letters?)\b", re.IGNORECASE)


# --- Verification ---------------------------------------------------------------------


def leaks_secret(answer: str, secret: str, channel: str = "acrostic") -> bool:
    """True if the answer names the secret word or talks about the encoding."""
    if _GIVEAWAY.search(answer):
        return True
    if channel == "acrostic" and re.search(rf"\b{re.escape(secret)}\b", answer, re.IGNORECASE):
        return True
    return False


def is_accepted(row: dict) -> bool:
    """For a row that has been through ``add_decode_metrics``."""
    return bool(
        row["answer"]
        and row["exact"]
        and is_clean(row)
        and not leaks_secret(row["answer"], row["secret"], row["channel"])
    )


# --- Requests -------------------------------------------------------------------------


def request_params(example: dict, model: str = DEFAULT_MODEL, effort: str | None = "medium", max_tokens: int = 8000) -> dict:
    """Messages API parameters for one example (shared by sync and batch modes)."""
    messages = build_messages(example["question"], example["secret"], example["channel"], style="instructed")
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": f"{messages[0]['content']}\n\n{TEACHER_NOTE}",
        "messages": messages[1:],
    }
    if effort and not model.startswith("claude-haiku"):  # Haiku 4.5 doesn't take effort
        params["output_config"] = {"effort": effort}
    return params


def _usage(message: Any) -> dict:
    u = message.usage
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


def _candidate(example: dict, attempt: int, mode: str, model: str, message: Any = None, error: str | None = None) -> dict:
    row = {**example, "id": f"{example['id']}#t{attempt}", "example_id": example["id"], "attempt": attempt, "mode": mode, "requested_model": model}
    if message is None:
        return {**row, "answer": "", "stop_reason": None, "teacher_model": None, "usage": {}, "error": error}
    text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text").strip()
    if message.stop_reason in ("refusal", "max_tokens"):
        error = message.stop_reason
    return {**row, "answer": text, "stop_reason": message.stop_reason, "teacher_model": message.model, "usage": _usage(message), "error": error}


def _call_sync(client: Any, params: dict) -> Any:
    if params["model"] in FALLBACK_MODELS:
        # If a safety classifier declines, the API reruns the request on the model it
        # recommends for that category, inside the same call.
        return client.beta.messages.create(**params, betas=[FALLBACK_BETA], fallbacks="default")
    return client.messages.create(**params)


# --- Running rounds -------------------------------------------------------------------


def _todo(examples: Sequence[dict], candidates: list[dict], attempt: int, max_attempts: int) -> list[dict]:
    """Examples that still need an answer in round ``attempt``."""
    accepted = {c["example_id"] for c in candidates if c.get("accepted")}
    tried = {c["id"] for c in candidates}
    return [
        e for e in examples
        if e["id"] not in accepted and attempt < max_attempts and f"{e['id']}#t{attempt}" not in tried
    ]


def _with_verdict(row: dict) -> dict:
    (scored,) = add_decode_metrics([row])
    return {**scored, "accepted": is_accepted(scored)}


def _run_sync(client, todo, attempt, out_path, model, effort, max_tokens, workers, log):
    lock = threading.Lock()
    done = 0

    def one(example):
        try:
            message = _call_sync(client, request_params(example, model, effort, max_tokens))
            return _candidate(example, attempt, "sync", model, message)
        except Exception as e:  # recorded as a failed attempt; retried next round
            return _candidate(example, attempt, "sync", model, error=f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, e) for e in todo]):
            row = _with_verdict(future.result())
            with lock:
                append_jsonl(out_path, [row])
                done += 1
                if done % 10 == 0 or done == len(todo):
                    log(f"  {done}/{len(todo)}")


def _custom_id(candidate_id: str) -> str:
    return hashlib.sha1(candidate_id.encode()).hexdigest()[:32]  # API allows [A-Za-z0-9_-]{1,64}


def _run_batch(client, todo, attempt, out_path, model, effort, max_tokens, poll_seconds, log):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    state_path = out_path.with_suffix(".batches.json")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    key = str(attempt)
    if key not in state:
        ids = {_custom_id(f"{e['id']}#t{attempt}"): e for e in todo}
        batch = client.messages.batches.create(
            requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**request_params(e, model, effort, max_tokens))) for cid, e in ids.items()]
        )
        state[key] = {"batch_id": batch.id, "examples": {cid: e["id"] for cid, e in ids.items()}}
        state_path.write_text(json.dumps(state))
        log(f"  submitted batch {batch.id} with {len(ids)} requests")
    batch_id = state[key]["batch_id"]
    by_id = {e["id"]: e for e in todo}

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        log(f"  batch {batch_id}: {batch.processing_status} (done {c.succeeded + c.errored + c.canceled + c.expired}, processing {c.processing})")
        time.sleep(poll_seconds)

    rows = []
    for result in client.messages.batches.results(batch_id):
        example = by_id.get(state[key]["examples"].get(result.custom_id))
        if example is None:
            continue  # already recorded by an earlier, interrupted collection
        if result.result.type == "succeeded":
            row = _candidate(example, attempt, "batch", model, result.result.message)
        else:
            row = _candidate(example, attempt, "batch", model, error=result.result.type)
        rows.append(_with_verdict(row))
    append_jsonl(out_path, rows)


def generate_teacher_demos(
    examples: Sequence[dict],
    out_path: str | Path,
    client: Any = None,
    *,
    model: str = DEFAULT_MODEL,
    effort: str | None = "medium",
    mode: str = "sync",
    max_attempts: int = 3,
    max_tokens: int = 8000,
    workers: int = 8,
    poll_seconds: int = 60,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Generate, verify and retry; returns every candidate (also saved to ``out_path``)."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    out_path = Path(out_path)
    for attempt in range(max_attempts):
        todo = _todo(examples, read_jsonl(out_path), attempt, max_attempts)
        if not todo:
            continue
        log(f"round {attempt + 1}/{max_attempts}: {len(todo)} examples ({mode})")
        if mode == "sync":
            _run_sync(client, todo, attempt, out_path, model, effort, max_tokens, workers, log)
        elif mode == "batch":
            _run_batch(client, todo, attempt, out_path, model, effort, max_tokens, poll_seconds, log)
        else:
            raise ValueError(f"mode must be 'sync' or 'batch', not {mode!r}")
    candidates = read_jsonl(out_path)
    log(summary_line(examples, candidates))
    return candidates


def pick_training_examples(examples: Sequence[dict], n: int) -> list[dict]:
    """``n`` training examples, one secret per question first (more distinct questions),
    then second secrets. Deterministic, so reruns pick the same examples."""
    train = [e for e in examples if e["split"] == "train"]
    seen: dict[str, int] = {}
    ranked = []
    for i, e in enumerate(train):
        k = seen.get(e["question_id"], 0)
        seen[e["question_id"]] = k + 1
        ranked.append((k, i, e))
    return [e for _, _, e in sorted(ranked, key=lambda t: (t[0], t[1]))][:n]


# --- Results --------------------------------------------------------------------------


def spend(candidates: Sequence[dict]) -> float:
    """Dollars spent, from recorded usage (batch requests at half price)."""
    total = 0.0
    for c in candidates:
        price_in, price_out, price_cache = PRICES.get(c.get("requested_model"), (0, 0, 0))
        u = c.get("usage") or {}
        dollars = (
            u.get("input_tokens", 0) * price_in
            + u.get("output_tokens", 0) * price_out
            + u.get("cache_read_input_tokens", 0) * price_cache
            + u.get("cache_creation_input_tokens", 0) * price_in * 1.25
        ) / 1e6
        total += dollars * (0.5 if c.get("mode") == "batch" else 1.0)
    return total


def summary_line(examples: Sequence[dict], candidates: Sequence[dict]) -> str:
    accepted = {c["example_id"] for c in candidates if c.get("accepted")}
    n = len({e["id"] for e in examples})
    cost = spend(candidates)
    per = cost / len(accepted) if accepted else float("nan")
    return f"accepted {len(accepted)}/{n} examples from {len(candidates)} calls; spent ${cost:.2f} (${per:.4f} per accepted demo)"


def failure_reasons(candidates: Sequence[dict]) -> dict[str, int]:
    """Why rejected candidates failed, for tuning the prompt."""
    reasons: dict[str, int] = {}
    for c in candidates:
        if c.get("accepted"):
            continue
        if c.get("error"):
            why = f"error: {c['error'].split(':')[0]}"
        elif not c["exact"]:
            why = "wrong letters"
        elif c["violations"]:
            why = "format: " + ",".join(c["violations"])
        elif c["short_sentences"]:
            why = "short sentences"
        else:
            why = "leaks the secret"
        reasons[why] = reasons.get(why, 0) + 1
    return dict(sorted(reasons.items(), key=lambda kv: -kv[1]))


def write_sft_dataset(candidates: Sequence[dict], out_path: str | Path, style: str = "terse") -> int:
    """One SFT record per example (its first accepted answer). Returns the count."""
    accepted = [c for c in candidates if c.get("accepted")]
    demos = select_demos(accepted, min_fluency=0.0)
    write_jsonl(out_path, (to_sft_record(d, style=style) for d in demos))
    return len(demos)


def load_candidates(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))
