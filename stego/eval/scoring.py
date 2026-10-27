"""Score generated answers and summarise them into the baseline table.

Rows are dicts from ``run_generation`` (question, secret, channel, split, answer, ...).
Decode and format metrics are pure Python; fluency needs the original model.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Sequence

from stego.chat import build_messages
from stego.rewards import score_answer
from stego.rewards.fluency import CONTINUATION_TOLERANCE, chat_completion_logprobs, fluency_score, reference_model


def add_decode_metrics(rows: Iterable[dict], answer_key: str = "answer") -> list[dict]:
    out = []
    for row in rows:
        b = score_answer(row[answer_key], row["secret"], channel=row["channel"])
        out.append(
            {
                **row,
                "decoded": b.decode.decoded,
                "letter_acc": b.decode.letter_acc,
                "positional_acc": b.decode.positional_acc,
                "exact": b.decode.exact,
                "violations": b.format.violations,
                "format_penalty": b.format.penalty,
                "n_sentences": b.format.n_sentences,
                "short_sentences": b.format.short_sentences,
                "n_words": b.format.n_words,
            }
        )
    return out


def add_fluency(
    rows: Sequence[dict],
    model: Any,
    tokenizer: Any,
    references: dict[str, str],
    tolerance: float = 1.0,
    template_kwargs: dict | None = None,
    tolerance_cont: float = CONTINUATION_TOLERANCE,
) -> list[dict]:
    """Add fluency columns, scoring under the original model.

    ``answer_lp``, ``ref_lp`` and ``fluency``: over all answer tokens (the original score).
    ``answer_lp_cont``, ``ref_lp_cont`` and ``fluency_cont``: every sentence's first word
    left out, in both answers, with ``tolerance_cont`` (see stego.rewards.fluency). Both come
    from the same forward pass.

    ``references`` maps question_id to the original model's own plain answer. Both
    answers are scored against the plain prompt (no secret), so fluency measures how
    natural the answer is as a reply to the question.
    """
    kwargs = {"enable_thinking": False} if template_kwargs is None else template_kwargs
    ref_lp: dict[str, tuple[float, float]] = {}
    out = []
    with reference_model(model):
        for row in rows:
            messages = build_messages(row["question"])
            qid = row["question_id"]
            if qid not in ref_lp:
                ref_lp[qid] = chat_completion_logprobs(model, tokenizer, messages, references[qid], **kwargs)
            ref_all, ref_cont = ref_lp[qid]
            if row["answer"].strip():
                lp, lp_cont = chat_completion_logprobs(model, tokenizer, messages, row["answer"], **kwargs)
                flu, flu_cont = fluency_score(lp, ref_all, tolerance), fluency_score(lp_cont, ref_cont, tolerance_cont)
            else:
                lp = lp_cont = None
                flu = flu_cont = 0.0
            out.append({**row, "answer_lp": lp, "ref_lp": ref_all, "fluency": flu,
                        "answer_lp_cont": lp_cont, "ref_lp_cont": ref_cont, "fluency_cont": flu_cont})
    return out


def length_bucket(secret: str) -> str:
    """Secret length group for breakdowns: 3-4, 5-6 or 7+ letters."""
    n = len(secret)
    return "3-4" if n <= 4 else "5-6" if n <= 6 else "7+"


def is_clean(row: dict) -> bool:
    """Plain prose with no too-short sentences."""
    return not row["violations"] and row["short_sentences"] == 0


def summarize(rows: Iterable[dict], by: Sequence[str] = ("method", "split")) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in by)].append(row)
    table = []
    for key, g in sorted(groups.items()):
        entry = dict(zip(by, key))
        hard = [r for r in g if r.get("hard")]
        flu = [r["fluency"] for r in g if r.get("fluency") is not None]
        flu_cont = [r["fluency_cont"] for r in g if r.get("fluency_cont") is not None]
        gaps = [r["answer_lp"] - r["ref_lp"] for r in g if r.get("answer_lp") is not None and r.get("ref_lp") is not None]
        entry.update(
            n=len(g),
            exact=mean(r["exact"] for r in g),
            letter_acc=mean(r["letter_acc"] for r in g),
            clean_format=mean(is_clean(r) for r in g),
            fluency=mean(flu) if flu else None,
            fluency_cont=mean(flu_cont) if flu_cont else None,  # first words left out (v2)
            lp_gap=mean(gaps) if gaps else None,  # nats/token vs the reference; 0 = as natural
            exact_and_clean=mean(r["exact"] and is_clean(r) for r in g),
            hard_exact=mean(r["exact"] for r in hard) if hard else None,
        )
        table.append(entry)
    return table


def markdown_table(summary: Sequence[dict]) -> str:
    if not summary:
        return "(no rows)"
    cols = list(summary[0].keys())

    def fmt(v):
        if v is None:
            return "–"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(row[c]) for c in cols) + " |" for row in summary]
    return "\n".join(lines)
