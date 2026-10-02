"""Phase 2 / E0: datasets, reference answers, baselines, and a first batch of SFT demos.

Every step writes to ``run_dir`` and skips work already done there, so after a Colab
disconnect you rerun the same notebook cells and it carries on.

Methods in the baseline table:

    no_secret    the model's plain answer, scored against the secret: the chance floor
    prompted     instructed prompt with one worked example, greedy decoding
    constrained  same prompt, with acrostic constrained decoding (acrostic only)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Sequence

from stego.chat import build_messages
from stego.data.examples import IN_DOMAIN, OUT_OF_DOMAIN, build_examples, load_dolly, load_words, questions_from_records
from stego.data.sft_demos import select_demos, to_sft_record
from stego.eval.scoring import add_decode_metrics, add_fluency, markdown_table, summarize
from stego.generation.runner import run_generation
from stego.io import read_jsonl, write_jsonl

EVAL_SPLITS = ("val", "test_words", "test_long", "test_ood")
TOKENS_PER_SENTENCE = 48


def build_datasets(run_dir: Path, channel: str = "acrostic", n_eval: int = 200, seed: int = 0, log: Callable = print) -> Path:
    """examples.jsonl with every split; reused if it already exists."""
    path = Path(run_dir) / f"examples_{channel}.jsonl"
    if path.exists():
        log(f"{path.name} exists; reusing it")
        return path
    records = load_dolly()
    examples = build_examples(
        questions_from_records(records, IN_DOMAIN),
        questions_from_records(records, OUT_OF_DOMAIN),
        load_words(),
        channel=channel,
        n_eval=n_eval,
        seed=seed,
    )
    write_jsonl(path, (e.to_dict() for e in examples))
    counts = {s: sum(e.split == s for e in examples) for s in dict.fromkeys(e.split for e in examples)}
    log(f"wrote {len(examples)} examples to {path.name}: {counts}")
    return path


def eval_rows(examples: Sequence[dict], n_per_split: int) -> list[dict]:
    """The first ``n_per_split`` examples of each eval split, shortest secrets first so
    batches have similar lengths."""
    rows = []
    for split in EVAL_SPLITS:
        rows += [e for e in examples if e["split"] == split][:n_per_split]
    return sorted(rows, key=lambda e: (len(e["secret"]), e["id"]))


def generate_references(
    run_dir: Path,
    rows: Sequence[dict],
    model: Any,
    tokenizer: Any,
    sampled: bool = False,
    batch_size: int = 16,
    log: Callable = print,
) -> dict[str, str]:
    """The model's own plain answer to each question, keyed by question_id.

    ``sampled=False``: greedy, the model's single most likely answer. This is the
    ``no_secret`` baseline.

    ``sampled=True``: one sample at temperature 1 with no top-k/top-p truncation. This
    is the fluency reference. Greedy text is far more probable than any normal text
    (about -0.2 nats/token for Qwen3-1.7B, versus about -1 for ordinary fluent answers),
    so measuring against it scores everything as disfluent. A full-distribution sample
    sits at the model's typical level, so normal fluent text scores near 1.
    """
    questions = {r["question_id"]: {"id": r["question_id"], "question_id": r["question_id"], "question": r["question"], "secret": "", "channel": ""} for r in rows}
    path = Path(run_dir) / ("references_sampled.jsonl" if sampled else "references.jsonl")
    sampling = {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0} if sampled else {}
    run_generation(
        list(questions.values()), path, model, tokenizer,
        lambda r: build_messages(r["question"]),
        batch_size=batch_size, max_new_tokens=320, log=log, **sampling,
    )
    return {r["example_id"]: r["answer"] for r in read_jsonl(path)}


def run_baselines(
    run_dir: Path,
    rows: Sequence[dict],
    model: Any,
    tokenizer: Any,
    vocab: Any = None,
    methods: Sequence[str] = ("prompted", "constrained"),
    batch_size: int = 16,
    log: Callable = print,
) -> None:
    for method in methods:
        if method == "constrained" and rows and rows[0]["channel"] != "acrostic":
            log("constrained decoding is only implemented for the acrostic channel; skipping")
            continue
        log(f"== {method} ==")
        run_generation(
            rows,
            Path(run_dir) / f"gen_{method}.jsonl",
            model,
            tokenizer,
            lambda r: build_messages(r["question"], r["secret"], r["channel"], style="instructed", few_shot=True),
            batch_size=batch_size,
            constrained=method == "constrained",
            vocab=vocab,
            tokens_per_sentence=TOKENS_PER_SENTENCE,
            log=log,
        )


def score_baselines(
    run_dir: Path,
    rows: Sequence[dict],
    references: dict[str, str],
    model: Any = None,
    tokenizer: Any = None,
    methods: Sequence[str] = ("prompted", "constrained"),
    fluency_refs: dict[str, str] | None = None,
) -> list[dict]:
    """Score every method; with a model, include fluency. Writes scored_*.jsonl and
    baseline_table.md, and returns the summary rows.

    ``references`` (greedy) are the no_secret answers; ``fluency_refs`` (sampled) are
    what fluency is measured against, and default to ``references``.
    """
    run_dir = Path(run_dir)
    scored_all = []
    no_secret = [{**r, "id": f"{r['id']}#0", "example_id": r["id"], "answer": references[r["question_id"]]} for r in rows]
    for method, gen in [("no_secret", no_secret)] + [(m, read_jsonl(run_dir / f"gen_{m}.jsonl")) for m in methods if (run_dir / f"gen_{m}.jsonl").exists()]:
        scored = add_decode_metrics({**g, "method": method} for g in gen)
        if model is not None:
            scored = add_fluency(scored, model, tokenizer, fluency_refs or references)
        write_jsonl(run_dir / f"scored_{method}.jsonl", scored)
        scored_all += scored
    table = summarize(scored_all, by=("method", "split"))
    (run_dir / "baseline_table.md").write_text(markdown_table(table) + "\n")
    return table


def show_examples(run_dir: Path, method: str, n: int = 3, seed: int = 0, split: str | None = None) -> str:
    """A few scored answers as readable text, for eyeballing quality."""
    rows = read_jsonl(Path(run_dir) / f"scored_{method}.jsonl")
    if split:
        rows = [r for r in rows if r["split"] == split]
    picks = random.Random(seed).sample(rows, min(n, len(rows)))
    out = []
    for r in picks:
        flu = f"{r['fluency']:.2f}" if r.get("fluency") is not None else "–"
        out.append(
            f"[{method} · {r['split']}] Q: {r['question']}\n"
            f"secret={r['secret']} decoded={r['decoded']} exact={r['exact']} fluency={flu} violations={r['violations']}\n"
            f"{r['answer']}\n"
        )
    return "\n".join(out)


def sft_demo_smoke_test(
    run_dir: Path,
    train_rows: Sequence[dict],
    model: Any,
    tokenizer: Any,
    vocab: Any,
    n_prompts: int = 32,
    samples: int = 4,
    min_fluency: float = 0.5,
    batch_size: int = 16,
    log: Callable = print,
) -> list[dict]:
    """Phase 2 exit check: sample constrained answers for a few training prompts and keep
    the best verified one per prompt. Phase 3 runs the same thing at full scale."""
    run_dir = Path(run_dir)
    rows = list(train_rows[:n_prompts])
    path = run_dir / "sft_candidates_smoke.jsonl"
    run_generation(
        rows, path, model, tokenizer,
        lambda r: build_messages(r["question"], r["secret"], r["channel"], style="instructed", few_shot=True),
        batch_size=batch_size, samples_per_row=samples, constrained=True, vocab=vocab,
        do_sample=True, temperature=0.8, tokens_per_sentence=TOKENS_PER_SENTENCE, log=log,
    )
    candidates = read_jsonl(path)
    refs = generate_references(run_dir, rows, model, tokenizer, sampled=True, batch_size=batch_size, log=log)
    scored = add_fluency(add_decode_metrics(candidates), model, tokenizer, refs)
    demos = select_demos(scored, min_fluency=min_fluency)
    write_jsonl(run_dir / "sft_demos_smoke.jsonl", (to_sft_record(d) for d in demos))
    log(f"{len(demos)}/{len(rows)} prompts produced a verified demo ({len(demos) / max(len(rows), 1):.0%})")
    return demos
