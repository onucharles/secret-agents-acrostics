"""Step 3: SFT. Build training data with constrained decoding, train a LoRA adapter, evaluate.

Every step writes to ``run_dir`` and skips work already done, so after a Colab
disconnect you rerun the same cells and it carries on.

    build_sft_data   sample constrained answers for training questions, keep the most
                     fluent one per question that decodes exactly, is clean prose and
                     doesn't leak the secret; save it with the terse prompt
    train_sft        LoRA SFT with TRL, loss on the answer only; resumes from checkpoints
    evaluate         the trained model answers val questions with only "Secret: X"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from stego.chat import build_messages
from stego.colab import latest_checkpoint
from stego.data.sft_demos import is_verified, select_demos, to_sft_record
from stego.eval.scoring import add_decode_metrics, add_fluency, markdown_table, summarize
from stego.experiments.baselines import TOKENS_PER_SENTENCE, generate_references
from stego.generation.runner import DEFAULT_TEMPLATE_KWARGS, run_generation
from stego.io import append_jsonl, done_ids, read_jsonl, write_jsonl


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


# --- 1. Data ---------------------------------------------------------------------------


def build_sft_data(
    run_dir: Path,
    examples: Sequence[dict],
    model: Any,
    tokenizer: Any,
    vocab: Any,
    *,
    samples: int = 4,
    temperature: float = 0.8,
    batch_size: int = 16,
    score_chunk: int = 128,
    log: Callable = print,
) -> Path:
    """Returns the path of ``sft_train.jsonl`` (one chat record per kept example)."""
    run_dir = Path(run_dir)
    candidates = run_dir / "sft_candidates.jsonl"
    log("== generating constrained candidates ==")
    run_generation(
        examples, candidates, model, tokenizer,
        lambda r: build_messages(r["question"], r["secret"], r["channel"], style="instructed", few_shot=True),
        batch_size=batch_size, samples_per_row=samples, constrained=True, vocab=vocab,
        do_sample=True, temperature=temperature, tokens_per_sentence=TOKENS_PER_SENTENCE, log=log,
    )

    # Fluency only matters for candidates that pass the other checks, so references are
    # generated (and log-probs computed) only for those.
    scored_path = run_dir / "sft_candidates_scored.jsonl"
    scored_ids = done_ids(scored_path)
    todo = add_decode_metrics(r for r in read_jsonl(candidates) if r["id"] not in scored_ids)
    passing = [r for r in todo if is_verified(r, min_fluency=0.0)]
    log(f"== scoring: {len(scored_ids)} done, {len(todo)} to score ({len(passing)} pass letters/format/leak) ==")
    refs = generate_references(run_dir, passing, model, tokenizer, sampled=True, batch_size=batch_size, log=log) if passing else {}
    passing_ids = {r["id"] for r in passing}
    for i in range(0, len(todo), score_chunk):
        chunk = todo[i : i + score_chunk]
        need = [r for r in chunk if r["id"] in passing_ids]
        with_flu = {r["id"]: r for r in add_fluency(need, model, tokenizer, refs)} if need else {}
        append_jsonl(scored_path, (with_flu.get(r["id"], {**r, "fluency": None}) for r in chunk))
        log(f"  scored {min(i + score_chunk, len(todo))}/{len(todo)}")

    demos = select_demos(read_jsonl(scored_path), min_fluency=0.0)  # no minimum: most fluent passing answer
    out = run_dir / "sft_train.jsonl"
    write_jsonl(out, (to_sft_record(d, style="terse") for d in demos))
    log(f"{len(demos)}/{len(examples)} examples have a verified answer -> {out.name}")
    return out


# --- 2. Training -----------------------------------------------------------------------


def to_prompt_completion(records: Sequence[dict], tokenizer: Any, template_kwargs: dict | None = None) -> list[dict]:
    """Chat records -> {"prompt", "completion"} strings, rendered exactly as at generation
    time (thinking off), so training and inference see the same text."""
    kwargs = DEFAULT_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs
    out = []
    for rec in records:
        *prompt, answer = rec["messages"]
        text = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False, **kwargs)
        out.append({"prompt": text, "completion": answer["content"]})
    return out


def warmup_kwargs(config_cls: Any, ratio: float = 0.03) -> dict:
    """transformers 5 takes a warmup ratio as warmup_steps=0.03; 4.x has warmup_ratio."""
    return {"warmup_ratio": ratio} if "warmup_ratio" in config_cls.__dataclass_fields__ else {"warmup_steps": ratio}


def train_sft(
    run_dir: Path,
    sft_path: Path,
    model: Any,
    tokenizer: Any,
    *,
    epochs: float = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_length: int = 768,
    save_steps: int = 50,
    max_steps: int = -1,
    template_kwargs: dict | None = None,
    log: Callable = print,
) -> Path:
    """LoRA SFT; resumes from the latest checkpoint. Returns the saved adapter folder."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    run_dir = Path(run_dir)
    adapter_dir = run_dir / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        log(f"{adapter_dir} exists; training already finished")
        return adapter_dir

    data = Dataset.from_list(to_prompt_completion(read_jsonl(sft_path), tokenizer, template_kwargs))
    log(f"training on {len(data)} examples")
    tokenizer.padding_side = "right"
    half = next(model.parameters()).dtype
    config = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        **warmup_kwargs(SFTConfig),
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=2,
        max_length=max_length,
        fp16=half == torch.float16,
        bf16=half == torch.bfloat16,
        gradient_checkpointing=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data,
        processing_class=tokenizer,
        peft_config=LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    resume = latest_checkpoint(run_dir / "checkpoints")
    if resume:
        log(f"resuming from {resume.name}")
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(adapter_dir))
    tokenizer.padding_side = "left"
    return adapter_dir


def load_sft_model(base_name: str, adapter_dir: Path):
    """Base model plus the trained LoRA adapter, ready for generation."""
    from peft import PeftModel

    from stego.generation.runner import load_model

    base, tokenizer = load_model(base_name)
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return model, tokenizer


# --- 3. Evaluation ---------------------------------------------------------------------


def evaluate(
    run_dir: Path,
    rows: Sequence[dict],
    model: Any,
    tokenizer: Any,
    fluency_refs: dict[str, str],
    *,
    method: str = "sft",
    scorer: Any = None,
    batch_size: int = 16,
    log: Callable = print,
) -> list[dict]:
    """Greedy answers with only the terse prompt, scored like E0. Fluency is computed
    under the original model: ``scorer`` if given (a separate copy of it), otherwise
    ``model`` with its adapter switched off."""
    run_dir = Path(run_dir)
    gen = run_dir / f"gen_{method}.jsonl"
    run_generation(
        rows, gen, model, tokenizer,
        lambda r: build_messages(r["question"], r["secret"], r["channel"], style="terse"),
        batch_size=batch_size, tokens_per_sentence=TOKENS_PER_SENTENCE, log=log,
    )
    scored = add_fluency(add_decode_metrics({**g, "method": method} for g in read_jsonl(gen)), model if scorer is None else scorer, tokenizer, fluency_refs)
    write_jsonl(run_dir / f"scored_{method}.jsonl", scored)
    return summarize(scored, by=("method", "split"))


def results_table(*scored_paths: Path, split: str = "val") -> str:
    """One markdown table from several scored_*.jsonl files (e.g. E0 and SFT), one split."""
    rows = [r for p in scored_paths if Path(p).exists() for r in read_jsonl(p) if r["split"] == split]
    return markdown_table(summarize(rows, by=("method", "split")))
