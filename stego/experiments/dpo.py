"""Step 4: DPO. Sample answers from the SFT model, pair the best with the worst, train.

Every step writes to ``run_dir`` and skips work already done, so after a Colab
disconnect you rerun the same cells and it carries on. Raising the number of questions
reuses everything already generated, so a small pilot run is never wasted.

    build_pairs   sample answers from the SFT model (terse prompt), score each with the
                  GRPO reward, keep the best and worst per question as a pair
    pair_stats    how many pairs a run yields, to decide whether to scale up
    train_dpo     a new LoRA on top of the merged SFT model; the reference is the SFT
                  model itself (the new adapter switched off)
    load_dpo_model  base + merged SFT adapter + DPO adapter, for evaluation

Evaluation reuses ``sft.evaluate`` with ``scorer`` set to a separate copy of the
original model, because switching the DPO adapter off gives the SFT model, not the
original one.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Sequence

from stego.chat import build_messages
from stego.colab import latest_checkpoint
from stego.data.sft_demos import leaks_secret
from stego.eval.scoring import add_decode_metrics, add_fluency
from stego.experiments.baselines import TOKENS_PER_SENTENCE, generate_references
from stego.experiments.sft import warmup_kwargs
from stego.generation.runner import DEFAULT_TEMPLATE_KWARGS, run_generation
from stego.io import append_jsonl, done_ids, read_jsonl, write_jsonl
from stego.rewards import RewardWeights
from stego.rewards.fluency import reference_model


def reward(row: dict, weights: RewardWeights = RewardWeights()) -> float:
    """The GRPO reward of a scored answer: letters + exact + fluency - format penalties."""
    return (
        weights.letter * row["letter_acc"]
        + weights.exact * float(row["exact"])
        + weights.fluency * (row.get("fluency") or 0.0)
        - row["format_penalty"]
    )


def make_pairs(scored: Sequence[dict], min_gap: float = 0.3) -> list[dict]:
    """Per question: chosen = best answer that doesn't name the secret, rejected = worst
    answer. Kept only if their rewards differ by at least ``min_gap``."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in scored:
        groups[row["example_id"]].append(row)
    pairs = []
    for example_id, rows in groups.items():
        ranked = sorted(rows, key=reward)
        chosen = next((r for r in reversed(ranked) if not leaks_secret(r["answer"], r["secret"], r["channel"])), None)
        rejected = ranked[0]
        if chosen is None or chosen is rejected or reward(chosen) - reward(rejected) < min_gap:
            continue
        pairs.append(
            {
                "id": example_id,
                "secret": chosen["secret"],
                "channel": chosen["channel"],
                "prompt": build_messages(chosen["question"], chosen["secret"], chosen["channel"], style="terse"),
                "chosen": chosen["answer"],
                "rejected": rejected["answer"],
                "chosen_reward": reward(chosen),
                "rejected_reward": reward(rejected),
                "chosen_decoded": chosen["decoded"],
                "rejected_decoded": rejected["decoded"],
            }
        )
    return pairs


def build_pairs(
    run_dir: Path,
    rows: Sequence[dict],
    model: Any,
    tokenizer: Any,
    *,
    samples: int = 8,
    temperature: float = 1.0,
    min_gap: float = 0.3,
    batch_size: int = 16,
    score_chunk: int = 128,
    log: Callable = print,
) -> Path:
    """Returns the path of ``dpo_pairs.jsonl``. ``model`` is the SFT model (base plus
    adapter); its adapter is switched off for the fluency references and scores.

    Pairs are rebuilt from all scores on every call, so changing ``min_gap`` costs nothing.
    """
    run_dir = Path(run_dir)
    samples_path = run_dir / "dpo_samples.jsonl"
    log("== sampling from the SFT model ==")
    run_generation(
        rows, samples_path, model, tokenizer,
        lambda r: build_messages(r["question"], r["secret"], r["channel"], style="terse"),
        batch_size=batch_size, samples_per_row=samples, do_sample=True, temperature=temperature,
        tokens_per_sentence=TOKENS_PER_SENTENCE, log=log,
    )

    wanted = {r["id"] for r in rows}
    scored_path = run_dir / "dpo_samples_scored.jsonl"
    scored_ids = done_ids(scored_path)
    todo = add_decode_metrics(r for r in read_jsonl(samples_path) if r["example_id"] in wanted and r["id"] not in scored_ids)
    log(f"== scoring: {len(scored_ids)} done, {len(todo)} to score ==")
    if todo:
        with reference_model(model):  # references come from the original model
            refs = generate_references(run_dir, todo, model, tokenizer, sampled=True, batch_size=batch_size, log=log)
        for i in range(0, len(todo), score_chunk):
            append_jsonl(scored_path, add_fluency(todo[i : i + score_chunk], model, tokenizer, refs))
            log(f"  scored {min(i + score_chunk, len(todo))}/{len(todo)}")

    scored = [r for r in read_jsonl(scored_path) if r["example_id"] in wanted]
    pairs = make_pairs(scored, min_gap)
    out = run_dir / "dpo_pairs.jsonl"
    write_jsonl(out, pairs)
    log(f"{len(pairs)}/{len(rows)} questions gave a pair -> {out.name}")
    return out


def pair_stats(run_dir: Path, rows: Sequence[dict]) -> dict:
    """Numbers for deciding whether to scale up (only for ``rows``' questions)."""
    run_dir = Path(run_dir)
    wanted = {r["id"] for r in rows}
    scored = [r for r in read_jsonl(run_dir / "dpo_samples_scored.jsonl") if r["example_id"] in wanted]
    pairs = [p for p in read_jsonl(run_dir / "dpo_pairs.jsonl") if p["id"] in wanted]
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_q[r["example_id"]].append(r)
    return {
        "questions": len(by_q),
        "samples": len(scored),
        "pairs": len(pairs),
        "pairs_kept": len(pairs) / max(len(by_q), 1),
        "median_gap": median(p["chosen_reward"] - p["rejected_reward"] for p in pairs) if pairs else None,
        "exact_rate": mean(r["exact"] for r in scored) if scored else None,
        "questions_with_exact": mean(any(r["exact"] for r in g) for g in by_q.values()) if by_q else None,
        "letter_acc": mean(r["letter_acc"] for r in scored) if scored else None,
        "mean_reward": mean(reward(r) for r in scored) if scored else None,
    }


def show_pairs(run_dir: Path, n: int = 2) -> str:
    """The first few pairs as readable text."""
    out = []
    for p in read_jsonl(Path(run_dir) / "dpo_pairs.jsonl")[:n]:
        out.append(
            f"Q: {p['prompt'][-1]['content']}\nsecret={p['secret']}\n"
            f"CHOSEN  (decoded={p['chosen_decoded']}, reward={p['chosen_reward']:.2f}): {p['chosen']}\n"
            f"REJECTED (decoded={p['rejected_decoded']}, reward={p['rejected_reward']:.2f}): {p['rejected']}\n"
        )
    return "\n".join(out)


# --- Training --------------------------------------------------------------------------


def load_merged_sft(base_name: str, sft_adapter: Path):
    """The SFT model as a plain model: base with the SFT adapter merged in."""
    from peft import PeftModel

    from stego.generation.runner import load_model

    base, tokenizer = load_model(base_name)
    return PeftModel.from_pretrained(base, str(sft_adapter)).merge_and_unload(), tokenizer


def train_dpo(
    run_dir: Path,
    pairs_path: Path,
    model: Any,
    tokenizer: Any,
    *,
    beta: float = 0.1,
    epochs: float = 1,
    learning_rate: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    batch_size: int = 2,
    grad_accum: int = 8,
    max_length: int = 1024,
    save_steps: int = 25,
    max_steps: int = -1,
    template_kwargs: dict | None = None,
    log: Callable = print,
) -> Path:
    """LoRA DPO on the merged SFT model (``load_merged_sft``); resumes from the latest
    checkpoint. With no ``ref_model``, TRL scores the reference by switching the new
    adapter off, which gives the SFT model. Returns the saved adapter folder."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer

    run_dir = Path(run_dir)
    adapter_dir = run_dir / "adapter"
    if (adapter_dir / "adapter_config.json").exists():
        log(f"{adapter_dir} exists; training already finished")
        return adapter_dir

    pairs = read_jsonl(pairs_path)
    kwargs = DEFAULT_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs
    # Prompts rendered as at generation time (thinking off); TRL appends EOS to the answers.
    data = Dataset.from_list([
        {"prompt": tokenizer.apply_chat_template(p["prompt"], add_generation_prompt=True, tokenize=False, **kwargs),
         "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])
    log(f"training on {len(data)} pairs")
    tokenizer.padding_side = "right"
    half = next(model.parameters()).dtype
    config = DPOConfig(
        output_dir=str(run_dir / "checkpoints"),
        beta=beta,
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        **warmup_kwargs(DPOConfig, 0.1),
        logging_steps=5,
        save_steps=save_steps,
        save_total_limit=2,
        max_length=max_length,
        fp16=half == torch.float16,
        bf16=half == torch.bfloat16,
        gradient_checkpointing=torch.cuda.is_available(),
        report_to="none",
    )
    trainer = DPOTrainer(
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


def load_dpo_model(base_name: str, sft_adapter: Path, dpo_adapter: Path):
    """Base + merged SFT adapter + DPO adapter, ready for generation."""
    from peft import PeftModel

    merged, tokenizer = load_merged_sft(base_name, sft_adapter)
    model = PeftModel.from_pretrained(merged, str(dpo_adapter))
    model.eval()
    return model, tokenizer
