"""Step 5: GRPO. Online RL against the full reward, starting from the SFT model.

Each step takes a few questions, samples a group of answers to each from the current
model, scores them, and updates the model towards the answers that beat their group's
average. Unlike DPO, the answers are fresh every step, so the bar rises as the model
improves.

    R = 1.0 · letter_acc + 0.5 · exact + 0.5 · fluency - format penalties

Fluency is scored by a separate copy of the original model (``scorer``): the policy is
the merged SFT model plus a new LoRA, so switching that LoRA off gives the SFT model,
not the original one.

    train_grpo       resumes from checkpoints; raising ``max_steps`` continues a run
    training_curve   reward per step, read from the latest checkpoint
    eval_snapshots   val scores of the step-50/100/... snapshots, for a val curve
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

from stego.chat import build_messages
from stego.colab import latest_checkpoint
from stego.data.sft_demos import leaks_secret
from stego.experiments.dpo import load_dpo_model, load_merged_sft
from stego.experiments.sft import evaluate
from stego.generation.runner import DEFAULT_TEMPLATE_KWARGS
from stego.io import append_jsonl
from stego.rewards import RewardWeights, completion_text, fluency_score, make_grpo_reward_funcs, score_answer
from stego.rewards.fluency import chat_completion_logprob


def make_dataset(rows: Sequence[dict], tokenizer: Any, template_kwargs: dict | None = None):
    """Prompts rendered as at generation time (terse style, thinking off), plus the
    columns the reward functions need."""
    from datasets import Dataset

    kwargs = DEFAULT_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs
    return Dataset.from_list([
        {
            "prompt": tokenizer.apply_chat_template(
                build_messages(r["question"], r["secret"], r["channel"], style="terse"),
                add_generation_prompt=True, tokenize=False, **kwargs,
            ),
            "secret": r["secret"],
            "channel": r["channel"],
            "question": r["question"],
            "question_id": r["question_id"],
        }
        for r in rows
    ])


class FluencyReward:
    """TRL reward function: fluency of each answer under the original model, against
    that question's sampled reference. Also saves one question's group of answers every
    ``sample_every`` steps to ``samples_path``, for reading during training."""

    __name__ = "fluency"

    def __init__(
        self,
        scorer: Any,
        tokenizer: Any,
        references: dict[str, str],
        samples_path: Path | None = None,
        sample_every: int = 10,
        weights: RewardWeights = RewardWeights(),
        tolerance: float = 1.0,
        template_kwargs: dict | None = None,
    ):
        self.scorer, self.tokenizer, self.references = scorer, tokenizer, references
        self.samples_path, self.sample_every, self.weights, self.tolerance = samples_path, sample_every, weights, tolerance
        self.template_kwargs = DEFAULT_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs
        self._ref_lp: dict[str, float] = {}
        self._logged_step = -1

    def _lp(self, question: str, answer: str) -> float:
        return chat_completion_logprob(self.scorer, self.tokenizer, build_messages(question), answer, **self.template_kwargs)

    def __call__(self, completions, question, question_id, secret, channel, trainer_state=None, **kwargs) -> list[float]:
        texts = [completion_text(c) for c in completions]
        scores = []
        for text, q, qid in zip(texts, question, question_id):
            if qid not in self._ref_lp:
                self._ref_lp[qid] = self._lp(q, self.references[qid])
            scores.append(fluency_score(self._lp(q, text), self._ref_lp[qid], self.tolerance) if text.strip() else 0.0)

        leaks = [leaks_secret(t, s, ch) for t, s, ch in zip(texts, secret, channel)]
        if kwargs.get("log_metric") is not None:  # TRL >= 1.x passes a metric logger
            kwargs["log_metric"]("leak_rate", sum(leaks) / max(len(leaks), 1))

        step = trainer_state.global_step if trainer_state is not None else 0
        if self.samples_path is not None and step % self.sample_every == 0 and step != self._logged_step:
            self._logged_step = step
            group = [i for i, qid in enumerate(question_id) if qid == question_id[0]]
            rows = []
            for i in group:
                b = score_answer(texts[i], secret[i], channel[i], fluency=scores[i], weights=self.weights)
                rows.append({
                    "step": step, "question": question[i], "secret": secret[i], "answer": texts[i],
                    "decoded": b.decode.decoded, "letter_acc": b.decode.letter_acc, "exact": b.decode.exact,
                    "fluency": scores[i], "format_penalty": b.format.penalty, "leak": leaks[i], "reward": b.total,
                })
            append_jsonl(self.samples_path, rows)
        return scores


def snapshot_callback(run_dir: Path, every: int):
    """Trainer callback: copy the adapter at every ``every``-th step to snapshots/step-N,
    so old checkpoints can be pruned while the val curve stays possible."""
    from transformers import TrainerCallback

    out = Path(run_dir) / "snapshots"

    class Snapshots(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if every and state.global_step % every == 0:
                src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                dst = out / f"step-{state.global_step}"
                if src.exists() and not dst.exists():
                    dst.mkdir(parents=True)
                    for name in ("adapter_config.json", "adapter_model.safetensors"):
                        if (src / name).exists():
                            shutil.copy2(src / name, dst / name)
            return control

    return Snapshots()


LIVE_KEYS = (("reward", "reward"), ("letter", "rewards/acrostic_letter_acc/mean"), ("exact", "rewards/acrostic_exact/mean"),
             ("fluency", "rewards/fluency/mean"), ("format", "rewards/format/mean"), ("leak", "leak_rate"),
             ("zero_std", "frac_reward_zero_std"), ("len", "completions/mean_length"), ("sec", "step_time"))


def live_callback(log: Callable = print):
    """Trainer callback: one line per step with the reward terms, since the notebook's
    progress table only shows the (meaningless) GRPO loss."""
    from transformers import TrainerCallback

    class Live(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "reward" in logs:
                parts = [f"{name}={float(logs[key]):.3g}" for name, key in LIVE_KEYS if key in logs]
                log(f"step {state.global_step}/{state.max_steps}  " + "  ".join(parts))
            return control

    return Live()


def _finished_steps(adapter_dir: Path) -> int:
    path = adapter_dir / "trained_steps.json"
    return json.loads(path.read_text())["steps"] if path.exists() else 0


def train_grpo(
    run_dir: Path,
    rows: Sequence[dict],
    model: Any,
    tokenizer: Any,
    scorer: Any,
    fluency_refs: dict[str, str],
    *,
    max_steps: int = 150,
    questions_per_step: int = 4,
    num_generations: int = 8,
    micro_batch: int = 8,
    max_completion_length: int = 192,
    temperature: float = 1.0,
    learning_rate: float = 1e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    save_steps: int = 10,
    snapshot_every: int = 50,
    sample_every: int = 10,
    weights: RewardWeights = RewardWeights(),
    seed: int = 0,
    log: Callable = print,
) -> Path:
    """GRPO with a new LoRA on the merged SFT model (``load_merged_sft``). Resumes from
    the latest checkpoint; calling again with a larger ``max_steps`` continues the run
    (e.g. after a pilot). Returns the saved adapter folder."""
    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    run_dir = Path(run_dir)
    adapter_dir = run_dir / "adapter"
    if _finished_steps(adapter_dir) >= max_steps:
        log(f"{adapter_dir} already trained for {_finished_steps(adapter_dir)} steps")
        return adapter_dir

    completions_per_step = questions_per_step * num_generations
    if completions_per_step % micro_batch:
        raise ValueError("questions_per_step * num_generations must be a multiple of micro_batch")
    half = next(model.parameters()).dtype
    config = GRPOConfig(
        output_dir=str(run_dir / "checkpoints"),
        max_steps=max_steps,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=completions_per_step // micro_batch,  # one generation round per step
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=temperature,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",  # so continuing after a pilot keeps the same rate
        beta=0.0,
        reward_weights=[weights.letter, weights.exact, 1.0, weights.fluency],
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=half == torch.float16,
        bf16=half == torch.bfloat16,
        gradient_checkpointing=torch.cuda.is_available(),
        seed=seed,
        report_to="none",
    )
    fluency = FluencyReward(scorer, tokenizer, fluency_refs, run_dir / "samples.jsonl", sample_every, weights)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=make_grpo_reward_funcs() + [fluency],
        args=config,
        train_dataset=make_dataset(rows, tokenizer),
        processing_class=tokenizer,
        peft_config=LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
        callbacks=[snapshot_callback(run_dir, snapshot_every), live_callback(log)],
    )
    resume = latest_checkpoint(run_dir / "checkpoints")
    if resume:
        log(f"resuming from {resume.name}")
    log(f"training to step {max_steps}: {questions_per_step} questions x {num_generations} answers per step")
    trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    trainer.save_model(str(adapter_dir))
    (adapter_dir / "trained_steps.json").write_text(json.dumps({"steps": trainer.state.global_step}))
    tokenizer.padding_side = "left"
    return adapter_dir


def load_grpo_model(base_name: str, sft_adapter: Path, grpo_adapter: Path):
    """Base + merged SFT adapter + a GRPO adapter (final or a snapshot)."""
    return load_dpo_model(base_name, sft_adapter, grpo_adapter)


# --- Monitoring and evaluation ---------------------------------------------------------


CURVE_KEYS = ("reward", "reward_std", "frac_reward_zero_std", "rewards/acrostic_letter_acc/mean",
              "rewards/acrostic_exact/mean", "rewards/fluency/mean", "rewards/format/mean",
              "leak_rate", "completions/mean_length", "completions/clipped_ratio")


def training_curve(run_dir: Path) -> list[dict]:
    """Per-step training metrics from the latest checkpoint (one row per logged step)."""
    ckpt = latest_checkpoint(Path(run_dir) / "checkpoints")
    if ckpt is None:
        return []
    history = json.loads((ckpt / "trainer_state.json").read_text())["log_history"]
    return [{"step": h["step"], **{k: h[k] for k in CURVE_KEYS if k in h}} for h in history if "reward" in h]


def eval_snapshots(
    run_dir: Path,
    base_name: str,
    sft_adapter: Path,
    rows: Sequence[dict],
    scorer: Any,
    fluency_refs: dict[str, str],
    *,
    batch_size: int = 16,
    log: Callable = print,
) -> list[Path]:
    """Evaluate every snapshot (snapshots/step-N) on ``rows``; writes scored_grpo_stepN.jsonl.
    The merged SFT model is loaded once and each snapshot adapter swapped in and out."""
    from peft import PeftModel

    run_dir = Path(run_dir)
    snaps = sorted((run_dir / "snapshots").glob("step-*"), key=lambda p: int(p.name.split("-")[1]))
    todo = [s for s in snaps if not (run_dir / f"scored_grpo_{s.name.replace('-', '')}.jsonl").exists()]
    if todo:
        merged, tokenizer = load_merged_sft(base_name, sft_adapter)
        for snap in todo:
            log(f"== {snap.name} ==")
            model = PeftModel.from_pretrained(merged, str(snap)).eval()
            evaluate(run_dir, rows, model, tokenizer, fluency_refs, method=f"grpo_{snap.name.replace('-', '')}",
                     scorer=scorer, batch_size=batch_size, log=log)
            merged = model.unload()
    return [run_dir / f"scored_grpo_{s.name.replace('-', '')}.jsonl" for s in snaps]
