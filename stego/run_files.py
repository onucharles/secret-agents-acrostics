"""Names of the folders and files the experiments write, plus a one-off migration for
run folders written under the old names.

Files are named ``<split>_<content>_<producer>[_scored].jsonl``. The split comes first,
so a folder listing groups files by the questions they cover:

    train            training questions
    val              validation questions
    eval             all four held-out splits together (val, test_words, test_long, test_ood)
    smoketest_train  the few training questions of the phase 2 smoke test

Folders are numbered after the notebook that writes them:

    00_data/                          all_splits_examples_<channel>.jsonl
    02_baselines-<channel>-<model>/   eval_* baselines, smoketest_train_* demos
    03_sft-<channel>-<model>/         train_* SFT data, lora_*, val_answers_sft_model*
    04_dpo-<channel>-<model>/         train_* DPO pairs, lora_*, val_answers_dpo_model*
    05_grpo-<channel>-<model>/        train_rollouts_*, lora_*, val_answers_grpo_model*
    06_sft_spelled-<channel>-<model>/ train_sft_records_spelled, lora_*, val_answers_sft_spelled_model*
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Sequence

from stego.io import read_jsonl, write_jsonl

EVAL_SPLITS = ("val", "test_words", "test_long", "test_ood")

DATA_DIR = "00_data"
STAGE_DIRS = {"baselines": "02_baselines", "sft": "03_sft", "dpo": "04_dpo", "grpo": "05_grpo", "sft_spelled": "06_sft_spelled"}

LORA_CHECKPOINTS = "lora_checkpoints"
LORA_ADAPTER = "lora_adapter_final"
LORA_SNAPSHOTS = "lora_adapter_snapshots"


def stage_dir_name(stage: str, channel: str, model: str) -> str:
    """Run folder name, e.g. ``03_sft-acrostic-Qwen3-1.7B``."""
    return f"{STAGE_DIRS[stage]}-{channel}-{model}"


def examples_file(channel: str) -> str:
    return f"all_splits_examples_{channel}.jsonl"


def split_label(rows: Sequence[dict]) -> str:
    """The file-name prefix for ``rows``: their split if they share one, ``eval`` for a
    mix of held-out splits, otherwise ``mixed``."""
    splits = {r["split"] for r in rows}
    if len(splits) == 1:
        return splits.pop()
    return "eval" if splits and splits <= set(EVAL_SPLITS) else "mixed"


def references_file(split: str, sampled: bool) -> str:
    """Greedy references are the ``no_secret`` answers; sampled ones are what fluency is
    measured against."""
    return f"{split}_fluency_references_sampled.jsonl" if sampled else f"{split}_answers_no_secret_greedy.jsonl"


def snapshot_name(step: int) -> str:
    """Zero-padded so snapshot folders sort by step."""
    return f"step-{step:04d}"


def snapshot_step(path: Path) -> int:
    return int(Path(path).name.split("-")[1])


def epoch_snapshot_name(epoch: int) -> str:
    """Per-epoch snapshot folder (SFT); parse it back with ``snapshot_step``."""
    return f"epoch-{epoch:02d}"


def experiment_number(path: str | Path) -> str | None:
    """The experiment a run file belongs to, from its folder name: ``03`` for
    ``03_sft-acrostic-Qwen3-1.7B/val_answers_sft_model_scored.jsonl``."""
    m = re.match(r"(\d+)_", Path(path).parent.name)
    return m[1] if m else None


# --- Migration from the old names ------------------------------------------------------

OLD_STAGE_PREFIXES = {"e0-": "baselines", "sft-": "sft", "dpo-": "dpo", "grpo-": "grpo"}


def _eval_renames(name: str) -> str | None:
    """02_baselines: every generated file covers all eval splits."""
    fixed = {
        "references.jsonl": references_file("eval", sampled=False),
        "baseline_table.md": "eval_baselines_summary_table.md",
        "sft_candidates_smoke.jsonl": "smoketest_train_candidates_constrained.jsonl",
        "sft_demos_smoke.jsonl": "smoketest_train_sft_records.jsonl",
    }
    if name in fixed:
        return fixed[name]
    if m := re.fullmatch(r"gen_(\w+)\.jsonl", name):
        return f"eval_answers_{m[1]}.jsonl"
    if m := re.fullmatch(r"scored_(\w+)\.jsonl", name):
        return f"eval_answers_{m[1]}_scored.jsonl"
    return None


def _val_answers_renames(name: str) -> str | None:
    """03-05: gen_/scored_ files are the trained model's answers to val questions."""
    m = re.fullmatch(r"(gen|scored)_([a-z]+?)(?:_step(\d+))?\.jsonl", name)
    if not m:
        return None
    step = f"_step{int(m[3]):04d}" if m[3] else ""
    return f"val_answers_{m[2]}_model{step}{'_scored' if m[1] == 'scored' else ''}.jsonl"


STAGE_RENAMES: dict[str, Callable[[str], str | None]] = {
    "baselines": _eval_renames,
    "sft": lambda n: {
        "sft_candidates.jsonl": "train_candidates_constrained.jsonl",
        "sft_candidates_scored.jsonl": "train_candidates_constrained_scored.jsonl",
        "sft_train.jsonl": "train_sft_records_selected.jsonl",
    }.get(n) or _val_answers_renames(n),
    "dpo": lambda n: {
        "dpo_samples.jsonl": "train_sft_model_samples.jsonl",
        "dpo_samples_scored.jsonl": "train_sft_model_samples_scored.jsonl",
        "dpo_pairs.jsonl": "train_dpo_preference_pairs.jsonl",
    }.get(n) or _val_answers_renames(n),
    "grpo": lambda n: {"samples.jsonl": "train_rollouts_logged_during_training.jsonl"}.get(n) or _val_answers_renames(n),
}

# Where the train and non-train halves of an old references_sampled.jsonl go.
REFERENCE_SPLITS = {
    "baselines": ("smoketest_train", "eval"),
    "sft": ("train", "val"),
    "dpo": ("train", "val"),
    "grpo": ("train", "val"),
}


def _move(src: Path, dst: Path, dry_run: bool, log: Callable) -> None:
    """Rename ``src`` to ``dst``. Folders that both exist are merged; an existing file
    at ``dst`` is never overwritten."""
    if src == dst:
        return
    if dst.is_dir() and src.is_dir():
        for child in sorted(src.iterdir()):
            _move(child, dst / child.name, dry_run, log)
        if not dry_run and not any(src.iterdir()):
            src.rmdir()
        return
    if dst.exists():
        log(f"  skip {src} -> {dst.name}: target already exists")
        return
    log(f"  {src.parent.name}/{src.name} -> {dst.name}")
    if not dry_run:
        src.rename(dst)


def _split_references(path: Path, stage: str, train_qids: set[str], dry_run: bool, log: Callable) -> None:
    train_name, other_name = (references_file(s, sampled=True) for s in REFERENCE_SPLITS[stage])
    rows = read_jsonl(path)
    parts = {train_name: [r for r in rows if r["example_id"] in train_qids],
             other_name: [r for r in rows if r["example_id"] not in train_qids]}
    parts = {name: part for name, part in parts.items() if part}
    clash = [name for name in parts if (path.parent / name).exists()]
    if clash:
        log(f"  skip {path.parent.name}/{path.name}: {', '.join(clash)} already exists")
        return
    for name, part in parts.items():
        log(f"  {path.parent.name}/{path.name} -> {name} ({len(part)} rows)")
    if not dry_run:
        for name, part in parts.items():
            write_jsonl(path.parent / name, part)
        path.unlink()


def migrate_run_dirs(root: str | Path | None = None, dry_run: bool = True, log: Callable = print) -> None:
    """Rename run folders and files written under the old names to the current ones.

    ``root`` is the folder holding the runs (default: the same place ``run_dir`` uses,
    ``MyDrive/stego-runs`` on Colab). With ``dry_run=True`` (the default) it only prints
    what it would do; run it again with ``dry_run=False`` to apply. Safe to rerun: it
    never overwrites a file, and already-migrated folders are left alone.

    Old ``references_sampled.jsonl`` files mixed training and held-out questions; they
    are split in two using the training question ids from the examples file.
    """
    from stego.colab import in_colab, mount_drive

    if root is None:
        root = mount_drive() / "stego-runs" if in_colab() else Path("runs")
    root = Path(root)
    log(f"{'DRY RUN: ' if dry_run else ''}migrating {root}")

    if (root / "data").is_dir():
        log("data/")
        for f in sorted((root / "data").glob("examples_*.jsonl")):
            _move(f, f.parent / examples_file(f.name[len("examples_"):-len(".jsonl")]), dry_run, log)
        _move(root / "data", root / DATA_DIR, dry_run, log)

    data_dir = root / DATA_DIR if (root / DATA_DIR).is_dir() else root / "data"
    train_qids = {e["question_id"] for f in data_dir.glob("*examples_*.jsonl") for e in read_jsonl(f) if e["split"] == "train"}

    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        stage = next((s for prefix, s in OLD_STAGE_PREFIXES.items() if folder.name.startswith(prefix)), None)
        if stage is None:
            continue
        log(f"{folder.name}/")
        for child in sorted(folder.iterdir()):
            new = STAGE_RENAMES[stage](child.name) if child.is_file() else None
            if child.name == "references_sampled.jsonl":
                if not train_qids:
                    log(f"  skip {child.name}: no examples file found in {data_dir} to tell train from held-out questions")
                    continue
                _split_references(child, stage, train_qids, dry_run, log)
            elif new:
                _move(child, folder / new, dry_run, log)
            elif child.is_dir() and child.name == "checkpoints":
                _move(child, folder / LORA_CHECKPOINTS, dry_run, log)
            elif child.is_dir() and child.name == "adapter":
                _move(child, folder / LORA_ADAPTER, dry_run, log)
            elif child.is_dir() and child.name == "snapshots":
                for snap in sorted(child.glob("step-*")):
                    _move(snap, child / snapshot_name(snapshot_step(snap)), dry_run, log)
                _move(child, folder / LORA_SNAPSHOTS, dry_run, log)
        new_folder = root / (STAGE_DIRS[stage] + folder.name[folder.name.index("-"):])
        _move(folder, new_folder, dry_run, log)
    log("done" if not dry_run else "dry run only; call again with dry_run=False to apply")
