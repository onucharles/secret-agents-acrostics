"""File naming and the migration of run folders written under the old names."""

from stego.io import read_jsonl, write_jsonl
from stego.run_files import migrate_run_dirs, snapshot_name, split_label


def test_split_label():
    assert split_label([{"split": "val"}, {"split": "val"}]) == "val"
    assert split_label([{"split": "val"}, {"split": "test_ood"}]) == "eval"
    assert split_label([{"split": "train"}, {"split": "val"}]) == "mixed"
    assert snapshot_name(50) == "step-0050"


def ref(qid):
    return {"id": f"{qid}#0", "example_id": qid, "question_id": qid, "answer": "a"}


def old_layout(root):
    """Run folders as the code wrote them before the rename."""
    write_jsonl(root / "data" / "examples_acrostic.jsonl",
                [{"question_id": "t1", "split": "train"}, {"question_id": "t2", "split": "train"}, {"question_id": "v1", "split": "val"}])
    e0 = root / "e0-acrostic-Qwen3-1.7B"
    for name in ("references.jsonl", "gen_prompted.jsonl", "scored_no_secret.jsonl", "scored_constrained.jsonl",
                 "sft_candidates_smoke.jsonl", "sft_demos_smoke.jsonl"):
        write_jsonl(e0 / name, [{"x": 1}])
    (e0 / "baseline_table.md").write_text("| method |\n")
    write_jsonl(e0 / "references_sampled.jsonl", [ref("v1"), ref("t1")])

    s = root / "sft-acrostic-Qwen3-1.7B"
    for name in ("sft_candidates.jsonl", "sft_candidates_scored.jsonl", "sft_train.jsonl", "gen_sft.jsonl", "scored_sft.jsonl"):
        write_jsonl(s / name, [{"x": 1}])
    write_jsonl(s / "references_sampled.jsonl", [ref("t1"), ref("t2"), ref("v1")])
    (s / "adapter").mkdir()
    (s / "adapter" / "adapter_config.json").write_text("{}")
    (s / "checkpoints" / "checkpoint-10").mkdir(parents=True)

    d = root / "dpo-acrostic-Qwen3-1.7B"
    for name in ("dpo_samples.jsonl", "dpo_samples_scored.jsonl", "dpo_pairs.jsonl", "scored_dpo.jsonl"):
        write_jsonl(d / name, [{"x": 1}])
    write_jsonl(d / "references_sampled.jsonl", [ref("t1")])

    g = root / "grpo-acrostic-Qwen3-1.7B"
    for name in ("samples.jsonl", "gen_grpo.jsonl", "scored_grpo.jsonl", "gen_grpo_step50.jsonl", "scored_grpo_step50.jsonl"):
        write_jsonl(g / name, [{"x": 1}])
    (g / "snapshots" / "step-50").mkdir(parents=True)
    (g / "snapshots" / "step-150").mkdir(parents=True)
    (g / "adapter").mkdir()
    (g / "adapter" / "trained_steps.json").write_text('{"steps": 150}')


def listing(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def test_dry_run_changes_nothing(tmp_path):
    old_layout(tmp_path)
    before = listing(tmp_path)
    logs = []
    migrate_run_dirs(tmp_path, log=logs.append)
    assert listing(tmp_path) == before
    assert any("train_candidates_constrained.jsonl" in line for line in logs)


def test_migration_renames_everything_and_is_idempotent(tmp_path):
    old_layout(tmp_path)
    migrate_run_dirs(tmp_path, dry_run=False, log=lambda *_: None)
    after = listing(tmp_path)
    assert after == sorted([
        "00_data", "00_data/all_splits_examples_acrostic.jsonl",
        "02_baselines-acrostic-Qwen3-1.7B",
        *(f"02_baselines-acrostic-Qwen3-1.7B/{n}" for n in (
            "eval_answers_no_secret_greedy.jsonl", "eval_answers_prompted.jsonl", "eval_answers_no_secret_scored.jsonl",
            "eval_answers_constrained_scored.jsonl", "eval_baselines_summary_table.md",
            "smoketest_train_candidates_constrained.jsonl", "smoketest_train_sft_records.jsonl",
            "eval_fluency_references_sampled.jsonl", "smoketest_train_fluency_references_sampled.jsonl")),
        "03_sft-acrostic-Qwen3-1.7B",
        *(f"03_sft-acrostic-Qwen3-1.7B/{n}" for n in (
            "train_candidates_constrained.jsonl", "train_candidates_constrained_scored.jsonl", "train_sft_records_selected.jsonl",
            "val_answers_sft_model.jsonl", "val_answers_sft_model_scored.jsonl",
            "train_fluency_references_sampled.jsonl", "val_fluency_references_sampled.jsonl",
            "lora_adapter_final", "lora_adapter_final/adapter_config.json", "lora_checkpoints", "lora_checkpoints/checkpoint-10")),
        "04_dpo-acrostic-Qwen3-1.7B",
        *(f"04_dpo-acrostic-Qwen3-1.7B/{n}" for n in (
            "train_sft_model_samples.jsonl", "train_sft_model_samples_scored.jsonl", "train_dpo_preference_pairs.jsonl",
            "val_answers_dpo_model_scored.jsonl", "train_fluency_references_sampled.jsonl")),
        "05_grpo-acrostic-Qwen3-1.7B",
        *(f"05_grpo-acrostic-Qwen3-1.7B/{n}" for n in (
            "train_rollouts_logged_during_training.jsonl", "val_answers_grpo_model.jsonl", "val_answers_grpo_model_scored.jsonl",
            "val_answers_grpo_model_step0050.jsonl", "val_answers_grpo_model_step0050_scored.jsonl",
            "lora_adapter_snapshots", "lora_adapter_snapshots/step-0050", "lora_adapter_snapshots/step-0150",
            "lora_adapter_final", "lora_adapter_final/trained_steps.json")),
    ])
    sft = tmp_path / "03_sft-acrostic-Qwen3-1.7B"
    assert [r["example_id"] for r in read_jsonl(sft / "train_fluency_references_sampled.jsonl")] == ["t1", "t2"]
    assert [r["example_id"] for r in read_jsonl(sft / "val_fluency_references_sampled.jsonl")] == ["v1"]

    migrate_run_dirs(tmp_path, dry_run=False, log=lambda *_: None)  # a second run finds nothing to do
    assert listing(tmp_path) == after


def test_new_and_old_folders_are_merged_without_overwriting(tmp_path):
    old_layout(tmp_path)
    new = tmp_path / "03_sft-acrostic-Qwen3-1.7B"
    write_jsonl(new / "train_sft_records_selected.jsonl", [{"kept": True}])  # already written by the new code
    logs = []
    migrate_run_dirs(tmp_path, dry_run=False, log=logs.append)
    assert read_jsonl(new / "train_sft_records_selected.jsonl") == [{"kept": True}]
    assert (new / "lora_adapter_final" / "adapter_config.json").exists()
    assert any("already exists" in line for line in logs)
