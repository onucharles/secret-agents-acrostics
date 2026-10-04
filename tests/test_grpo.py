"""Step 5 (GRPO) on the tiny local model: reward function, pilot then continue, snapshots, curves."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")
pytest.importorskip("peft")

from stego.experiments import grpo  # noqa: E402
from stego.io import read_jsonl  # noqa: E402
from tests.tiny_model import build_model, build_tokenizer  # noqa: E402

GOOD = "Rivers carry water down to the sea. In spring they often flood the valleys nearby."


def example(i, secret="RI"):
    return {"id": f"q{i}-{secret}", "question_id": f"q{i}", "question": f"Tell me about rivers {i}.", "secret": secret,
            "channel": "acrostic", "split": "train"}


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer()


def test_fluency_reward_scores_and_logs_one_group(tok, tmp_path):
    scorer = build_model(tok)
    f = grpo.FluencyReward(scorer, tok, {"q0": GOOD, "q1": GOOD}, samples_path=tmp_path / "samples.jsonl", sample_every=10)
    state = type("S", (), {"global_step": 10})()
    kw = dict(question=["Q0", "Q0", "Q1"], question_id=["q0", "q0", "q1"], secret=["RI"] * 3, channel=["acrostic"] * 3)
    metrics = {}
    scores = f([GOOD, "", "The word RI is hidden here."], trainer_state=state, log_metric=metrics.__setitem__, **kw)
    assert scores[0] == pytest.approx(1.0) and scores[1] == 0.0  # the reference itself scores 1; empty scores 0
    assert metrics["leak_rate"] == pytest.approx(1 / 3)
    logged = read_jsonl(tmp_path / "samples.jsonl")
    assert [r["question"] for r in logged] == ["Q0", "Q0"] and logged[0]["exact"] and logged[0]["step"] == 10
    f([GOOD], trainer_state=state, **{k: v[:1] for k, v in kw.items()})  # same step: not logged twice
    state.global_step = 11
    f([GOOD], trainer_state=state, **{k: v[:1] for k, v in kw.items()})  # not a logging step
    assert len(read_jsonl(tmp_path / "samples.jsonl")) == 2


def test_pilot_then_continue(tok, tmp_path):
    rows = [example(i) for i in range(6)]
    refs = {r["question_id"]: GOOD for r in rows}
    scorer = build_model(tok)
    kw = dict(questions_per_step=2, num_generations=2, micro_batch=2, max_completion_length=12,
              save_steps=1, snapshot_every=2, sample_every=1, log=lambda *_: None)

    adapter = grpo.train_grpo(tmp_path, rows, build_model(tok, seed=1), tok, scorer, refs, max_steps=2, **kw)
    assert (adapter / "adapter_config.json").exists()
    assert (tmp_path / "snapshots" / "step-2" / "adapter_model.safetensors").exists()
    curve = grpo.training_curve(tmp_path)
    assert [c["step"] for c in curve] == [1, 2] and "rewards/fluency/mean" in curve[0]
    assert read_jsonl(tmp_path / "samples.jsonl")

    logs = []
    assert grpo.train_grpo(tmp_path, rows, None, tok, scorer, refs, max_steps=2, log=logs.append, **{k: v for k, v in kw.items() if k != "log"}) == adapter
    assert "already trained" in logs[0]

    # Raising max_steps continues from the pilot's checkpoint.
    logs = []
    grpo.train_grpo(tmp_path, rows, build_model(tok, seed=1), tok, scorer, refs, max_steps=4, log=logs.append,
                    **{k: v for k, v in kw.items() if k != "log"})
    assert "resuming from checkpoint-2" in logs
    assert [c["step"] for c in grpo.training_curve(tmp_path)] == [1, 2, 3, 4]
    assert sorted(p.name for p in (tmp_path / "snapshots").iterdir()) == ["step-2", "step-4"]
