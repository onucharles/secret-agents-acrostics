"""Step 3 (SFT) on the tiny local model: data building, a short training run, evaluation."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("trl")
pytest.importorskip("peft")

from stego.experiments import sft  # noqa: E402
from stego.io import read_jsonl, write_jsonl  # noqa: E402
from tests.tiny_model import build_model, build_tokenizer  # noqa: E402

GOOD = "Rivers carry water down to the sea. In spring they often flood the valleys nearby."
LEAK = "Rivers carry water down to the sea. In spring the word RI shows up in the valleys."
WRONG = "Water flows down to the sea every day. In spring it often floods the valleys nearby."


def example(i, secret="RI"):
    return {"id": f"q{i}-{secret}", "question_id": f"q{i}", "question": f"Tell me about rivers {i}.", "secret": secret,
            "channel": "acrostic", "split": "train", "category": "open_qa", "hard": False}


@pytest.fixture(scope="module")
def tiny():
    tok = build_tokenizer()
    return build_model(tok), tok


def test_pick_training_examples_spreads_over_questions():
    exs = [{"id": f"q{q}-{k}", "question_id": f"q{q}", "split": "train"} for q in range(3) for k in range(2)]
    exs.append({"id": "v", "question_id": "q9", "split": "val"})
    assert [e["id"] for e in sft.pick_training_examples(exs, 4)] == ["q0-0", "q1-0", "q2-0", "q0-1"]


def test_build_sft_data_keeps_one_verified_answer_per_example(tiny, tmp_path):
    from stego.generation.constrained import AcrosticVocab

    model, tok = tiny
    exs = [example(0), example(1), example(2)]
    # Pre-filled candidates, so generation is skipped (already "done") and the rest runs for real.
    cands = [
        {**exs[0], "id": "q0-RI#0", "example_id": "q0-RI", "sample": 0, "answer": GOOD},
        {**exs[0], "id": "q0-RI#1", "example_id": "q0-RI", "sample": 1, "answer": GOOD.replace("nearby", "around them")},
        {**exs[1], "id": "q1-RI#0", "example_id": "q1-RI", "sample": 0, "answer": LEAK},
        {**exs[1], "id": "q1-RI#1", "example_id": "q1-RI", "sample": 1, "answer": WRONG},
        {**exs[2], "id": "q2-RI#0", "example_id": "q2-RI", "sample": 0, "answer": WRONG},
        {**exs[2], "id": "q2-RI#1", "example_id": "q2-RI", "sample": 1, "answer": GOOD},
    ]
    write_jsonl(tmp_path / "train_candidates_constrained.jsonl", cands)
    vocab = AcrosticVocab(tok, model.config.vocab_size)
    out = sft.build_sft_data(tmp_path, exs, model, tok, vocab, samples=2, batch_size=4, score_chunk=4, log=lambda *_: None)

    records = read_jsonl(out)
    assert sorted(r["id"] for r in records) == ["q0-RI", "q2-RI"]  # q1: one leaks, one has wrong letters
    for r in records:
        assert r["messages"][0]["content"].endswith("Secret: RI")
        assert r["messages"][-1]["role"] == "assistant"
    scored = read_jsonl(tmp_path / "train_candidates_constrained_scored.jsonl")
    assert len(scored) == 6
    assert all((r["fluency"] is None) == (r["answer"] in (LEAK, WRONG)) for r in scored)

    # Rerun: nothing is rescored.
    sft.build_sft_data(tmp_path, exs, model, tok, vocab, samples=2, batch_size=4, log=lambda *_: None)
    assert len(read_jsonl(tmp_path / "train_candidates_constrained_scored.jsonl")) == 6


def test_to_prompt_completion_matches_the_generation_prompt(tiny):
    _, tok = tiny
    rec = {"messages": [{"role": "system", "content": "Secret: RI"}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": GOOD}]}
    (pc,) = sft.to_prompt_completion([rec], tok)
    assert pc["prompt"] == tok.apply_chat_template(rec["messages"][:2], add_generation_prompt=True, tokenize=False)
    assert pc["completion"] == GOOD


def test_train_then_evaluate(tiny, tmp_path):
    from peft import PeftModel

    _, tok = tiny
    model = build_model(tok, seed=1)
    recs = [
        {"id": f"q{i}", "messages": [{"role": "system", "content": "Secret: RI"}, {"role": "user", "content": f"Tell me about rivers {i}."},
                                     {"role": "assistant", "content": GOOD}]}
        for i in range(8)
    ]
    write_jsonl(tmp_path / "train_sft_records_selected.jsonl", recs)
    adapter = sft.train_sft(tmp_path, tmp_path / "train_sft_records_selected.jsonl", model, tok, max_steps=2, batch_size=2, grad_accum=1, save_steps=1, log=lambda *_: None)
    assert (adapter / "adapter_config.json").exists()
    assert tok.padding_side == "left"
    # Already trained: a rerun returns straight away.
    assert sft.train_sft(tmp_path, tmp_path / "train_sft_records_selected.jsonl", model, tok, log=lambda *_: None) == adapter

    trained = PeftModel.from_pretrained(build_model(tok, seed=1), str(adapter)).eval()
    rows = [{**example(i), "split": "val"} for i in range(3)]
    refs = {r["question_id"]: GOOD for r in rows}
    table = sft.evaluate(tmp_path, rows, trained, tok, refs, batch_size=3, log=lambda *_: None)
    assert [t["method"] for t in table] == ["sft"] and table[0]["n"] == 3
    md = sft.results_table(tmp_path / "val_answers_sft_model_scored.jsonl")
    assert md.startswith("| exp | method | split |") and "| – | sft | val | 3 |" in md

    # A separate scorer (the original model) gives the same fluency as switching the adapter off.
    sft.evaluate(tmp_path, rows, trained, tok, refs, method="sft2", scorer=build_model(tok, seed=1), batch_size=3, log=lambda *_: None)
    a, b = read_jsonl(tmp_path / "val_answers_sft_model_scored.jsonl"), read_jsonl(tmp_path / "val_answers_sft2_model_scored.jsonl")
    assert [r["fluency"] for r in a] == pytest.approx([r["fluency"] for r in b])


def test_training_resumes_from_the_latest_checkpoint(tiny, tmp_path):
    import shutil

    _, tok = tiny
    recs = [{"id": f"q{i}", "messages": [{"role": "system", "content": "Secret: RI"}, {"role": "user", "content": "Hi"},
                                         {"role": "assistant", "content": GOOD}]} for i in range(8)]
    write_jsonl(tmp_path / "train_sft_records_selected.jsonl", recs)
    sft.train_sft(tmp_path, tmp_path / "train_sft_records_selected.jsonl", build_model(tok), tok, max_steps=2, batch_size=2, grad_accum=1, save_steps=1, log=lambda *_: None)
    shutil.rmtree(tmp_path / "lora_adapter_final")  # simulate a disconnect before the final save
    logs = []
    sft.train_sft(tmp_path, tmp_path / "train_sft_records_selected.jsonl", build_model(tok), tok, max_steps=3, batch_size=2, grad_accum=1, save_steps=1, log=logs.append)
    assert "resuming from checkpoint-2" in logs
    assert (tmp_path / "lora_checkpoints" / "checkpoint-3").exists() and (tmp_path / "lora_adapter_final" / "adapter_config.json").exists()


def scored_row(method, secret="RI", split="val", exact=True):
    return {"method": method, "split": split, "secret": secret, "exact": exact, "letter_acc": 1.0 if exact else 0.5,
            "violations": [], "short_sentences": 0, "fluency": 0.5, "answer_lp": -2.0, "ref_lp": -1.5}


def test_results_table_sorts_by_experiment_number(tmp_path):
    files = {
        "06_sft_spelled-x/val_answers_sft_spelled_model_epoch01_scored.jsonl": "sft_spelled_ep1",
        "03_sft-x/val_answers_sft_model_scored.jsonl": "sft",
        "02_baselines-x/eval_answers_prompted_scored.jsonl": "prompted",
        "02_baselines-x/eval_answers_constrained_scored.jsonl": "constrained",
        "05_grpo-x/val_answers_grpo_model_scored.jsonl": "grpo",
    }
    for rel, method in files.items():
        write_jsonl(tmp_path / rel, [scored_row(method), scored_row(method, split="test_ood")])
    md = sft.results_table(*(tmp_path / rel for rel in files))
    body = [line.split(" | ")[:2] for line in md.splitlines()[2:]]
    assert body == [["| 02", "prompted"], ["| 02", "constrained"], ["| 03", "sft"], ["| 05", "grpo"], ["| 06", "sft_spelled_ep1"]]


def test_results_by_length(tmp_path):
    rows = [scored_row("sft", "CAT"), scored_row("sft", "RIVER", exact=False), scored_row("sft", "NORWAY"), scored_row("sft", "ILLINOIS")]
    write_jsonl(tmp_path / "03_sft-x" / "val_answers_sft_model_scored.jsonl", rows)
    md = sft.results_by_length(tmp_path / "03_sft-x" / "val_answers_sft_model_scored.jsonl")
    assert md.startswith("| exp | method | secret_len |")
    assert "| 03 | sft | 3-4 | 1 |" in md and "| 03 | sft | 5-6 | 2 | 0.500 |" in md and "| 03 | sft | 7+ | 1 |" in md


def test_rerender_with_spelled_secret(tmp_path):
    from stego.eval.scoring import add_decode_metrics

    cand = add_decode_metrics([{**example(0), "id": "q0-RI#0", "example_id": "q0-RI", "answer": GOOD}])
    write_jsonl(tmp_path / "scored.jsonl", [{**cand[0], "fluency": 0.3}])
    out = sft.rerender_sft_data(tmp_path / "scored.jsonl", tmp_path / "out.jsonl", style="spelled", log=lambda *_: None)
    (rec,) = read_jsonl(out)
    assert rec["messages"][0]["content"].endswith("Secret: R I") and rec["messages"][-1]["content"] == GOOD


def test_epoch_snapshots_and_stale_answers(tiny, tmp_path):
    _, tok = tiny
    recs = [{"id": f"q{i}", "messages": [{"role": "system", "content": "Secret: R I"}, {"role": "user", "content": "Hi"},
                                         {"role": "assistant", "content": GOOD}]} for i in range(4)]
    write_jsonl(tmp_path / "train.jsonl", recs)
    sft.train_sft(tmp_path, tmp_path / "train.jsonl", build_model(tok, seed=1), tok, epochs=2, batch_size=2, grad_accum=1,
                  epoch_snapshots=True, log=lambda *_: None)
    snaps = sorted(p.name for p in (tmp_path / "lora_adapter_snapshots").iterdir())
    assert snaps == ["epoch-01", "epoch-02"]

    rows = [{**example(i), "split": "val"} for i in range(2)]
    refs = {r["question_id"]: GOOD for r in rows}
    from unittest import mock

    with mock.patch("stego.generation.runner.load_model", lambda name: (build_model(tok, seed=1), tok)):
        paths = sft.eval_epoch_snapshots(tmp_path, "tiny", rows, refs, batch_size=2, log=lambda *_: None)
    assert [p.name for p in paths] == ["val_answers_sft_spelled_model_epoch01_scored.jsonl", "val_answers_sft_spelled_model_epoch02_scored.jsonl"]
    assert all(r["method"] == "sft_spelled_ep1" for r in read_jsonl(paths[0]))

    # Same adapter: answers are reused. A retrained adapter (newer weights): regenerated.
    from peft import PeftModel

    snap = tmp_path / "lora_adapter_snapshots" / "epoch-01"
    gen = tmp_path / "val_answers_sft_spelled_model_epoch01.jsonl"
    write_jsonl(gen, [{**r, "answer": "OLD"} for r in read_jsonl(gen)])
    model = PeftModel.from_pretrained(build_model(tok, seed=1), str(snap))
    kw = dict(method="sft_spelled_ep1", file_name="sft_spelled_model_epoch01", style="spelled", adapter=snap, batch_size=2)
    logs = []
    sft.evaluate(tmp_path, rows, model, tok, refs, log=logs.append, **kw)
    assert all(r["answer"] == "OLD" for r in read_jsonl(gen)) and not any("regenerating" in l for l in logs)
    import os, time

    weights = snap / "adapter_model.safetensors"
    os.utime(weights, (time.time() + 5, time.time() + 5))
    sft.evaluate(tmp_path, rows, model, tok, refs, log=logs.append, **kw)
    assert any("regenerating" in l for l in logs) and all(r["answer"] != "OLD" for r in read_jsonl(gen))
