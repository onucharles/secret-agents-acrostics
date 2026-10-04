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
    write_jsonl(tmp_path / "sft_candidates.jsonl", cands)
    vocab = AcrosticVocab(tok, model.config.vocab_size)
    out = sft.build_sft_data(tmp_path, exs, model, tok, vocab, samples=2, batch_size=4, score_chunk=4, log=lambda *_: None)

    records = read_jsonl(out)
    assert sorted(r["id"] for r in records) == ["q0-RI", "q2-RI"]  # q1: one leaks, one has wrong letters
    for r in records:
        assert r["messages"][0]["content"].endswith("Secret: RI")
        assert r["messages"][-1]["role"] == "assistant"
    scored = read_jsonl(tmp_path / "sft_candidates_scored.jsonl")
    assert len(scored) == 6
    assert all((r["fluency"] is None) == (r["answer"] in (LEAK, WRONG)) for r in scored)

    # Rerun: nothing is rescored.
    sft.build_sft_data(tmp_path, exs, model, tok, vocab, samples=2, batch_size=4, log=lambda *_: None)
    assert len(read_jsonl(tmp_path / "sft_candidates_scored.jsonl")) == 6


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
    write_jsonl(tmp_path / "sft_train.jsonl", recs)
    adapter = sft.train_sft(tmp_path, tmp_path / "sft_train.jsonl", model, tok, max_steps=2, batch_size=2, grad_accum=1, save_steps=1, log=lambda *_: None)
    assert (adapter / "adapter_config.json").exists()
    assert tok.padding_side == "left"
    # Already trained: a rerun returns straight away.
    assert sft.train_sft(tmp_path, tmp_path / "sft_train.jsonl", model, tok, log=lambda *_: None) == adapter

    trained = PeftModel.from_pretrained(build_model(tok, seed=1), str(adapter)).eval()
    rows = [{**example(i), "split": "val"} for i in range(3)]
    refs = {r["question_id"]: GOOD for r in rows}
    table = sft.evaluate(tmp_path, rows, trained, tok, refs, batch_size=3, log=lambda *_: None)
    assert [t["method"] for t in table] == ["sft"] and table[0]["n"] == 3
    md = sft.results_table(tmp_path / "scored_sft.jsonl")
    assert md.startswith("| method | split |") and "| sft | val | 3 |" in md

    # A separate scorer (the original model) gives the same fluency as switching the adapter off.
    sft.evaluate(tmp_path, rows, trained, tok, refs, method="sft2", scorer=build_model(tok, seed=1), batch_size=3, log=lambda *_: None)
    a, b = read_jsonl(tmp_path / "scored_sft.jsonl"), read_jsonl(tmp_path / "scored_sft2.jsonl")
    assert [r["fluency"] for r in a] == pytest.approx([r["fluency"] for r in b])


def test_training_resumes_from_the_latest_checkpoint(tiny, tmp_path):
    import shutil

    _, tok = tiny
    recs = [{"id": f"q{i}", "messages": [{"role": "system", "content": "Secret: RI"}, {"role": "user", "content": "Hi"},
                                         {"role": "assistant", "content": GOOD}]} for i in range(8)]
    write_jsonl(tmp_path / "sft_train.jsonl", recs)
    sft.train_sft(tmp_path, tmp_path / "sft_train.jsonl", build_model(tok), tok, max_steps=2, batch_size=2, grad_accum=1, save_steps=1, log=lambda *_: None)
    shutil.rmtree(tmp_path / "adapter")  # simulate a disconnect before the final save
    logs = []
    sft.train_sft(tmp_path, tmp_path / "sft_train.jsonl", build_model(tok), tok, max_steps=3, batch_size=2, grad_accum=1, save_steps=1, log=logs.append)
    assert "resuming from checkpoint-2" in logs
    assert (tmp_path / "checkpoints" / "checkpoint-3").exists() and (tmp_path / "adapter" / "adapter_config.json").exists()
