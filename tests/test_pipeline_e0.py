"""Phase 2 pipeline end to end on the tiny local model, with synthetic questions."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from stego.experiments import baselines as e0  # noqa: E402
from stego.io import read_jsonl  # noqa: E402
from tests.tiny_model import build_model, build_tokenizer  # noqa: E402


def fake_dolly():
    topics = ["rivers", "cats", "stars", "bread", "trains", "volcanoes", "bees", "tides", "clouds", "maps"]
    records = []
    for i in range(60):
        records.append({"instruction": f"Why do {topics[i % 10]} matter, part {i}?", "context": "", "category": "open_qa"})
    for i in range(20):
        records.append({"instruction": f"What are some ideas for a day out, variant {i}?", "context": "", "category": "brainstorming"})
    return records


def test_e0_pipeline_runs_and_resumes(tmp_path, monkeypatch):
    from stego.generation.constrained import AcrosticVocab

    monkeypatch.setattr(e0, "load_dolly", fake_dolly)
    tok = build_tokenizer()
    model = build_model(tok)
    vocab = AcrosticVocab(tok, model.config.vocab_size)
    quiet = lambda *_: None  # noqa: E731

    path = e0.build_datasets(tmp_path, n_eval=4, log=quiet)
    examples = read_jsonl(path)
    rows = e0.eval_rows(examples, n_per_split=2)
    assert len(rows) == 8 and {r["split"] for r in rows} == set(e0.EVAL_SPLITS)

    refs = e0.generate_references(tmp_path, rows, model, tok, batch_size=4, log=quiet)
    assert set(refs) == {r["question_id"] for r in rows}

    e0.run_baselines(tmp_path, rows, model, tok, vocab, batch_size=4, log=quiet)
    table = e0.score_baselines(tmp_path, rows, refs, model, tok)
    methods = {t["method"] for t in table}
    assert methods == {"no_secret", "prompted", "constrained"}
    # A random model rarely ends sentences, so it can't finish secrets here (the
    # constraint itself is tested in test_constrained.py). What must hold is that the
    # constraint ran: every answer opens with its secret's first letter, on one line.
    for r in read_jsonl(tmp_path / "scored_constrained.jsonl"):
        assert r["decoded"][:1] == r["secret"][0] and "\n" not in r["answer"]
    assert (tmp_path / "baseline_table.md").read_text().startswith("| method | split |")
    assert "secret=" in e0.show_examples(tmp_path, "constrained", n=2)

    # Rerunning skips finished work instead of regenerating it.
    logs = []
    e0.run_baselines(tmp_path, rows, model, tok, vocab, batch_size=4, log=logs.append)
    assert any("8 done, 0 to generate" in line for line in logs)

    train = [e for e in examples if e["split"] == "train"]
    demos = e0.sft_demo_smoke_test(tmp_path, train, model, tok, vocab, n_prompts=3, samples=2, min_fluency=0.0, batch_size=4, log=quiet)
    for d in demos:
        assert d["exact"] and d["n_candidates"] == 2
