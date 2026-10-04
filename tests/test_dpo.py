"""Step 4 (DPO): pair building, pilot -> scale-up reuse, training and loading, on the tiny model."""

import pytest

from stego.experiments import dpo

GOOD = "Rivers carry water down to the sea. In spring they often flood the valleys nearby."
LEAK = "Rivers carry water down to the sea. In spring the word RI shows up in the valleys."
WRONG = "Water flows down to the sea every day. In spring it often floods the valleys nearby."


def scored(example_id, answer, letter_acc, exact, fluency=0.5, penalty=0.0, secret="RI"):
    return {"example_id": example_id, "question": f"Q {example_id}?", "secret": secret, "channel": "acrostic",
            "answer": answer, "decoded": "?", "letter_acc": letter_acc, "exact": exact, "fluency": fluency,
            "format_penalty": penalty}


def test_reward_matches_the_grpo_formula():
    assert dpo.reward(scored("a", GOOD, 1.0, True, fluency=0.4, penalty=0.25)) == pytest.approx(1.0 + 0.5 + 0.2 - 0.25)
    assert dpo.reward(scored("a", "", 0.0, False, fluency=None)) == 0.0


def test_make_pairs_best_vs_worst_with_a_gap():
    rows = [
        scored("a", GOOD, 1.0, True), scored("a", WRONG, 0.5, False), scored("a", WRONG + " Extra.", 0.0, False),
        scored("b", GOOD, 0.5, False), scored("b", WRONG, 0.4, False),  # gap 0.1: dropped
        scored("c", LEAK, 1.0, True), scored("c", WRONG, 0.0, False),  # best leaks; WRONG can't be both
    ]
    pairs = dpo.make_pairs(rows, min_gap=0.3)
    assert [p["id"] for p in pairs] == ["a"]
    (p,) = pairs
    assert p["chosen"] == GOOD and p["rejected"] == WRONG + " Extra."
    assert p["prompt"][0]["content"].endswith("Secret: RI") and p["prompt"][-1]["role"] == "user"
    assert [p["id"] for p in dpo.make_pairs(rows, min_gap=0.05)] == ["a", "b"]


torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def example(i, secret="RI"):
    return {"id": f"q{i}-{secret}", "question_id": f"q{i}", "question": f"Tell me about rivers {i}.", "secret": secret,
            "channel": "acrostic", "split": "train", "category": "open_qa", "hard": False}


@pytest.fixture(scope="module")
def tiny():
    from tests.tiny_model import build_model, build_tokenizer

    tok = build_tokenizer()
    return build_model(tok), tok


def test_pilot_then_scale_up_reuses_samples(tiny, tmp_path):
    from stego.io import read_jsonl

    model, tok = tiny
    exs = [example(i) for i in range(4)]
    kw = dict(samples=3, batch_size=6, score_chunk=4, log=lambda *_: None)
    dpo.build_pairs(tmp_path, exs[:2], model, tok, **kw)
    assert len(read_jsonl(tmp_path / "dpo_samples.jsonl")) == 6
    assert len(read_jsonl(tmp_path / "dpo_samples_scored.jsonl")) == 6
    stats = dpo.pair_stats(tmp_path, exs[:2])
    assert stats["questions"] == 2 and stats["samples"] == 6 and stats["pairs"] <= 2

    first = read_jsonl(tmp_path / "dpo_samples.jsonl")
    dpo.build_pairs(tmp_path, exs, model, tok, **kw)
    after = read_jsonl(tmp_path / "dpo_samples.jsonl")
    assert len(after) == 12 and after[:6] == first  # pilot samples kept, only new ones generated
    assert len(read_jsonl(tmp_path / "dpo_samples_scored.jsonl")) == 12
    assert dpo.pair_stats(tmp_path, exs)["questions"] == 4
    assert dpo.pair_stats(tmp_path, exs[:2])["samples"] == 6  # stats can still be read for the pilot only


def test_train_dpo_and_load(tiny, tmp_path):
    pytest.importorskip("trl")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    from stego.io import write_jsonl
    from tests.tiny_model import build_model

    _, tok = tiny
    pairs = [{"id": f"q{i}", "secret": "RI", "channel": "acrostic",
              "prompt": [{"role": "system", "content": "Secret: RI"}, {"role": "user", "content": f"Tell me about rivers {i}."}],
              "chosen": GOOD, "rejected": WRONG, "chosen_reward": 2.0, "rejected_reward": 0.5} for i in range(8)]
    write_jsonl(tmp_path / "dpo_pairs.jsonl", pairs)
    adapter = dpo.train_dpo(tmp_path, tmp_path / "dpo_pairs.jsonl", build_model(tok, seed=2), tok,
                            max_steps=2, batch_size=2, grad_accum=1, save_steps=1, log=lambda *_: None)
    assert (adapter / "adapter_config.json").exists()
    assert tok.padding_side == "left"
    assert dpo.train_dpo(tmp_path, tmp_path / "dpo_pairs.jsonl", None, tok, log=lambda *_: None) == adapter

    # load_dpo_model's merge step: an SFT adapter folds into the base, then the DPO adapter loads on top.
    from peft import PeftModel

    sft_dir = tmp_path / "sft_adapter"
    get_peft_model(build_model(tok, seed=2), LoraConfig(r=4, target_modules="all-linear")).save_pretrained(str(sft_dir))
    merged = PeftModel.from_pretrained(build_model(tok, seed=2), str(sft_dir)).merge_and_unload()
    model = PeftModel.from_pretrained(merged, str(adapter))
    assert model.peft_config["default"].r == 16
