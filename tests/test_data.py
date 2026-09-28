from collections import Counter

import pytest

from stego.chat import _EXAMPLE, build_messages, system_prompt
from stego.data.examples import (
    SPLITS,
    Question,
    build_examples,
    is_usable_question,
    load_words,
    questions_from_records,
    split_words,
)
from stego.rewards import score_answer


@pytest.mark.parametrize(
    "q, ok",
    [
        ("Why do cats purr?", True),
        ("What causes the seasons on Earth?", True),
        ("List five fruits that are red.", False),
        ("Write a Python function to reverse a string.", False),
        ("Classify these animals as mammals or birds.", False),
        ("Write a poem about the sea.", False),
        ("Hi?", False),  # too short
        ("Qu'est-ce que c'est, café?", False),  # non-ASCII
    ],
)
def test_is_usable_question(q, ok):
    assert is_usable_question(q) is ok


def test_context_questions_are_dropped():
    assert not is_usable_question("What does the passage say about rivers?", context="Rivers are long.")


def test_questions_from_records_filters_and_dedupes():
    records = [
        {"instruction": "Why do cats purr?", "context": "", "category": "open_qa"},
        {"instruction": "why do cats purr?", "context": "", "category": "general_qa"},  # duplicate
        {"instruction": "Why is the sky blue?", "context": "", "category": "closed_qa"},  # wrong category
        {"instruction": "How do tides work?", "context": "", "category": "general_qa"},
    ]
    qs = questions_from_records(records, ["open_qa", "general_qa"])
    assert [(q.id, q.text) for q in qs] == [("dolly-0", "Why do cats purr?"), ("dolly-3", "How do tides work?")]


def test_word_pool():
    words = load_words()
    assert len(words) > 10_000
    assert all(w.isalpha() and w.isupper() and 3 <= len(w) <= 12 for w in words)
    ws = split_words(words)
    assert not set(ws.train) & set(ws.held_out)
    assert all(3 <= len(w) <= 8 for w in ws.train + ws.held_out)
    assert all(9 <= len(w) <= 12 for w in ws.long)
    assert split_words(words) == ws  # deterministic


def fake_questions(n, category, prefix):
    return [Question(id=f"{prefix}-{i}", text=f"Question number {i} about {category}?", category=category) for i in range(n)]


@pytest.fixture(scope="module")
def examples():
    return build_examples(fake_questions(600, "open_qa", "in"), fake_questions(300, "brainstorming", "ood"), load_words(), n_eval=50)


def test_splits_sizes_and_disjointness(examples):
    by_split = Counter(e.split for e in examples)
    assert set(by_split) == set(SPLITS)
    assert all(by_split[s] == 50 for s in SPLITS if s != "train")
    assert by_split["train"] > 800

    train_q = {e.question_id for e in examples if e.split == "train"}
    for s in ("val", "test_words", "test_long"):
        assert not train_q & {e.question_id for e in examples if e.split == s}
    assert all(e.question_id.startswith("ood") for e in examples if e.split == "test_ood")

    train_words = {e.secret for e in examples if e.split == "train"}
    assert not train_words & {e.secret for e in examples if e.split in ("test_words", "test_long")}
    assert all(len(e.secret) >= 9 for e in examples if e.split == "test_long")
    assert len({e.id for e in examples}) == len(examples)


def test_build_is_deterministic(examples):
    again = build_examples(fake_questions(600, "open_qa", "in"), fake_questions(300, "brainstorming", "ood"), load_words(), n_eval=50)
    assert again == examples


def test_hard_flag(examples):
    for e in examples:
        assert e.hard == bool(set("QXZY") & set(e.secret))
    assert any(e.hard for e in examples)


def test_parity_examples():
    exs = build_examples(fake_questions(100, "open_qa", "in"), fake_questions(50, "brainstorming", "ood"), load_words(), channel="parity", n_eval=10)
    assert all(set(e.secret) <= {"0", "1"} and not e.hard for e in exs)


def test_prompt_styles():
    plain = build_messages("Why?")
    assert [m["role"] for m in plain] == ["system", "user"] and "RIVER" not in plain[0]["content"]
    instructed = system_prompt("instructed", "acrostic", "river")
    assert "spell RIVER" in instructed and "sentence 5 starts with 'R'" in instructed
    assert system_prompt("terse", "acrostic", "river").endswith("Secret: RIVER")
    assert "sentence 2 has an even number of words" in system_prompt("instructed", "parity", "101")
    shot = build_messages("Why?", "RIVER", style="instructed", few_shot=True)
    assert [m["role"] for m in shot] == ["system", "user", "assistant", "user"]
    with pytest.raises(ValueError):
        system_prompt("terse", "acrostic", None)


@pytest.mark.parametrize("channel", ["acrostic", "parity"])
def test_few_shot_examples_follow_their_own_rule(channel):
    _, secret, answer = _EXAMPLE[channel]
    r = score_answer(answer, secret, channel=channel)
    assert r.decode.exact and r.format.penalty == 0
