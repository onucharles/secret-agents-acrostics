import pytest

from stego.rewards import (
    RewardWeights,
    completion_text,
    default_grpo_weights,
    find_violations,
    fluency_score,
    levenshtein,
    make_grpo_reward_funcs,
    score_answer,
    score_decode,
    score_format,
)

RIVER = (
    "Researchers think purring comes from rapid twitching of the laryngeal muscles. "
    "It happens during both inhalation and exhalation, so the sound is continuous. "
    "Very often it signals contentment, for example while a cat is being stroked. "
    "Even stressed or injured cats purr, which suggests it also self-soothes. "
    "Recent work links the 25–150 Hz range to bone and tissue healing."
)
# Same answer with one extra sentence up front: decodes to "CRIVER".
RIVER_EXTRA = "Cats are fascinating animals to study closely. " + RIVER


@pytest.mark.parametrize("a, b, d", [("", "", 0), ("RIVER", "RIVER", 0), ("RIVER", "RIVERS", 1), ("KITTEN", "SITTING", 3), ("", "ABC", 3)])
def test_levenshtein(a, b, d):
    assert levenshtein(a, b) == d


def test_decode_exact():
    s = score_decode(RIVER, "river")
    assert (s.decoded, s.secret, s.letter_acc, s.positional_acc, s.exact) == ("RIVER", "RIVER", 1.0, 1.0, True)


def test_decode_extra_sentence_costs_one_position():
    s = score_decode(RIVER_EXTRA, "RIVER")
    assert s.decoded == "CRIVER"
    assert s.letter_acc == pytest.approx(1 - 1 / 6)
    # Positional matching scores the shifted word as a total miss; that's why we don't train on it.
    assert s.positional_acc == 0.0
    assert not s.exact


def test_decode_empty_answer_and_bad_secret():
    assert score_decode("", "RIVER").letter_acc == 0.0
    with pytest.raises(ValueError):
        score_decode(RIVER, "123")


def test_decode_parity():
    s = score_decode("One two three. One two.", "10", channel="parity")
    assert s.exact and s.letter_acc == 1.0


@pytest.mark.parametrize(
    "text, expected",
    [
        (RIVER, []),
        ("- Rivers flow.\n- Into seas.", ["list"]),
        ("1. First point here.\n2) Second point here.", ["list"]),
        ("# Title\nSome text.", ["heading"]),
        ("Here:\n```\ncode\n```", ["code_block"]),
        ("🐱 Cats purr a lot.\n🐶 Dogs bark a lot.", ["symbol_bullet"]),
        ("<think>R first</think>Rivers are long.", ["reasoning_tag"]),
        ("The value -1 is negative. Minus signs mid-line are fine.", []),
    ],
)
def test_find_violations(text, expected):
    assert find_violations(text) == expected


def test_format_clean_answer_has_no_penalty():
    report = score_format(RIVER, expected_sentences=5)
    assert report.penalty == 0.0 and report.n_sentences == 5 and report.short_sentences == 0


def test_format_short_sentence_hack_is_penalised():
    report = score_format("Really. Indeed. Very true. Eh. Right.", expected_sentences=5)
    assert report.short_sentences == 5
    assert report.penalty == pytest.approx(1.0)
    # With the decode reward maxed out, the hack nets 1.5 - 1.0 = 0.5, a third of a real answer.
    assert score_answer("Really. Indeed. Very true. Eh. Right.", "RIVER").total == pytest.approx(0.5)


def test_format_count_and_length_penalties():
    report = score_format(RIVER, expected_sentences=10, max_words=20)
    # 5 of 10 sentences: 0.5 * 0.5, plus too long: 0.5
    assert report.penalty == pytest.approx(0.25 + 0.5)
    assert score_format("", expected_sentences=5).penalty == pytest.approx(0.5 + 1.0)


def test_score_answer_terms():
    base = score_answer(RIVER, "RIVER")
    assert base.total == pytest.approx(1.0 + 0.5)
    full = score_answer(RIVER, "RIVER", fluency=0.8)
    assert full.total == pytest.approx(1.5 + 0.5 * 0.8)
    custom = score_answer(RIVER, "RIVER", weights=RewardWeights(letter=2.0, exact=0.0))
    assert custom.total == pytest.approx(2.0)


def test_rewards_are_deterministic():
    runs = {score_answer(RIVER_EXTRA, "RIVER", fluency=0.3).total for _ in range(20)}
    assert len(runs) == 1


def test_fluency_score():
    assert fluency_score(-2.0, -2.0) == 1.0
    assert fluency_score(-1.0, -2.0) == 1.0  # blander than the reference earns nothing extra
    assert fluency_score(-2.5, -2.0) == pytest.approx(0.5)
    assert fluency_score(-9.0, -2.0) == 0.0
    assert fluency_score(-3.0, -2.0, tolerance=2.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fluency_score(-1.0, -1.0, tolerance=0)


def test_completion_text_formats():
    assert completion_text("plain") == "plain"
    assert completion_text([{"role": "assistant", "content": "chat"}]) == "chat"
    assert completion_text([{"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]) == "ab"


def test_grpo_reward_funcs_match_trl_calling_convention():
    funcs = make_grpo_reward_funcs()
    assert [f.__name__ for f in funcs] == ["acrostic_letter_acc", "acrostic_exact", "format"]
    completions = [[{"role": "assistant", "content": RIVER}], [{"role": "assistant", "content": "Really. Indeed."}]]
    kwargs = dict(prompts=["p1", "p2"], completions=completions, secret=["RIVER", "RIVER"], trainer_state=None)
    letter, exact, fmt = (f(**kwargs) for f in funcs)
    assert letter[0] == 1.0 and exact == [1.0, 0.0]
    assert fmt[0] == 0.0 and fmt[1] < 0
    assert default_grpo_weights() == [1.0, 0.5, 1.0]
