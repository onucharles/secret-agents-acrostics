import pytest

from stego.eval.errors import align, classify, double_positions, letter_outcomes, looks_degenerate


def ops(secret, decoded):
    return [op for op, _, _ in align(secret, decoded)]


def test_align_counts_a_skip_once_instead_of_shifting_everything():
    tidal = ops("TIDAL", "TIALL")  # D skipped, one L extra (which L is arbitrary)
    assert tidal[:3] == ["match", "match", "skip"] and sorted(tidal[3:]) == ["extra", "match", "match"]
    assert ops("TALE", "TAME") == ["match", "match", "sub", "match"]
    assert sorted(ops("DEF", "DEEF")) == ["extra", "match", "match", "match"]
    assert ops("", "AB") == ["extra", "extra"] and ops("AB", "") == ["skip", "skip"]


@pytest.mark.parametrize("secret, decoded, kind", [
    ("RIVER", "RIVER", "exact"),
    ("BELGIUM", "BELGIU", "stopped_early"),
    ("MUG", "MUGS", "ran_on"),
    ("TALE", "TAME", "one_wrong"),
    ("SLEEVES", "SLEVES", "one_skipped"),  # the doubled E written once
    ("DEF", "DEEF", "one_extra"),
    ("ARROWS", "AROWS", "one_skipped"),
    ("CITRUS", "CHWRUS", "several"),
])
def test_classify_kinds(secret, decoded, kind):
    assert classify(secret, decoded)["kind"] == kind


def test_classify_details():
    c = classify("ARROWS", "AROWS")
    assert (c["first_error"], c["error_near_double"], c["length_diff"], c["skips"]) == (2, True, -1, 1)
    c = classify("TALE", "TAME")
    assert (c["first_error"], c["error_near_double"]) == (3, False)
    assert classify("MUG", "MUGS")["first_error"] == 3  # an extra at the end counts at the last position


def test_double_positions_and_letter_outcomes():
    assert double_positions("ILLINOIS") == {1, 2}
    rows = letter_outcomes("ILLINOIS", "ILLLN")
    assert [r["position"] for r in rows] == list(range(1, 9))
    assert [r["double"] for r in rows][:4] == [False, True, True, False]
    assert sum(r["hit"] for r in rows) == 4
    assert letter_outcomes("QUIZ", "QUIZ")[0]["rare"] and not letter_outcomes("QUIZ", "QUIZ")[1]["rare"]


def test_looks_degenerate():
    assert looks_degenerate("E is for Executive, R is for Representative.")
    assert looks_degenerate("Cats purr when happy. Dogs bark. Cats purr when happy.")
    assert not looks_degenerate("Cats purr when happy. Dogs bark at strangers.")


def test_build_probe_rows_groups_and_no_training_secrets():
    from stego.eval.errors import build_probe_rows, probe_group

    examples = [{"question_id": f"q{i}", "question": f"Q{i}?", "secret": "JAZZ", "split": "train"} for i in range(3)]
    examples += [{"question_id": f"v{i}", "question": f"V{i}?", "secret": "RIVER", "split": "val"} for i in range(5)]
    words = ["JAZZ", "QUIZ", "OXEN", "JOKE", "BALL", "TREE", "ADDS", "LOOP", "ROAD", "MINT", "CART", "SOFA", "X1"]
    rows = build_probe_rows(examples, words, n_per_group=3)
    groups = [r["group"] for r in rows]
    assert groups == ["rare"] * 3 + ["double"] * 3 + ["control"] * 3
    assert "JAZZ" not in {r["secret"] for r in rows}  # a training secret
    assert all(probe_group(r["secret"]) == r["group"] for r in rows)
    assert {r["question_id"] for r in rows} <= {f"v{i}" for i in range(5)} and all(r["split"] == "probe" for r in rows)
    assert sorted(len(r["secret"]) for r in rows if r["group"] == "control") == sorted(len(r["secret"]) for r in rows if r["group"] == "rare")
