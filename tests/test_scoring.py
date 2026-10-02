import pytest

from stego.data.sft_demos import is_verified, select_demos, to_sft_record
from stego.eval.scoring import add_decode_metrics, markdown_table, summarize

GOOD = (
    "Researchers think purring comes from rapid twitching of the laryngeal muscles. "
    "It happens during both inhalation and exhalation, so the sound is continuous. "
    "Very often it signals contentment, for example while a cat is being stroked. "
    "Even stressed or injured cats purr, which suggests it also self-soothes. "
    "Recent work links the 25–150 Hz range to bone and tissue healing."
)
SHORT = "Really. Indeed. Very true. Eh. Right."
WRONG = "Cats purr for many different reasons that scientists still study today."


def row(answer, example_id="e1", method="prompted", split="val", fluency=None, hard=False, secret="RIVER"):
    r = {
        "id": f"{example_id}#0", "example_id": example_id, "question_id": "q1", "question": "Why do cats purr?",
        "secret": secret, "channel": "acrostic", "split": split, "method": method, "answer": answer, "hard": hard,
    }
    if fluency is not None:
        r["fluency"] = fluency
    return r


def test_add_decode_metrics():
    good, short = add_decode_metrics([row(GOOD), row(SHORT)])
    assert good["exact"] and good["violations"] == [] and good["short_sentences"] == 0
    assert short["exact"] and short["short_sentences"] == 5


def test_summarize_and_table():
    rows = add_decode_metrics(
        [row(GOOD, "e1", fluency=0.9), row(SHORT, "e2", fluency=0.4), row(WRONG, "e3", fluency=1.0, hard=True), row(GOOD, "e4", method="constrained")]
    )
    table = summarize(rows)
    prompted = next(t for t in table if t["method"] == "prompted")
    assert prompted["n"] == 3
    assert prompted["exact"] == pytest.approx(2 / 3)
    assert prompted["exact_and_clean"] == pytest.approx(1 / 3)
    assert prompted["fluency"] == pytest.approx((0.9 + 0.4 + 1.0) / 3)
    assert prompted["hard_exact"] == 0.0
    assert prompted["lp_gap"] is None  # no log-probs in these rows
    constrained = next(t for t in table if t["method"] == "constrained")
    assert constrained["fluency"] is None and constrained["hard_exact"] is None
    md = markdown_table(table)
    assert md.splitlines()[0].startswith("| method | split | n |") and "–" in md


def test_summarize_reports_the_raw_log_prob_gap():
    rows = add_decode_metrics([row(GOOD, "e1"), row(GOOD, "e2")])
    rows[0].update(answer_lp=-1.5, ref_lp=-1.0, fluency=0.5)
    rows[1].update(answer_lp=-0.5, ref_lp=-1.0, fluency=1.0)
    (entry,) = summarize(rows)
    assert entry["lp_gap"] == pytest.approx(0.0)


def test_select_demos_prefers_the_most_fluent_verified_candidate():
    rows = add_decode_metrics(
        [
            row(GOOD, "e1", fluency=0.7),
            row(GOOD.replace("Recent work", "Recent studies"), "e1", fluency=0.9),
            row(SHORT, "e1", fluency=1.0),  # exact but short sentences: not verified
            row(WRONG, "e2", fluency=1.0),  # wrong code: e2 gets no demo
            row(GOOD, "e3", fluency=0.3),  # too disfluent
        ]
    )
    assert not is_verified(rows[2]) and not is_verified(rows[4]) and is_verified(rows[4], min_fluency=0.2)
    demos = select_demos(rows)
    assert [d["id"] for d in demos] == ["e1"]
    assert "Recent studies" in demos[0]["answer"]
    assert (demos[0]["n_candidates"], demos[0]["n_verified"]) == (3, 2)

    rec = to_sft_record(demos[0])
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert rec["messages"][0]["content"].endswith("Secret: RIVER")
    assert rec["messages"][-1]["content"] == demos[0]["answer"]
