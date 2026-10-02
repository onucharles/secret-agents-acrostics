"""Teacher-data generation against a fake Anthropic client (no network, no cost)."""

from types import SimpleNamespace as NS

import pytest

from stego.data import teacher
from stego.io import read_jsonl

WORDS = {"R": "Rivers", "I": "In", "V": "Valleys", "E": "Every", "Q": "Quiet", "U": "Under", "Z": "Zebras", "A": "Almost", "B": "Bees", "C": "Cats"}


def good_answer(secret):
    return " ".join(f"{WORDS[c]} carry a lot of interesting facts today." for c in secret)


def message(text, stop="end_turn"):
    usage = NS(input_tokens=200, output_tokens=300, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return NS(content=[NS(type="thinking", thinking=""), NS(type="text", text=text)], stop_reason=stop, model="claude-opus-5-5", usage=usage)


class FakeClient:
    """Answers correctly, except: the first attempt for QUIZ spells the wrong word, and
    every attempt for CAB names the secret (a leak)."""

    def __init__(self):
        self.calls = []
        self.messages = NS(create=self._create, batches=NS(create=self._batch_create, retrieve=self._retrieve, results=self._results))
        self.beta = NS(messages=NS(create=self._beta_create))
        self._batches = {}

    def _answer(self, params):
        secret = params["system"].split("must spell ")[1].split(" ")[0]
        attempt = sum(1 for c in self.calls if c == secret) - 1
        if secret == "QUIZ" and attempt == 0:
            return good_answer("RIVE")
        if secret == "CAB":
            return good_answer(secret).replace("facts today.", "facts, like CAB.", 1)
        return good_answer(secret)

    def _create(self, **params):
        secret = params["system"].split("must spell ")[1].split(" ")[0]
        self.calls.append(secret)
        return message(self._answer(params))

    def _beta_create(self, **params):
        assert params.pop("fallbacks") == "default" and params.pop("betas") == [teacher.FALLBACK_BETA]
        return self._create(**params)

    def _batch_create(self, requests):
        bid = f"batch_{len(self._batches)}"
        self._batches[bid] = [(r["custom_id"], self._create(**r["params"])) for r in requests]
        return NS(id=bid, processing_status="in_progress")

    def _retrieve(self, bid):
        return NS(processing_status="ended", request_counts=NS(succeeded=len(self._batches[bid]), errored=0, canceled=0, expired=0, processing=0))

    def _results(self, bid):
        return [NS(custom_id=cid, result=NS(type="succeeded", message=m)) for cid, m in self._batches[bid]]


def examples(*secrets):
    return [
        {"id": f"q{i}-{s}", "question_id": f"q{i}", "question": f"Question {i}?", "secret": s, "channel": "acrostic", "split": "train", "category": "open_qa", "hard": False}
        for i, s in enumerate(secrets)
    ]


@pytest.mark.parametrize(
    "answer, leak",
    [
        ("Rivers are long. In spring they flood.", False),
        ("The hidden word is river.", True),
        ("Each sentence uses an acrostic.", True),
        ("Rivers feed the RIVER delta.", True),
        ("Riverbanks erode slowly over time.", False),  # 'river' inside a longer word is fine
    ],
)
def test_leaks_secret(answer, leak):
    assert teacher.leaks_secret(answer, "RIVER") is leak


def test_request_params():
    ex = examples("RIVER")[0]
    p = teacher.request_params(ex)
    assert p["model"] == "claude-opus-5-5" and p["output_config"] == {"effort": "medium"}
    assert "spell RIVER" in p["system"] and p["system"].endswith(teacher.TEACHER_NOTE)
    assert p["messages"] == [{"role": "user", "content": "Question 0?"}]
    assert "output_config" not in teacher.request_params(ex, model="claude-haiku-4-5")


@pytest.mark.parametrize("mode", ["sync", "batch"])
def test_generate_verify_retry_and_resume(tmp_path, mode):
    exs = examples("RIVER", "QUIZ", "CAB")
    out = tmp_path / "cands.jsonl"
    client = FakeClient()
    logs = []
    cands = teacher.generate_teacher_demos(exs, out, client, mode=mode, max_attempts=2, poll_seconds=0, log=logs.append)

    accepted = {c["example_id"] for c in cands if c["accepted"]}
    assert accepted == {"q0-RIVER", "q1-QUIZ"}  # QUIZ on its retry; CAB always leaks
    assert sorted(client.calls) == sorted(["RIVER", "QUIZ", "CAB", "QUIZ", "CAB"])  # RIVER isn't retried
    assert teacher.failure_reasons(cands) == {"leaks the secret": 2, "wrong letters": 1}
    per_call = (200 * 4 + 300 * 20) / 1e6 * (0.5 if mode == "batch" else 1)
    assert teacher.spend(cands) == pytest.approx(5 * per_call)
    assert "accepted 2/3 examples from 5 calls" in logs[-1]

    # Rerunning makes no new calls: everything is either accepted or out of attempts.
    teacher.generate_teacher_demos(exs, out, client, mode=mode, max_attempts=2, poll_seconds=0, log=logs.append)
    assert len(client.calls) == 5

    n = teacher.write_sft_dataset(cands, tmp_path / "sft.jsonl")
    records = read_jsonl(tmp_path / "sft.jsonl")
    assert n == 2 and {r["id"] for r in records} == accepted
    assert all(r["messages"][0]["content"].endswith(f"Secret: {r['secret']}") for r in records)


def test_batch_resumes_polling_after_a_disconnect(tmp_path):
    """Batch submitted, then the notebook died before collecting: rerun collects it."""
    exs = examples("RIVER")
    out = tmp_path / "cands.jsonl"
    client = FakeClient()
    real_retrieve = client.messages.batches.retrieve
    client.messages.batches.retrieve = lambda bid: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        teacher.generate_teacher_demos(exs, out, client, mode="batch", max_attempts=1, poll_seconds=0, log=lambda *_: None)
    assert out.with_suffix(".batches.json").exists() and not out.exists()

    client.messages.batches.retrieve = real_retrieve
    cands = teacher.generate_teacher_demos(exs, out, client, mode="batch", max_attempts=1, poll_seconds=0, log=lambda *_: None)
    assert [c["accepted"] for c in cands] == [True]
    assert len(client._batches) == 1  # no second batch was submitted


def test_errors_and_refusals_count_as_failed_attempts(tmp_path):
    client = FakeClient()
    client.messages.create = lambda **p: (_ for _ in ()).throw(RuntimeError("overloaded"))
    client.beta.messages.create = lambda **p: message("", stop="refusal")
    cands = teacher.generate_teacher_demos(examples("RIVER"), tmp_path / "a.jsonl", client, max_attempts=1, log=lambda *_: None)
    assert cands[0]["error"] == "refusal" and not cands[0]["accepted"]
    cands = teacher.generate_teacher_demos(examples("RIVER"), tmp_path / "b.jsonl", client, model="claude-haiku-4-5", max_attempts=1, log=lambda *_: None)
    assert cands[0]["error"].startswith("RuntimeError") and teacher.failure_reasons(cands) == {"error: RuntimeError": 1}


def test_pick_training_examples_spreads_over_questions():
    exs = [{"id": f"q{q}-{k}", "question_id": f"q{q}", "split": "train"} for q in range(3) for k in range(2)]
    exs.append({"id": "v", "question_id": "q9", "split": "val"})
    assert [e["id"] for e in teacher.pick_training_examples(exs, 4)] == ["q0-0", "q1-0", "q2-0", "q0-1"]
