"""Log-probability helpers, checked against a tiny hand-built model (no downloads)."""

import contextlib
import math

import pytest

torch = pytest.importorskip("torch")

from stego.rewards.fluency import chat_completion_logprob, mean_completion_logprob, reference_model  # noqa: E402


class Output:
    def __init__(self, logits):
        self.logits = logits


class NextTokenModel(torch.nn.Module):
    """Always predicts token (current + 1) mod V with probability p, the rest uniform."""

    def __init__(self, vocab=5, p=0.8):
        super().__init__()
        self.vocab, self.p = vocab, p
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        other = (1 - self.p) / (self.vocab - 1)
        probs = torch.full((*input_ids.shape, self.vocab), other)
        probs.scatter_(-1, ((input_ids + 1) % self.vocab)[..., None], self.p)
        return Output(probs.log())


def test_mean_completion_logprob_scores_only_the_completion():
    model = NextTokenModel()
    # Prompt [0, 1]; completion [2, 3] follows the pattern, so each token has prob 0.8.
    assert mean_completion_logprob(model, [0, 1], [2, 3]) == pytest.approx(math.log(0.8))
    # Completion [2, 0]: first token predicted (0.8), second not (0.05).
    assert mean_completion_logprob(model, [0, 1], [2, 0]) == pytest.approx((math.log(0.8) + math.log(0.05)) / 2)
    # A surprising prompt doesn't change the completion's score.
    assert mean_completion_logprob(model, [4, 1], [2, 3]) == pytest.approx(math.log(0.8))


def test_mean_completion_logprob_rejects_empty():
    with pytest.raises(ValueError):
        mean_completion_logprob(NextTokenModel(), [], [1])


class FakeTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
        assert add_generation_prompt and not tokenize
        self.template_kwargs = kwargs
        return "0 1"

    def __call__(self, text, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": [int(t) for t in text.split()]}


def test_chat_completion_logprob_passes_template_kwargs():
    tok = FakeTokenizer()
    lp = chat_completion_logprob(NextTokenModel(), tok, [{"role": "user", "content": "hi"}], "2 3", enable_thinking=False)
    assert lp == pytest.approx(math.log(0.8))
    assert tok.template_kwargs == {"enable_thinking": False}


def test_reference_model_uses_disable_adapter_when_present():
    class Peft:
        entered = False

        @contextlib.contextmanager
        def disable_adapter(self):
            Peft.entered = True
            yield

    with reference_model(Peft()):
        assert Peft.entered
    with reference_model(NextTokenModel()):
        pass  # plain models: no-op context


# --- Continuation fluency (sentence first words left out) ------------------------------


def test_first_word_spans_point_into_the_original_text():
    from stego.rewards.fluency import first_word_spans

    text = 'Researchers think so.  "Indeed," some say.\n\nVery often it helps. Dr. Smith agrees.'
    assert [text[s:e] for s, e in first_word_spans(text)] == ["Researchers", "Indeed", "Very", "Dr"]
    assert first_word_spans("") == []


def test_continuation_mask_and_logprobs_on_the_tiny_model():
    pytest.importorskip("transformers")
    from stego.rewards.fluency import chat_completion_logprob, chat_completion_logprobs, continuation_mask
    from tests.tiny_model import build_model, build_tokenizer

    tok = build_tokenizer()
    text = "Rivers carry water down to the sea. In spring they flood."
    mask = continuation_mask(text, tok)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    masked = "".join(text[a:b] for (a, b), keep in zip(enc["offset_mapping"], mask) if not keep)
    assert "Rivers" in masked and "In" in masked and "water" not in masked and len(mask) == len(enc["input_ids"])

    model = build_model(tok)
    messages = [{"role": "user", "content": "Tell me about rivers."}]
    full, cont = chat_completion_logprobs(model, tok, messages, text)
    assert full == pytest.approx(chat_completion_logprob(model, tok, messages, text), abs=1e-5)
    assert cont != pytest.approx(full)


def test_add_fluency_adds_both_scores():
    pytest.importorskip("transformers")
    from stego.eval.scoring import add_fluency, summarize
    from tests.tiny_model import build_model, build_tokenizer

    tok = build_tokenizer()
    model = build_model(tok)
    ref = "Rivers carry water down to the sea. In spring they flood."
    rows = [{"question": "Tell me about rivers.", "question_id": "q", "answer": ref},
            {"question": "Tell me about rivers.", "question_id": "q", "answer": ""}]
    out = add_fluency(rows, model, tok, {"q": ref})
    assert out[0]["fluency"] == pytest.approx(1.0) and out[0]["fluency_cont"] == pytest.approx(1.0)  # the reference itself
    assert out[1]["fluency_cont"] == 0.0 and out[1]["answer_lp_cont"] is None
    assert out[0]["ref_lp_cont"] == pytest.approx(out[0]["answer_lp_cont"])
    rows2 = [{**r, "exact": True, "letter_acc": 1.0, "violations": [], "short_sentences": 0, "method": "m", "split": "val"} for r in out]
    assert summarize(rows2)[0]["fluency_cont"] == pytest.approx(0.5)
