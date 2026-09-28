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
        assert add_generation_prompt and tokenize
        self.template_kwargs = kwargs
        return [0, 1]

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
