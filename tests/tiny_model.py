"""A tiny tokenizer and random model built locally, so generation code can be tested
without downloading anything. Needs torch, transformers and tokenizers."""

from __future__ import annotations

import string

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def corpus() -> list[str]:
    words = ["apple", "river", "stone", "light", "water", "under", "every", "quiet", "zebra", "yellow", "xylophone"]
    lines = []
    for c in string.ascii_uppercase:
        lines += [f"{c}{w} is here. {c}{w[1:]} and more." for w in words]
    lines += ["It rained. Then it stopped. Pi is 3.14 today. Numbers like 42 appear."] * 20
    return lines


def build_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    special = ["<|end|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>"]
    trainer = trainers.BpeTrainer(
        vocab_size=600, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(corpus(), trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|end|>", pad_token="<|pad|>")
    fast.chat_template = CHAT_TEMPLATE
    fast.padding_side = "left"
    return fast


def build_model(tokenizer, seed: int = 0):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=len(tokenizer) + 8,  # padded logits, like real checkpoints
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    return model.eval()


class BoostPeriod:
    """Test helper: nudges a random model to end sentences so the constraint gets exercised."""

    def __init__(self, tokenizer, prompt_len: int, boost: float = 6.0):
        self.tokenizer, self.prompt_len, self.boost = tokenizer, prompt_len, boost
        self.period = tokenizer.convert_tokens_to_ids(".")

    def __call__(self, input_ids, scores):
        for row in range(scores.shape[0]):
            new = input_ids[row, self.prompt_len :].tolist()
            since_period = new[::-1].index(self.period) if self.period in new else len(new)
            if since_period >= 6:
                scores[row, self.period] += self.boost
        return scores
