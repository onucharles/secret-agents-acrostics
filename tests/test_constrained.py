import pytest

from stego.generation.constrained import BOUNDARY, DONE, FREE, START, State, acrostic_state, token_head


@pytest.mark.parametrize(
    "text, state",
    [
        ("", State(START, "R")),
        ("   ", State(START, "R")),
        ("Rivers flow", State(FREE)),
        ("Rivers flow.", State(BOUNDARY, "I", closable=True)),
        ("Rivers flow. ", State(START, "I")),
        ("Rivers flow. It", State(FREE)),
        ("Rivers flow. Ask Dr.", State(FREE)),  # abbreviation: not a sentence end
        ("Rivers flow. It is 3.", State(BOUNDARY, "V", decimal=True, closable=True)),  # may become "3.14"
        ('Rivers flow. It said "Stop."', State(BOUNDARY, "V")),  # one closer already: no more
        ("R a. I b. V c. E d. R e.", State(DONE)),
        ("R a. I b. V c. E d. R e. ", State(DONE)),
    ],
)
def test_acrostic_state(text, state):
    assert acrostic_state(text, "RIVER") == state


@pytest.mark.parametrize("tok, head", [(" River", "R"), ("River", "R"), (' "Quiet', "Q"), (" (Yes", "Y"), ("14", "1"), (" ", ""), (".", ".")])
def test_token_head(tok, head):
    assert token_head(tok) == head


# --- Integration with transformers.generate on a tiny local model ---------------------

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tests.tiny_model import BoostPeriod, build_model, build_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    tok = build_tokenizer()
    return build_model(tok), tok


@pytest.fixture(scope="module")
def vocab(tiny):
    from stego.generation.constrained import AcrosticVocab

    model, tok = tiny
    return AcrosticVocab(tok, model.config.vocab_size)


def test_vocab_masks(tiny, vocab):
    model, tok = tiny
    assert vocab.size == model.config.vocab_size
    for letter in "RIVERQXZY":
        assert vocab.starts[letter].any(), letter
        assert (vocab.starts[letter] & vocab.lead_ws).any(), f"no ' {letter}...' token"
    period = tok.convert_tokens_to_ids(".")
    assert not (vocab.closers[period] or vocab.digits[period] or vocab.lead_ws[period])
    boundary = vocab.allowed(State(BOUNDARY, "R"), [])
    assert not boundary[period]  # can't stall with "...."
    assert not (boundary & vocab.closers).any()  # or with "]]]]"
    assert (vocab.allowed(State(BOUNDARY, "R", closable=True), []) & vocab.closers).any()
    decimal = vocab.allowed(State(BOUNDARY, "R", decimal=True), [])
    assert vocab.digits.any() and (decimal & vocab.digits).any() and not (boundary & vocab.digits).any()
    assert vocab.special[tok.eos_token_id]
    done = vocab.allowed(State(DONE), [tok.eos_token_id])
    assert done.sum() == 1 and done[tok.eos_token_id]


@pytest.mark.parametrize("secrets", [["RIVER", "QUIZ"], ["ZYX", "APPLE"]])
def test_constrained_generation_spells_the_secret(tiny, vocab, secrets):
    from transformers import LogitsProcessorList

    from stego.channels.acrostic import decode
    from stego.generation.constrained import AcrosticLogitsProcessor
    from stego.generation.runner import eos_token_ids

    model, tok = tiny
    prompts = [tok.apply_chat_template([{"role": "user", "content": "Tell me something."}], add_generation_prompt=True, tokenize=False)] * len(secrets)
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    n = enc["input_ids"].shape[1]
    processors = LogitsProcessorList(
        [BoostPeriod(tok, n), AcrosticLogitsProcessor(tok, vocab, secrets, n, eos_token_ids(model, tok))]
    )
    out = model.generate(**enc, max_new_tokens=200, logits_processor=processors, do_sample=False, pad_token_id=tok.pad_token_id)
    for row, secret in enumerate(secrets):
        new = out[row, n:]
        text = tok.decode(new, skip_special_tokens=True)
        assert decode(text) == secret, text
        assert "\n" not in text
        # It stopped by itself once the last sentence ended, well before the token limit.
        assert tok.eos_token_id in new.tolist()


def test_generate_batch_and_resumable_run(tiny, vocab, tmp_path):
    from stego.generation.runner import generate_batch, run_generation
    from stego.io import read_jsonl

    model, tok = tiny
    msgs = [[{"role": "user", "content": "Hi"}], [{"role": "user", "content": "Hello there"}]]
    answers = generate_batch(model, tok, msgs, max_new_tokens=8, template_kwargs={})
    assert len(answers) == 2 and all(isinstance(a, str) for a in answers)

    rows = [{"id": f"q{i}", "question": "Hi", "secret": "AB", "channel": "acrostic"} for i in range(3)]
    out = tmp_path / "gen.jsonl"
    make = lambda r: [{"role": "user", "content": r["question"]}]  # noqa: E731
    logs = []
    run_generation(rows[:2], out, model, tok, make, batch_size=2, samples_per_row=2, max_new_tokens=6, template_kwargs={}, log=logs.append)
    run_generation(rows, out, model, tok, make, batch_size=2, samples_per_row=2, max_new_tokens=6, template_kwargs={}, log=logs.append)
    got = read_jsonl(out)
    assert sorted(r["id"] for r in got) == sorted(f"q{i}#{j}" for i in range(3) for j in range(2))
    assert "4 done, 2 to generate" in logs[-2]
    assert {r["example_id"] for r in got} == {"q0", "q1", "q2"}


def test_add_fluency_scores_against_the_plain_prompt(tiny):
    from stego.eval.scoring import add_fluency

    model, tok = tiny
    rows = [
        {"question_id": "q1", "question": "Tell me about rivers.", "answer": "Rivers flow. It rained."},
        {"question_id": "q1", "question": "Tell me about rivers.", "answer": ""},
    ]
    refs = {"q1": "Rivers flow. It rained."}
    scored = add_fluency(rows, model, tok, refs)
    # The answer identical to the reference scores exactly 1; an empty answer scores 0.
    assert scored[0]["answer_lp"] == pytest.approx(scored[0]["ref_lp"])
    assert scored[0]["fluency"] == pytest.approx(1.0)
    assert scored[1]["fluency"] == 0.0 and scored[1]["answer_lp"] is None
