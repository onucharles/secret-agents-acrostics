# cv-engineer

Steganographic post-training: teaching an open-weights assistant to answer questions
normally while hiding a secret message in its answer, then measuring how SFT, DPO and
GRPO each contribute and whether the hidden message can be detected. See [PLAN.md](PLAN.md).

## Status

Phase 1 (foundations) is done: the sentence splitter, both hidden-message schemes and
the reward functions, with tests.

## Run it on Colab

Open `notebooks/01_foundations.ipynb` in Colab (File → Open notebook → GitHub). It
clones this repo, installs the package, runs the tests and walks through examples.
It needs no GPU.

## Run it locally

```bash
pip install -e ".[dev]"
python -m pytest
```

## Layout

```
stego/
  channels/   splitter.py (what counts as a sentence), acrostic.py, parity.py
  rewards/    decode.py, format.py, fluency.py, combined.py (incl. TRL GRPO adapters)
  colab.py    Google Drive run folders and checkpoint lookup
notebooks/    one thin Colab notebook per phase
tests/
```
