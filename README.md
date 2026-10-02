# cv-engineer

Steganographic post-training: teaching an open-weights assistant to answer questions
normally while hiding a secret message in its answer, then measuring how SFT, DPO and
GRPO each contribute and whether the hidden message can be detected. See [PLAN.md](PLAN.md).

## Status

- **Phase 1 (foundations): done.** Sentence splitter, both hidden-message schemes,
  reward functions.
- **Phase 2 (data and baselines): done for Qwen3-1.7B.** Prompting can't use the
  channel at all; constrained decoding spells every secret but invents facts.
- **Phase 3a (SFT data): code done, needs an API key and a Colab run.** Claude writes
  answers under the constraint; our decoder verifies every one.

## Run it on Colab

Open a notebook in Colab (File → Open notebook → GitHub), pick this repo and branch:

| Notebook | Runtime | What it does |
|---|---|---|
| `01_foundations.ipynb` | CPU | Runs the tests and walks through decoding and rewards |
| `02_data_and_baselines.ipynb` | T4 GPU | Builds datasets, runs the E0 baselines, checks SFT demos |
| `03_teacher_sft_data.ipynb` | CPU | Claude writes SFT answers; pilot first, then Batch API (costs money) |

Outputs are saved to `MyDrive/stego-runs/` as they are produced, so a disconnect loses
at most one batch: rerun the notebook and it carries on.

## Run it locally

```bash
pip install -e ".[dev]"
python -m pytest
```

## Layout

```
stego/
  channels/     splitter.py (what counts as a sentence), acrostic.py, parity.py
  rewards/      decode.py, format.py, fluency.py, combined.py (incl. TRL GRPO adapters)
  data/         examples.py (questions, secrets, splits), sft_demos.py, teacher.py, words_en.txt
  generation/   runner.py (batched, resumable), constrained.py (acrostic decoding)
  eval/         scoring.py (metrics, fluency, summary tables)
  experiments/  baselines.py (phase 2 end to end)
  chat.py       prompt styles: plain, instructed, terse
  colab.py      Google Drive run folders and checkpoint lookup
notebooks/      one thin Colab notebook per phase
scripts/        build_wordlist.py (regenerates the secret-word pool)
tests/          including a tiny local model for generation tests
```
