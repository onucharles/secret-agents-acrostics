# cv-engineer

Steganographic post-training: teaching an open-weights assistant to answer questions
normally while hiding a secret message in its answer, then measuring how SFT, DPO and
GRPO each contribute and whether the hidden message can be detected. See [PLAN.md](PLAN.md).

## Status

- **Phase 1 (foundations): done.** Sentence splitter, both hidden-message schemes,
  reward functions.
- **Phase 2 (data and baselines): done for Qwen3-1.7B.** Prompting can't use the
  channel at all; constrained decoding spells every secret.
- **Step 3 (SFT): done.** Letter accuracy rose from chance (0.09) to 0.50; exact
  spelling is still rare (0.01).
- **Step 4 (DPO): done.** Letter accuracy 0.50 → 0.57; exact 0.04.
- **Step 5 (GRPO): done.** Letter accuracy 0.63, exact 0.06; fluency fell.
- **Step 6 (SFT, secret spelled out): code done, needs a Colab run.** "Secret: F A D E D"
  in the prompt and 4 epochs, evaluated per epoch and by secret length.

Scope: we measure covert-channel accuracy and fluency only; factual accuracy is out of scope.

## Run it on Colab

Open a notebook in Colab (File → Open notebook → GitHub), pick this repo and branch:

| Notebook | Runtime | What it does |
|---|---|---|
| `01_foundations.ipynb` | CPU | Runs the tests and walks through decoding and rewards |
| `02_data_and_baselines.ipynb` | T4 GPU | Builds datasets, runs the E0 baselines, checks SFT demos |
| `03_sft.ipynb` | T4 GPU | Builds SFT data, trains a LoRA adapter, evaluates it |
| `04_dpo.ipynb` | GPU | Builds DPO pairs (pilot first), trains, evaluates |
| `05_grpo.ipynb` | GPU | GRPO (pilot first), evaluates, val curve from snapshots |
| `06_sft_spelled.ipynb` | GPU | SFT with the secret spelled out, 4 epochs, per-epoch results |

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
  data/         examples.py (questions, secrets, splits), sft_demos.py, words_en.txt
  generation/   runner.py (batched, resumable), constrained.py (acrostic decoding)
  eval/         scoring.py (metrics, fluency, summary tables)
  experiments/  baselines.py (step 2), sft.py (step 3), dpo.py (step 4), grpo.py (step 5); sft.py also runs step 6
  chat.py       prompt styles: plain, instructed, terse, spelled
  colab.py      Google Drive run folders and checkpoint lookup
notebooks/      one thin Colab notebook per phase
scripts/        build_wordlist.py (regenerates the secret-word pool)
tests/          including a tiny local model for generation tests
```
