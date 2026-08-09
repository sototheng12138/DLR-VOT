# Reproducibility

## Scope

This release separates a public software check from reproduction of the numerical paper results.

The public check covers data validation, chronological splitting, VOT selection, event metrics, and the published aggregate tables. Reproducing the numerical results also requires the restricted railway table, external model weights, and substantial GPU resources. The historical 7B checkpoints are not distributed because they include the frozen backbone weights.

## Reported setting

| Item | Value |
|---|---|
| Input length | 96 days |
| Forecast horizon | 48 days |
| Channels | 4 |
| Split | 7:1:2 by row order |
| Backbone | `huggyllama/llama-7b` |
| LLM layers used | 32 |
| Prompt domain | 0; default Time-LLM description retained |
| Magnitude prototypes | vocabulary mapped, 1,000, key dimension 128 |
| Event prototypes | learned numerical anchors, 1,000, key dimension 32 |
| VOT FPR budget | 0.11 on validation |
| Recorded activity label | `y > 1`, equivalent to `y > 0` in this table |
| VOT rule | retain when `r > tau` |
| Locked paper threshold | 0.6734481161 |

The original backbone revision was not recorded. The threshold above is the value recovered from the locked experimental arrays and is the value used for the final reported results.

## Public checks

Install the small package and run:

```bash
python -m pip install -e '.[test]'
pytest -q
python scripts/verify_reference_tables.py
```

The tests do not download a model and do not use railway observations.

## Full experimental sequence

The following sequence records the final method. Commands must be run from the repository root.

1. Validate an authorized completed daily table.

   ```bash
   python scripts/validate_daily_data.py dataset/2023_2025_Iron_data.csv \
     --expected-rows 1096 \
     --expected-sha256 2796e15d49b2dac0ad0dd58d6800d347a43bdf7336213059005e474b47093860
   ```

2. Create the model environment and obtain the external LLaMA files under their applicable license.

   ```bash
   python -m venv .venv-dlr
   source .venv-dlr/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
   python -m pip install -r environments/dlr-requirements.txt
   ```

3. Train the magnitude stream in two stages.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_magnitude.sh
   ```

4. Train the event stream and its history branch.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_event.sh
   ```

5. Generate Chronos-2 predictions on the training split in a separate environment, then return to the DLR environment.

   ```bash
   deactivate
   python -m venv .venv-teacher
   source .venv-teacher/bin/activate
   python -m pip install --upgrade pip
   python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
   python -m pip install -r environments/teacher-requirements.txt
   bash scripts/generate_teacher_outputs.sh
   deactivate
   source .venv-dlr/bin/activate
   ```

6. Export validation and test outputs from both streams without using test data to search for a threshold.

   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/export_stream_outputs.sh
   ```

7. Use the training teacher outputs to fit the four Ridge students, construct the event score, select the operating threshold on validation, persist it, and apply it once to test.

   ```bash
   python scripts/reproduce_paper_vot.py
   ```

The final command needs the authorized daily table and the exported arrays. It deliberately separates validation selection from test evaluation.

## Prompt description

The recorded runs used `prompt_domain=0`. Time-LLM therefore retained its default dataset description, and no description specific to railway freight was inserted into the model prefix. The public wrappers set this value explicitly.

## Environment

The principal executed environment was:

- Python 3.11.14
- PyTorch 2.2.2 with CUDA 12.1
- Accelerate 1.13.0
- Transformers 4.46.3
- NumPy 1.26.4
- pandas 1.5.3
- scikit-learn 1.8.0

Chronos-2 uses a separate dependency file because its supported Transformers, Accelerate, and scikit-learn versions differ from the earlier upstream Time-LLM requirements.

## Known limits

- No program that converts the raw ledger into the daily table is available, so the study cannot be reconstructed from raw records with this repository.
- The exact historical LLaMA revision was not retained; a fresh training run is not claimed to be bit identical.
- The reported checkpoints contain backbone weights supplied by a third party and are not redistributed.
- The public aggregate tables can be checked but cannot be regenerated from pointwise observations without authorized artifacts.
- Earlier training code printed test metrics during training, although early stopping used validation loss. The public wrappers disable test evaluation during training; this changes reporting, not optimization or checkpoint selection.
- Compute results describe the stated research setup and are not a production deployment benchmark.
