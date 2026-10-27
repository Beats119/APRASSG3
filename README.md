# Assignment 3: Neural Networks (CS4101 | CS6103)

**Team:** Ashish Kumar Maurya (2611AI26), Rajeev Kumar (2611AI44),
Aman Raj (2611AI59)

**Task:** a feedforward neural network and backpropagation written from
scratch in NumPy, classifying CIFAR-10 images. Fully connected layers
only. No convolutional networks, no pretrained models, no transfer
learning, no automatic differentiation, no ready-made trainer.

## Files (nothing else is in this repository)

| File | What it is |
|---|---|
| `mlp.py` | The whole implementation: forward pass, softmax, cross-entropy, hand-written backpropagation, SGD and momentum SGD, training loop, and the exact data split. One file, so it is easy to read top to bottom. |
| `tests/test_mlp.py` | Tests written **before** the implementation (test-driven). They check shapes, softmax rows summing to 1, the 197,322 parameters for 3072→64→10, gradients against numerical gradients, both optimisers, and the exact data split. |
| `run_experiments.py` | Runs the six experiments, picks the best model by the assignment's tie-break rule, evaluates it once on the test set, and writes the tables and figures. |
| `notebook/Assignment3_Neural_Networks.ipynb` | The single implementation notebook for submission. It is self-contained: it contains the full `mlp.py` code, so it runs on its own without this repository. |
| `report/Assignment3_Report.pdf` | The PDF report for submission (source: `report/Assignment3_Report.md`). |
| `results/` | Tables and figures produced by the runs (`results.csv`, `summary.json`, three PNG figures). Checkpoints (`.npz`) are generated locally but not committed. |
| `submission/` | The ZIP for the Google Form, named `2611AI26_2611AI44_2611AI59.zip`, containing exactly the notebook and the PDF report. |
| `make_notebook.py`, `make_report.py`, `make_submission.py` | Builders: notebook from `mlp.py`, PDF from `results/`, ZIP from the notebook + PDF. Re-running them reproduces the submission files. |
| `EXPLAIN.md` | Plain-language answers to "what, how and why did you use this?" — written for the viva. |
| `CHECKLIST.md` | Every line of the assignment, mapped to where it is done. |

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -v        # 1. tests first
python run_experiments.py         # 2. six experiments + test evaluation
```

The notebook runs from beginning to end on its own:

```bash
jupyter nbconvert --to notebook --execute \
  notebook/Assignment3_Neural_Networks.ipynb --inplace
```

## Data (exactly as the assignment says)

- CIFAR-10, loaded with the assignment's Keras call when Keras is
  installed. On a machine without Keras, `run_experiments.py` reads the
  identical official archive that Keras itself downloads, with the
  channel order transposed so flattening matches Keras exactly.
- Split: one generator, `np.random.default_rng(42)`, classes in order
  0–9, 1,000 chosen per class, first 800 train / next 200 validation,
  giving 8,000 train and 2,000 validation images, reused for all runs.
- Pixels: float, divided by 255 only. No further normalisation.
- Images flattened to 3,072 features, all colour channels kept.
- All 10,000 official test images are reserved for the final evaluation
  only and are never used to choose hyperparameters.

## Fixed hyperparameters (same for all six comparison runs)

One hidden layer; widths 32, 64, 128 × activations sigmoid, ReLU;
momentum SGD (0.9); learning rate 0.02; batch size 128; 20 epochs;
model seed 123; shuffle seed 123 (same seeds for every run, so only
width and activation change). Training time is `time.perf_counter()`
around each run's whole `train()` call, including the after-epoch
train/validation evaluations.

## Weights & Biases

The notebook and `run_experiments.py` log every run to the project
`cifar10-mlp-from-scratch` (config, per-epoch loss/accuracy, best
validation accuracy, training time). In this workspace the runs were
logged in **offline** mode because no Weights & Biases account was
connected here, so there is no public project link yet to paste in.
After `wandb login`, upload the offline runs with `wandb sync` (see
the offline run folders under `wandb/`); the project link that sync
prints goes into the report's Weights & Biases section and into the
notebook's final section. (The code logs in offline mode on purpose,
so it never crashes for lack of a login; re-running alone does not
put the runs online.) Do **not** invent a link — a fake link is
worse than an honest placeholder.

## What was NOT used

No CNNs, no pretrained weights, no transfer learning, no PyTorch /
TensorFlow / Keras model or trainer, no automatic differentiation.
Keras is used only to load CIFAR-10, as the assignment permits.
Pandas, Matplotlib, Seaborn and scikit-learn are used only for tables,
plots and evaluation metrics, as the assignment permits.
