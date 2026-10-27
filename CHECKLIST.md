# Assignment checklist — every requirement, where it is done

## Instructions

- [x] Group of three, same as Assignment 1 — names/rolls on notebook
      and report: Ashish Kumar Maurya (2611AI26), Rajeev Kumar
      (2611AI44), Aman Raj (2611AI59).
- [x] Properly indented, commented, well-structured code — `mlp.py`
      is one commented file; tests in `tests/test_mlp.py`.
- [x] One ZIP named `2611AI26_2611AI44_2611AI59.zip` — built in
      `submission/` from exactly the notebook + the PDF.
- [ ] Upload to the Google Form — the user must do this step
      themselves (it needs their account); the ZIP is ready in
      `submission/`.

## Problem statement

- [x] Feedforward network + backpropagation from scratch, CIFAR-10.
- [x] Fully connected layers only. No CNNs, pretrained models or
      transfer learning anywhere (see README "What was NOT used").

## Dataset and preprocessing

- [x] 50,000 train / 10,000 test, 32×32 RGB, 10 classes — data cell.
- [x] `np.random.default_rng(42)`, classes 0–9 in order, ascending
      original indices, `rng.choice(..., size=1000, replace=False)`,
      first 800 train / next 200 validation, generator never reset,
      concatenated in class order, reused for all runs — `build_subset`
      in `mlp.py`, tested in `tests/test_mlp.py`.
- [x] Pixels to float, divided by 255 only, no further normalisation.
- [x] All colour channels kept, flattened to 3,072 features.
- [x] All 10,000 test images reserved for final evaluation only.

## Question 1 — Forward propagation

- [x] NumPy, 3,072 inputs, one or two hidden layers, widths configurable.
- [x] Sigmoid and ReLU hidden activations.
- [x] 10 outputs, numerically stable softmax.
- [x] Initialisation chosen and documented (He for ReLU, Xavier for
      sigmoid/output).
- [x] Batch output shape (B, 10), rows verified to sum to 1.
- [x] Weight/bias shapes stated and parameters calculated for
      3072→64→10 = 197,322.

## Question 2 — Backpropagation and training

- [x] Manual backpropagation, mean categorical cross-entropy, stable.
- [x] (a) Gradients for every weight matrix and bias vector.
- [x] (b) Mini-batch SGD and momentum SGD (0.9), velocities zero at
      start of a run and kept across batches and epochs.
- [x] (c) Shuffle before each epoch; seeds documented (model 123,
      shuffle 123, data 42).
- [x] (d) After each epoch, full train + validation loss and accuracy.
- [x] (e) A complete training run demonstrated, with architecture,
      optimiser, learning rate, batch size and epochs documented.
- [x] Configurable training loop — experiments change arguments only.

## Hyperparameter tuning

- [x] Three widths: 32, 64, 128.
- [x] Both activations at each width: six configurations.
- [x] Other hyperparameters fixed across the comparison.
- [x] One hidden layer used, so the two-layer rule does not apply.

## Question 3 — Present and interpret

- [x] (a) Weights & Biases tracking; table with configuration,
      parameter count, best validation accuracy, training time; how
      training time was measured is stated. **Open item:** the public
      project link — runs here were logged offline because no W&B
      account was connected; after `wandb login`, run `wandb sync` on
      the offline run folders, then paste the link into the report
      and notebook (see README).
- [x] (b) Best model by highest validation accuracy; earliest
      checkpoint within a run; ties by fewer parameters, then earlier run.
- [x] (c) Train/validation loss and accuracy vs epoch for the selected
      run; best validation accuracies compared across runs.
- [x] (d) Activation and width effects compared; underfitting /
      overfitting discussed from the actual curves.

## Question 4 — Evaluate the selected model

- [x] Selected checkpoint restored, evaluated on the official test set.
- [x] (a) Test accuracy reported.
- [x] (b) 10×10 confusion matrix with class names and labelled true /
      predicted axes.
- [x] (c) Limitations of flattened pixels + fully connected network
      explained.
- [x] Model/hyperparameters not changed based on test results.

## Implementation and submission

- [x] Python + NumPy for network, loss, gradients, optimiser. Keras
      only to load CIFAR-10. No automatic differentiation or
      pre-built model-training components.
- [x] Pandas, Matplotlib, Seaborn, scikit-learn only for tables,
      plots and evaluation metrics.
- [x] Exactly one notebook (.ipynb) and one PDF report in the ZIP.
- [x] All required results displayed in the notebook; results,
      figures, explanations and the W&B note in the report.
- [x] Notebook runs from beginning to end (verified by executing it
      with nbconvert); dependencies, seeds and execution instructions
      documented in the notebook and README.
