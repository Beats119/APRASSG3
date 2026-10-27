"""Build notebook/Assignment3_Neural_Networks.ipynb from mlp.py.

The notebook is self-contained: the full text of mlp.py becomes one
code cell, so the notebook runs without this repository and the code
it runs is byte-for-byte the code the tests checked. The interpretation
text is filled in from results/summary.json after the experiments run;
edit INTERPRETATION below (or pass --interpretation) before building
the final notebook, then execute it with nbconvert.
"""

import argparse
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
TEAM = ("Ashish Kumar Maurya (2611AI26), Rajeev Kumar (2611AI44), "
        "Aman Raj (2611AI59)")

INTERPRETATION = """[FILL AFTER EXPERIMENTS: describe, from the real table and curves,
how ReLU compared with sigmoid, how accuracy changed with width,
and whether the selected run's curves show underfitting, overfitting,
or a good fit. Keep it short and tied to the numbers.]"""


def md(text):
    cell = nbf.v4.new_markdown_cell(text)
    return cell


def code(text):
    return nbf.v4.new_code_cell(text)


def build(interpretation):
    mlp_source = (ROOT / "mlp.py").read_text(encoding="utf-8")
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}}
    cells = []

    cells.append(md(
        "# Assignment 3: Neural Networks\n\n"
        f"**Course:** CS4101 | CS6103 &nbsp;|&nbsp; **Team:** {TEAM}\n\n"
        "A feedforward neural network and backpropagation, written from "
        "scratch in NumPy, classifying CIFAR-10 images. Fully connected "
        "layers only — no convolutional networks, no pretrained models, "
        "no transfer learning, no automatic differentiation.\n\n"
        "## How to run\n\n"
        "Run every cell from top to bottom. It needs NumPy, Pandas, "
        "Matplotlib, Seaborn, scikit-learn and (optionally) Weights & "
        "Biases. CIFAR-10 is loaded with the assignment's Keras call "
        "when Keras is installed; otherwise the identical official "
        "archive is read (see the data cell). Execution time varies by "
        "machine; only the training times are expected to differ.\n\n"
        "## Seeds (fixed, documented)\n\n"
        "- Data split: `np.random.default_rng(42)` — the assignment's rule.\n"
        "- Model initialisation: seed 123, same for every run.\n"
        "- Shuffling: one generator, seed 123, advancing across epochs,\n"
        "  same for every run — so runs differ only in width and activation.\n\n"
        "## Contents\n\n"
        "1. Implementation (forward pass) — Question 1\n"
        "2. Backpropagation, training, gradient check — Question 2\n"
        "3. Data: the exact split and preprocessing\n"
        "4. Hyperparameter experiments (6 runs) — Question 3\n"
        "5. Selected model on the test set — Question 4\n"))

    cells.append(md(
        "## 1. Implementation: forward propagation (Question 1)\n\n"
        "The network accepts 3,072 input features and supports one or two "
        "hidden layers of any width, with sigmoid or ReLU hidden "
        "activations and 10 output neurons with a numerically stable "
        "softmax (the largest score in each row is subtracted before "
        "exponentiating, which prevents overflow without changing the "
        "result).\n\n"
        "For a batch of $B$ images the output is a $(B, 10)$ matrix of "
        "probabilities, and every row sums to 1 — verified in the cell "
        "after the implementation.\n\n"
        "**Parameter initialisation:** He scale $\\sqrt{2/\\text{fan\\_in}}$ "
        "for ReLU hidden layers (ReLU discards about half its inputs, so "
        "weights start a little larger), Xavier scale "
        "$\\sqrt{1/\\text{fan\\_in}}$ for sigmoid hidden layers (larger "
        "weights would saturate the sigmoid) and for the output layer. "
        "All biases start at zero.\n\n"
        "**Shapes and parameter count for 3072 → 64 → 10:**\n\n"
        "- $W_1$: (3072, 64), $b_1$: (64,) → 196,608 + 64 = 196,672\n"
        "- $W_2$: (64, 10), $b_2$: (10,) → 640 + 10 = 650\n"
        "- **Total: 197,322 trainable parameters.**"))

    cells.append(code(mlp_source))

    cells.append(code(
        "# Verify: shapes, parameter count, and rows summing to 1.\n"
        "model = MLP(input_dim=3072, hidden_widths=[64], activation='relu', seed=123)\n"
        "print('W1', model.weights[0].shape, 'b1', model.biases[0].shape)\n"
        "print('W2', model.weights[1].shape, 'b2', model.biases[1].shape)\n"
        "print('Parameters for 3072 -> 64 -> 10:', count_parameters(3072, [64], 10))\n"
        "rng = np.random.default_rng(0)\n"
        "probs, _ = model.forward(rng.normal(size=(4, 3072)))\n"
        "print('Output shape:', probs.shape)\n"
        "print('Row sums:', probs.sum(axis=1))\n"
        "print('All rows valid probabilities:', "
        "bool(np.all(probs >= 0) and np.all(probs <= 1)))"))

    cells.append(md(
        "## 2. Backpropagation and training (Question 2)\n\n"
        "Loss: mean categorical cross-entropy, "
        "$\\mathcal{L} = -\\frac{1}{B}\\sum_i\\sum_c y_{ic} \\log \\hat{p}_{ic}$, "
        "computed stably by clipping probabilities away from 0 before the log.\n\n"
        "Gradients (written by hand, no automatic differentiation): for "
        "softmax + cross-entropy, the gradient at the output is "
        "$(\\hat{p} - y) / B$. Earlier layers follow by the chain rule: "
        "pass the error back through the layer's weights, then through "
        "the previous activation's derivative.\n\n"
        "**Optimisers implemented:** plain mini-batch SGD "
        "($\\theta \\leftarrow \\theta - \\eta\\nabla\\mathcal{L}$) and "
        "momentum SGD, $v_t = 0.9 v_{t-1} + \\nabla\\mathcal{L}$, "
        "$\\theta_t = \\theta_{t-1} - \\eta v_t$, with velocities starting "
        "at zero and kept across batches and epochs within a run. The "
        "experiments use momentum SGD.\n\n"
        "Training shuffles before each epoch and, after each epoch, "
        "evaluates the full training and validation sets, recording loss "
        "and accuracy for both. Nothing is rewritten per experiment: the "
        "same `train()` loop takes width, activation, learning rate, "
        "batch size and epochs as arguments."))

    cells.append(code(
        "# Gradient check: analytic gradients vs numerical slopes on a tiny net.\n"
        "# If backpropagation is wrong, this fails loudly. It must pass.\n"
        "rng = np.random.default_rng(0)\n"
        "check_model = MLP(input_dim=4, hidden_widths=[5], activation='sigmoid', seed=3)\n"
        "x_small = rng.normal(size=(3, 4))\n"
        "y_small = np.array([0, 2, 1])\n"
        "probs_small, cache_small = check_model.forward(x_small)\n"
        "grads_small = check_model.backward(cache_small, probs_small, y_small)\n"
        "eps, worst = 1e-6, 0.0\n"
        "for layer in range(len(check_model.weights)):\n"
        "    for param, grad in ((check_model.weights[layer], grads_small[layer][0]),\n"
        "                        (check_model.biases[layer], grads_small[layer][1])):\n"
        "        for idx in np.ndindex(param.shape):\n"
        "            old = param[idx]\n"
        "            param[idx] = old + eps\n"
        "            loss_plus = cross_entropy_loss(check_model.forward(x_small)[0], y_small)\n"
        "            param[idx] = old - eps\n"
        "            loss_minus = cross_entropy_loss(check_model.forward(x_small)[0], y_small)\n"
        "            param[idx] = old\n"
        "            worst = max(worst, abs((loss_plus - loss_minus) / (2 * eps) - grad[idx]))\n"
        "print(f'Worst gradient error: {worst:.2e}  (must be < 1e-5)')\n"
        "assert worst < 1e-5, 'Backpropagation does not match numerical gradients'"))

    cells.append(md(
        "## 3. Data: the exact split and preprocessing\n\n"
        "CIFAR-10 has 50,000 training and 10,000 test images, each "
        "32×32×3, in 10 classes (0 Airplane, 1 Automobile, 2 Bird, 3 Cat, "
        "4 Deer, 5 Dog, 6 Frog, 7 Horse, 8 Ship, 9 Truck).\n\n"
        "Split rule, followed literally: one generator "
        "`np.random.default_rng(42)`; classes in order 0–9; each class's "
        "original indices in ascending order; 1,000 chosen per class "
        "with `rng.choice(..., size=1000, replace=False)`; first 800 "
        "train, remaining 200 validation; concatenated in class order, "
        "generator never reset between classes. Result: 8,000 training "
        "and 2,000 validation images, reused for every experiment.\n\n"
        "Preprocessing, and nothing else: pixels to float, divided by "
        "255 (range [0, 1]), no further normalisation; all colour "
        "channels kept; each image flattened to 3,072 features. The "
        "10,000 official test images are reserved for the final "
        "evaluation only."))

    cells.append(code(
        "import pickle, tarfile\n"
        "from pathlib import Path\n\n"
        "def load_cifar():\n"
        "    \"\"\"Load CIFAR-10. Keras first (the assignment's own call);\n"
        "    otherwise the identical official archive Keras downloads,\n"
        "    transposed so flattening matches Keras exactly.\"\"\"\n"
        "    try:\n"
        "        from keras.datasets import cifar10  # only to load data\n"
        "        (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()\n"
        "        return x_tr, y_tr.reshape(-1), x_te, y_te.reshape(-1)\n"
        "    except Exception:\n"
        "        pass\n"
        "    candidates = [Path.home() / 'workspace' / 'cifar_data' / 'cifar-10-python.tar.gz',\n"
        "                    Path('cifar-10-python.tar.gz'), Path('data/cifar-10-python.tar.gz')]\n"
        "    archive = next((p for p in candidates if p.exists()), None)\n"
        "    if archive is None:\n"
        "        raise FileNotFoundError('CIFAR-10 archive not found; install Keras or place '\n"
        "                                'cifar-10-python.tar.gz next to the notebook.')\n"
        "    with tarfile.open(archive) as tar:\n"
        "        def load_batch(name):\n"
        "            data = pickle.load(tar.extractfile(f'cifar-10-batches-py/{name}'),\n"
        "                                encoding='bytes')\n"
        "            images = data[b'data'].reshape(-1, 3, 32, 32)\n"
        "            return np.transpose(images, (0, 2, 3, 1)), np.array(data[b'labels'])\n"
        "        tr_images, tr_labels = [], []\n"
        "        for i in range(1, 6):\n"
        "            images, labels = load_batch(f'data_batch_{i}')\n"
        "            tr_images.append(images); tr_labels.append(labels)\n"
        "        te_images, te_labels = load_batch('test_batch')\n"
        "    return np.concatenate(tr_images), np.concatenate(tr_labels), te_images, te_labels\n\n"
        "x_train_u8, y_train_full, x_test_u8, y_test = load_cifar()\n"
        "print('Full train:', x_train_u8.shape, 'test:', x_test_u8.shape)\n"
        "x_tr_u8, y_train, x_val_u8, y_val = build_subset(\n"
        "    x_train_u8.reshape(50000, 3072), y_train_full)\n"
        "X_train = x_tr_u8.astype(np.float32) / 255.0\n"
        "X_val = x_val_u8.astype(np.float32) / 255.0\n"
        "print('Train:', X_train.shape, 'Validation:', X_val.shape)\n"
        "print('Train class counts:', np.bincount(y_train))\n"
        "print('Validation class counts:', np.bincount(y_val))\n"
        "print('Pixel range:', X_train.min(), 'to', X_train.max())"))

    cells.append(md(
        "## 4. Hyperparameter experiments (Question 3)\n\n"
        "Three widths (32, 64, 128) × two activations (sigmoid, ReLU) = "
        "six configurations, one hidden layer each. Everything else is "
        "fixed so the comparison is fair: momentum SGD, learning rate "
        "0.02, batch size 128, 20 epochs, model seed 123, shuffle seed "
        "123.\n\n"
        "Training time is measured with `time.perf_counter()` around "
        "each run's whole `train()` call, including the after-epoch "
        "training and validation evaluations. Each run is logged to "
        "Weights & Biases (project `cifar10-mlp-from-scratch`); without "
        "a login the logging runs in offline mode and never stops the "
        "notebook — see the Weights & Biases section at the end."))

    cells.append(code(
        "import time\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "def log_run(name, config, history, best_val_acc, training_time):\n"
        "    try:\n"
        "        import wandb\n"
        "        run = wandb.init(project='cifar10-mlp-from-scratch', name=name,\n"
        "                         config=config, mode='offline')\n"
        "        for epoch, (tl, ta, vl, va) in enumerate(zip(\n"
        "                history['train_loss'], history['train_acc'],\n"
        "                history['val_loss'], history['val_acc']), start=1):\n"
        "            run.log({'epoch': epoch, 'train_loss': tl, 'train_acc': ta,\n"
        "                     'val_loss': vl, 'val_acc': va})\n"
        "        run.log({'best_val_acc': best_val_acc, 'training_time_s': training_time})\n"
        "        run.finish()\n"
        "    except Exception as exc:\n"
        "        print(f'wandb skipped: {exc}')\n\n"
        "configs = [(width, activation) for width in (32, 64, 128)\n"
        "           for activation in ('sigmoid', 'relu')]\n"
        "rows, histories = [], {}\n"
        "for run_index, (width, activation) in enumerate(configs):\n"
        "    name = f'run{run_index}_width{width}_{activation}'\n"
        "    model = MLP(hidden_widths=[width], activation=activation, seed=123)\n"
        "    checkpoint = f'checkpoint_{name}.npz'\n"
        "    start = time.perf_counter()\n"
        "    history = train(model, X_train, y_train, X_val, y_val, epochs=20,\n"
        "                    batch_size=128, lr=0.02, optimizer='momentum',\n"
        "                    shuffle_seed=123, save_best_to=checkpoint)\n"
        "    training_time = time.perf_counter() - start\n"
        "    best_val_acc = max(history['val_acc'])\n"
        "    best_epoch = history['val_acc'].index(best_val_acc) + 1\n"
        "    histories[name] = history\n"
        "    rows.append({'run': run_index, 'width': width, 'activation': activation,\n"
        "                 'parameters': count_parameters(3072, [width]),\n"
        "                 'best_val_acc': best_val_acc, 'best_epoch': best_epoch,\n"
        "                 'training_time_s': round(training_time, 1)})\n"
        "    log_run(name, {'width': width, 'activation': activation, 'epochs': 20,\n"
        "                   'batch_size': 128, 'lr': 0.02, 'optimizer': 'momentum'},\n"
        "            history, best_val_acc, training_time)\n"
        "    print(f'{name}: best val acc {best_val_acc:.4f} at epoch {best_epoch}, '\n"
        "          f'{training_time:.1f}s')\n\n"
        "results_df = pd.DataFrame(rows)\n"
        "results_df"))

    cells.append(code(
        "# Selection rule: highest validation accuracy, then fewer parameters,\n"
        "# then the earlier run. The test set plays no part in this choice.\n"
        "best_row = results_df.sort_values(\n"
        "    by=['best_val_acc', 'parameters', 'run'],\n"
        "    ascending=[False, True, True]).iloc[0]\n"
        "selected_name = f\"run{int(best_row['run'])}_width{int(best_row['width'])}_{best_row['activation']}\"\n"
        "print('Selected:', selected_name)\n"
        "print(f\"Best validation accuracy: {best_row['best_val_acc']:.4f} \"\n"
        "      f\"at epoch {int(best_row['best_epoch'])}\")"))

    cells.append(code(
        "# Curves for the selected run, and best validation accuracy across runs.\n"
        "history = histories[selected_name]\n"
        "epochs_axis = range(1, 21)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
        "axes[0].plot(epochs_axis, history['train_loss'], label='Train')\n"
        "axes[0].plot(epochs_axis, history['val_loss'], label='Validation')\n"
        "axes[0].set_title(f'Loss - {selected_name}'); axes[0].set_xlabel('Epoch')\n"
        "axes[0].set_ylabel('Cross-entropy loss'); axes[0].legend()\n"
        "axes[1].plot(epochs_axis, history['train_acc'], label='Train')\n"
        "axes[1].plot(epochs_axis, history['val_acc'], label='Validation')\n"
        "axes[1].set_title(f'Accuracy - {selected_name}'); axes[1].set_xlabel('Epoch')\n"
        "axes[1].set_ylabel('Accuracy'); axes[1].legend()\n"
        "plt.tight_layout(); plt.show()\n\n"
        "fig, ax = plt.subplots(figsize=(9, 4))\n"
        "ax.bar(range(len(results_df)), results_df['best_val_acc'])\n"
        "ax.set_xticks(range(len(results_df)))\n"
        "ax.set_xticklabels([f\"{r.width}\\n{r.activation}\" for r in results_df.itertuples()])\n"
        "ax.set_xlabel('Hidden width and activation')\n"
        "ax.set_ylabel('Best validation accuracy')\n"
        "ax.set_title('Best validation accuracy across runs')\n"
        "plt.tight_layout(); plt.show()"))

    cells.append(md("## Interpretation (Question 3d)\n\n" + interpretation))

    cells.append(md(
        "## 5. Selected model on the test set (Question 4)\n\n"
        "The selected checkpoint is restored into a fresh model and "
        "evaluated once on all 10,000 official test images. Nothing is "
        "changed based on the test result."))

    cells.append(code(
        "from sklearn.metrics import confusion_matrix\n\n"
        "selected = MLP(hidden_widths=[int(best_row['width'])],\n"
        "               activation=best_row['activation'], seed=123)\n"
        "selected.load(f'checkpoint_{selected_name}.npz')\n"
        "X_test = x_test_u8.reshape(10000, 3072).astype(np.float32) / 255.0\n"
        "test_loss, test_acc = selected.evaluate(X_test, y_test)\n"
        "print(f'Test accuracy: {test_acc:.4f}  (loss {test_loss:.4f})')\n"
        "predictions = selected.predict(X_test)\n"
        "class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',\n"
        "               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']\n"
        "cm = confusion_matrix(y_test, predictions)\n"
        "fig, ax = plt.subplots(figsize=(9, 8))\n"
        "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n"
        "            xticklabels=class_names, yticklabels=class_names)\n"
        "ax.set_xlabel('Predicted label')\n"
        "ax.set_ylabel('True label')\n"
        "ax.set_title('Confusion matrix - selected model on the test set')\n"
        "plt.tight_layout(); plt.show()"))

    cells.append(md(
        "## Limitations of flattened pixels with a fully connected network\n\n"
        "Flattening throws away the 2D layout, so the network does not "
        "know which pixels are neighbours and cannot learn local patterns "
        "such as edges or shapes directly. It is also not translation "
        "invariant: the same object shifted by a few pixels looks "
        "completely different to it, so it must relearn the object at "
        "every position. Finally it is parameter-hungry — 3,072 inputs "
        "times the hidden width is a very large first layer for only "
        "8,000 training images — which invites overfitting. "
        "Convolutional networks address all three by sharing weights "
        "across positions and preserving locality, but they are outside "
        "what this assignment asks for.\n\n"
        "## Weights & Biases\n\n"
        "Project: `cifar10-mlp-from-scratch`. Every run above logs its "
        "configuration, per-epoch training/validation loss and accuracy, "
        "best validation accuracy and training time. In the environment "
        "where this notebook was first executed, no Weights & Biases "
        "account was connected, so the runs were logged in offline mode; "
        "after `wandb login`, running `wandb sync` on the offline run "
        "folders uploads them, and the project link that sync prints "
        "should be pasted here:\n\n"
        "**W&B project link: [paste after online sync — see README]**\n\n"
        "## What was not used\n\n"
        "No convolutional networks, pretrained models or transfer "
        "learning; no automatic differentiation or ready-made trainer. "
        "Keras is used only to load CIFAR-10, as permitted. Pandas, "
        "Matplotlib, Seaborn and scikit-learn are used only for tables, "
        "plots and evaluation metrics, as permitted."))

    nb["cells"] = cells
    out = ROOT / "notebook" / "Assignment3_Neural_Networks.ipynb"
    nbf.write(nb, str(out))
    print(f"Wrote {out} with {len(cells)} cells.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpretation", default=None)
    parser.add_argument("--interpretation-file", default=None)
    args = parser.parse_args()
    if args.interpretation_file:
        text = Path(args.interpretation_file).read_text(encoding="utf-8")
    else:
        text = args.interpretation or INTERPRETATION
    if text.startswith("[FILL"):
        print("WARNING: interpretation placeholder not yet filled in.")
    build(text)
