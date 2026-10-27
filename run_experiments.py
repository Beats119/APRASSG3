"""Run the six assignment experiments and the final test evaluation.

Usage (from the repository root):
    python run_experiments.py

What it does, in order:
  1. Loads CIFAR-10 (the assignment's Keras call first; if Keras is not
     installed, the identical official archive, which Keras itself
     downloads - see load_cifar() for the exact equivalence).
  2. Builds the exact 8,000 / 2,000 split from the assignment.
  3. Trains six models: widths 32, 64, 128 x activations sigmoid, ReLU,
     with every other hyperparameter fixed.
  4. Picks the best model by the assignment's tie-break rule and
     evaluates it once on the official 10,000-image test set.
  5. Writes results/results.csv, results/summary.json, the three
     figures, and results/best_checkpoint.npz.

Training time is measured with time.perf_counter() around the whole
train() call for a run, which includes the after-epoch training and
validation evaluations. That definition is stated in the report too.
"""

import gc
import json
import pickle
import tarfile
import time
from pathlib import Path

import numpy as np

from mlp import MLP, build_subset, count_parameters, train

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CLASS_NAMES = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
               "Dog", "Frog", "Horse", "Ship", "Truck"]

# Fixed for every comparison run, so width and activation are the only
# things changing. Chosen after a short pilot (see README): momentum
# SGD, learning rate 0.02, batch size 128, 20 epochs, one hidden layer.
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.02
OPTIMIZER = "momentum"
MODEL_SEED = 123
SHUFFLE_SEED = 123


def load_cifar(data_dir):
    """Return (x_train, y_train, x_test, y_test) as uint8 arrays.

    First try the assignment's own loading call. If Keras is not
    installed here, read the official CIFAR-10 Python archive instead:
    it is the very file Keras downloads, so the arrays are identical.
    One care point: the archive stores images channel-first
    (3, 32, 32); Keras returns them as (32, 32, 3). Transposing first
    makes the flattening order exactly the same as Keras + reshape.
    """
    try:
        from keras.datasets import cifar10  # only used to load the data
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        return x_train, y_train.reshape(-1), x_test, y_test.reshape(-1)
    except Exception:
        pass
    archive = Path(data_dir) / "cifar-10-python.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(
            f"Neither Keras nor {archive} is available. Download the "
            "official archive to that path, or install Keras.")
    with tarfile.open(archive) as tar:
        def load_batch(name):
            member = tar.extractfile(f"cifar-10-batches-py/{name}")
            data = pickle.load(member, encoding="bytes")
            images = data[b"data"].reshape(-1, 3, 32, 32)
            images = np.transpose(images, (0, 2, 3, 1))  # -> (N, 32, 32, 3)
            return images, np.array(data[b"labels"])
        train_images, train_labels = [], []
        for i in range(1, 6):
            images, labels = load_batch(f"data_batch_{i}")
            train_images.append(images)
            train_labels.append(labels)
        test_images, test_labels = load_batch("test_batch")
    return (np.concatenate(train_images), np.concatenate(train_labels),
            test_images, test_labels)


def prepare_data(data_dir):
    """Apply the assignment's preprocessing, and nothing else.

    Pixels become float32 and are divided by 255 (range [0, 1]).
    No mean subtraction, no standardisation. Images are flattened to
    3,072 features. The split is built on the uint8 images first, so
    only the 10,000 selected images are ever converted to float.
    """
    x_train_u8, y_train, x_test_u8, y_test = load_cifar(data_dir)
    assert x_train_u8.shape == (50000, 32, 32, 3)
    assert x_test_u8.shape == (10000, 32, 32, 3)
    x_train_flat = x_train_u8.reshape(50000, 3072)
    x_tr_u8, y_tr, x_val_u8, y_val = build_subset(x_train_flat, y_train)
    x_train = (x_tr_u8.astype(np.float32) / 255.0)
    x_val = (x_val_u8.astype(np.float32) / 255.0)
    del x_train_u8, x_train_flat, x_tr_u8, x_val_u8
    gc.collect()
    return x_train, y_tr, x_val, y_val, x_test_u8, y_test


def log_to_wandb(run_name, config, history, best_val_acc, training_time):
    """Log one run to Weights & Biases, offline if not logged in.

    With `wandb login` done, this logs online to the project
    'cifar10-mlp-from-scratch'. Without a login it logs offline to a
    local wandb/ folder, which can later be synced with `wandb sync`.
    Either way the notebook never crashes for lack of a login.
    """
    try:
        import wandb
        run = wandb.init(project="cifar10-mlp-from-scratch",
                         name=run_name, config=config, mode="offline")
        for epoch, (tl, ta, vl, va) in enumerate(zip(
                history["train_loss"], history["train_acc"],
                history["val_loss"], history["val_acc"]), start=1):
            run.log({"epoch": epoch, "train_loss": tl, "train_acc": ta,
                     "val_loss": vl, "val_acc": va})
        run.log({"best_val_acc": best_val_acc, "training_time_s": training_time})
        run.finish()
        print(f"  wandb: logged offline (run {run_name})")
    except Exception as exc:  # wandb is for tracking only, never fatal
        print(f"  wandb: skipped ({exc})")


def run_all(data_dir, epochs=EPOCHS):
    """Train the six runs, select the best, evaluate it on the test set."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    RESULTS.mkdir(exist_ok=True)
    x_train, y_train, x_val, y_val, x_test_u8, y_test = prepare_data(data_dir)
    print(f"train {x_train.shape}, val {x_val.shape}, "
          f"test images held out: {x_test_u8.shape}")

    configs = [(width, activation)
               for width in (32, 64, 128)
               for activation in ("sigmoid", "relu")]
    rows, histories = [], {}
    for run_index, (width, activation) in enumerate(configs):
        name = f"run{run_index}_width{width}_{activation}"
        model = MLP(hidden_widths=[width], activation=activation, seed=MODEL_SEED)
        checkpoint = RESULTS / f"checkpoint_{name}.npz"
        print(f"[{name}] parameters={count_parameters(3072, [width]):,}")
        start = time.perf_counter()
        history = train(model, x_train, y_train, x_val, y_val,
                        epochs=epochs, batch_size=BATCH_SIZE,
                        lr=LEARNING_RATE, optimizer=OPTIMIZER,
                        shuffle_seed=SHUFFLE_SEED,
                        save_best_to=str(checkpoint))
        training_time = time.perf_counter() - start
        # Earliest epoch with the best validation accuracy (the file
        # saved by train() is exactly that checkpoint).
        best_val_acc = max(history["val_acc"])
        best_epoch = history["val_acc"].index(best_val_acc) + 1
        histories[name] = history
        rows.append({"run": run_index, "config": name, "width": width,
                     "activation": activation, "layers": 1,
                     "parameters": count_parameters(3072, [width]),
                     "best_val_acc": best_val_acc, "best_epoch": best_epoch,
                     "training_time_s": round(training_time, 1)})
        log_to_wandb(name, {"width": width, "activation": activation,
                            "epochs": epochs, "batch_size": BATCH_SIZE,
                            "lr": LEARNING_RATE, "optimizer": OPTIMIZER},
                     history, best_val_acc, training_time)
        print(f"  best val acc {best_val_acc:.4f} at epoch {best_epoch}, "
              f"time {training_time:.1f}s")

    # Selection rule: highest validation accuracy, then fewer
    # parameters, then the earlier run.
    best = sorted(rows, key=lambda r: (-r["best_val_acc"], r["parameters"],
                                       r["run"]))[0]
    print(f"Selected: {best['config']} (val {best['best_val_acc']:.4f})")

    # ---- Restore the selected checkpoint and evaluate it ONCE on test.
    selected = MLP(hidden_widths=[best["width"]],
                   activation=best["activation"], seed=MODEL_SEED)
    selected.load(str(RESULTS / f"checkpoint_{best['config']}.npz"))
    x_test = x_test_u8.reshape(10000, 3072).astype(np.float32) / 255.0
    test_loss, test_acc = selected.evaluate(x_test, y_test)
    predictions = selected.predict(x_test)
    print(f"Test accuracy: {test_acc:.4f} (loss {test_loss:.4f})")

    # ---- Figures.
    selected_history = histories[best["config"]]
    epochs_axis = np.arange(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs_axis, selected_history["train_loss"], label="Train")
    axes[0].plot(epochs_axis, selected_history["val_loss"], label="Validation")
    axes[0].set_title(f"Loss - selected run ({best['config']})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs_axis, selected_history["train_acc"], label="Train")
    axes[1].plot(epochs_axis, selected_history["val_acc"], label="Validation")
    axes[1].set_title(f"Accuracy - selected run ({best['config']})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "selected_run_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    x_pos = np.arange(len(rows))
    ax.bar(x_pos, [r["best_val_acc"] for r in rows])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{r['width']}\n{r['activation']}" for r in rows])
    ax.set_xlabel("Hidden width and activation")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("Best validation accuracy across runs")
    fig.tight_layout()
    fig.savefig(RESULTS / "best_val_acc_comparison.png", dpi=150)
    plt.close(fig)

    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix - selected model on the test set")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ---- Tables and summary.
    import pandas as pd
    pd.DataFrame(rows).to_csv(RESULTS / "results.csv", index=False)
    # Per-epoch histories, so the curves in the report and notebook can
    # be checked against numbers, not just pictures.
    (RESULTS / "histories.json").write_text(json.dumps(histories, indent=2))
    summary = {"selected_config": best["config"],
               "selected_width": best["width"],
               "selected_activation": best["activation"],
               "selected_best_epoch": best["best_epoch"],
               "selected_best_val_acc": best["best_val_acc"],
               "test_acc": test_acc, "test_loss": test_loss,
               "confusion_matrix": cm.tolist(),
               "fixed_hyperparameters": {
                   "epochs": epochs, "batch_size": BATCH_SIZE,
                   "lr": LEARNING_RATE, "optimizer": OPTIMIZER,
                   "model_seed": MODEL_SEED, "shuffle_seed": SHUFFLE_SEED},
               "training_time_note": (
                   "time.perf_counter() around each run's train() call, "
                   "including after-epoch train/validation evaluation.")}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote results/results.csv, results/summary.json and 3 figures.")
    return rows, histories, best, summary


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path.home() / "workspace" / "cifar_data")
    run_all(data_dir)
