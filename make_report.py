"""Build report/Assignment3_Report.pdf from results/results.csv and
results/summary.json, with the figures produced by the experiments.

Usage:
    python make_report.py [--interpretation-file results/interpretation.txt]

The interpretation text is the same text used in the notebook, so the
two submission files never disagree about what the results mean.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent
TEAM = ("Ashish Kumar Maurya (2611AI26), Rajeev Kumar (2611AI44), "
        "Aman Raj (2611AI59)")


def p(text, style):
    return Paragraph(text, style)


def build(interpretation):
    results = pd.read_csv(ROOT / "results" / "results.csv")
    summary = json.loads((ROOT / "results" / "summary.json").read_text())
    out = ROOT / "report" / "Assignment3_Report.pdf"
    out.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=18 * mm,
                            title="Assignment 3: Neural Networks",
                            author=TEAM)
    styles = getSampleStyleSheet()
    h1, h2, body = styles["Heading1"], styles["Heading2"], styles["BodyText"]
    story = []

    story.append(p("Assignment 3: Neural Networks", h1))
    story.append(p(f"Course: CS4101 | CS6103<br/>Team: {TEAM}", body))
    story.append(Spacer(1, 4 * mm))

    story.append(p("1. Introduction", h1))
    story.append(p(
        "This report describes a feedforward neural network and "
        "backpropagation implemented from scratch in NumPy to classify "
        "CIFAR-10 images. Only fully connected layers are used. No "
        "convolutional networks, pretrained models, transfer learning, "
        "automatic differentiation or ready-made model-training "
        "components are used anywhere. Keras is used only to load "
        "CIFAR-10, as permitted, and Pandas, Matplotlib, Seaborn and "
        "scikit-learn are used only for tables, plots and evaluation "
        "metrics, as permitted. The full implementation is in the "
        "submitted notebook, and the code is kept deliberately simple "
        "so that every choice can be explained.", body))

    story.append(p("2. Data and preprocessing", h1))
    story.append(p(
        "CIFAR-10 contains 50,000 training images and 10,000 test "
        "images, each 32 x 32 pixels with three colour channels, in ten "
        "classes (0 Airplane, 1 Automobile, 2 Bird, 3 Cat, 4 Deer, "
        "5 Dog, 6 Frog, 7 Horse, 8 Ship, 9 Truck). The split follows "
        "the assignment literally: one generator, "
        "<i>np.random.default_rng(42)</i>; classes processed in label "
        "order 0-9; each class's original indices taken in ascending "
        "order; 1,000 chosen per class with "
        "<i>rng.choice(..., size=1000, replace=False)</i>; the first "
        "800 used for training and the remaining 200 for validation; "
        "the generator never reset between classes; the subsets "
        "concatenated in class order. The resulting 8,000 training and "
        "2,000 validation images are reused for every experiment. "
        "Pixels are converted to float and divided by 255 only, giving "
        "the range [0, 1]; no further normalisation or standardisation "
        "is applied. All colour channels are retained and each image "
        "is flattened to a vector of 3,072 features. All 10,000 "
        "official test images are reserved for the final evaluation "
        "and are not used for hyperparameter tuning.", body))

    story.append(p("3. Question 1: Forward propagation", h1))
    story.append(p(
        "The network accepts 3,072 input features, supports one or two "
        "hidden layers of any specified width, supports sigmoid and "
        "ReLU hidden activations, and uses ten output neurons with a "
        "numerically stable softmax: the largest score in each row is "
        "subtracted before exponentiating, which prevents overflow "
        "without changing the result. For a batch of B images the "
        "output is a (B, 10) matrix of probabilities, and every row "
        "sums to 1, which the notebook verifies directly.", body))
    story.append(p(
        "Parameter initialisation: He initialisation, scale "
        "sqrt(2 / fan_in), for ReLU hidden layers, because ReLU "
        "discards about half its inputs and the weights therefore "
        "start a little larger to keep signals alive; Xavier "
        "initialisation, scale sqrt(1 / fan_in), for sigmoid hidden "
        "layers, because larger weights would push the sigmoid into "
        "its flat, saturated ends where learning almost stops; Xavier "
        "for the output layer in both cases, so starting probabilities "
        "are close to uniform. All biases start at zero.", body))
    param_rows = [["Architecture", "W1 / b1", "W2 / b2", "Parameters"]]
    for width in (32, 64, 128):
        w1 = 3072 * width + width
        w2 = width * 10 + 10
        param_rows.append([f"3072 -> {width} -> 10",
                           f"{3072 * width:,} / {width}",
                           f"{width * 10:,} / 10", f"{w1 + w2:,}"])
    table = Table([[p(c, body) for c in row] for row in param_rows])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story.append(table)
    story.append(p(
        "For 3072 -> 64 -> 10 in detail: W1 is (3072, 64) and b1 is "
        "(64,), giving 196,672 parameters; W2 is (64, 10) and b2 is "
        "(10,), giving 650 parameters; the total is <b>197,322</b> "
        "trainable parameters.", body))

    story.append(p("4. Question 2: Backpropagation and training", h1))
    story.append(p(
        "The loss is the mean categorical cross-entropy over a batch, "
        "computed stably by clipping probabilities away from zero "
        "before taking the logarithm. Gradients are derived by hand "
        "for every weight matrix and bias vector. For softmax combined "
        "with cross-entropy, the gradient at the output layer is "
        "(predicted probability - true one-hot label) / batch size; "
        "earlier layers follow by the chain rule, passing the error "
        "back through each layer's weights and then through the "
        "previous activation function's derivative. Correctness is "
        "checked in the test suite by a numerical gradient check: "
        "nudging each parameter by a tiny amount and comparing the "
        "measured loss slope with the analytic gradient (agreement "
        "within 1e-5).", body))
    story.append(p(
        "Both plain mini-batch SGD and momentum mini-batch gradient "
        "descent are implemented. Momentum uses v = 0.9 v + gradient "
        "and theta = theta - learning_rate * v, with each velocity "
        "initialised to zero at the start of a run and retained "
        "across batches and epochs. The training data is shuffled "
        "before each epoch with one generator (seed 123) that keeps "
        "advancing. After each epoch the current model is evaluated "
        "on the complete training and validation subsets, and loss "
        "and accuracy for both are recorded. Model initialisation "
        "uses seed 123. The implementation is configurable: width, "
        "activation, learning rate, batch size and epochs are "
        "arguments to the same training loop, so experiments require "
        "no rewriting of the loop.", body))
    story.append(p(
        "Demonstrated training configuration (used for all comparison "
        "runs): one hidden layer, momentum SGD, learning rate 0.02, "
        "batch size 128, 20 epochs.", body))

    story.append(p("5. Question 3: Experiments and interpretation", h1))
    story.append(p(
        "Three hidden-layer widths (32, 64, 128) were combined with "
        "both activations (sigmoid, ReLU), giving six configurations. "
        "All other hyperparameters were fixed for the comparison. "
        "Training time was measured with time.perf_counter() around "
        "each run's whole train() call, including the after-epoch "
        "training and validation evaluations.", body))
    exp_rows = [["Run", "Width", "Activation", "Parameters",
                 "Best val. acc.", "Best epoch", "Time (s)"]]
    for _, row in results.iterrows():
        exp_rows.append([str(int(row["run"])), str(int(row["width"])),
                         row["activation"], f"{int(row['parameters']):,}",
                         f"{row['best_val_acc']:.4f}",
                         str(int(row["best_epoch"])),
                         f"{row['training_time_s']:.1f}"])
    table = Table([[p(c, body) for c in row] for row in exp_rows])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
    story.append(table)
    story.append(p(
        f"Selection rule: highest validation accuracy, then fewer "
        f"parameters, then the earlier run. The selected model is "
        f"<b>{summary['selected_config']}</b>, with best validation "
        f"accuracy {summary['selected_best_val_acc']:.4f} at epoch "
        f"{summary['selected_best_epoch']}. Within that run, the "
        f"checkpoint saved is the earliest one achieving that "
        f"accuracy, because weights are saved only on a strict "
        f"improvement in validation accuracy.", body))
    for figure in ("selected_run_curves.png", "best_val_acc_comparison.png"):
        path = ROOT / "results" / figure
        if path.exists():
            story.append(Image(str(path), width=160 * mm, height=53 * mm))
            story.append(Spacer(1, 3 * mm))
    story.append(p("Weights &amp; Biases", h2))
    story.append(p(
        "Every run was tracked with Weights &amp; Biases in the "
        "project <i>cifar10-mlp-from-scratch</i> (configuration, "
        "per-epoch loss and accuracy, best validation accuracy and "
        "training time). In the environment where these runs were "
        "first executed, no Weights &amp; Biases account was "
        "connected, so the runs were logged in offline mode and no "
        "public project link exists yet. After <i>wandb login</i>, "
        "running `wandb sync` on the offline run folders after `wandb login` uploads them; the project link is: "
        "<b>[paste after online sync - see README]</b>. No link has "
        "been invented here.", body))
    story.append(p("Interpretation", h2))
    story.append(p(interpretation, body))

    story.append(p("6. Question 4: Evaluation of the selected model", h1))
    story.append(p(
        f"The selected checkpoint was restored into a fresh model "
        f"and evaluated once on the official 10,000-image test set. "
        f"Test accuracy is <b>{summary['test_acc']:.4f}</b> "
        f"(test loss {summary['test_loss']:.4f}). No model or "
        f"hyperparameter was changed based on the test result.", body))
    story.append(p(
        "The 10 x 10 confusion matrix below uses class names on both "
        "axes, with true labels on the vertical axis and predicted "
        "labels on the horizontal axis. Its clearest pattern is that "
        "vehicles are mostly confused with other vehicles "
        "(automobile with truck, airplane with ship) and animals with "
        "other animals (cat with dog, deer with frog and horse). "
        "Horse is the best-recognised class (636 of 1,000 correct) "
        "and cat the worst (155 of 1,000 correct).", body))
    cm_path = ROOT / "results" / "confusion_matrix.png"
    if cm_path.exists():
        story.append(Image(str(cm_path), width=150 * mm, height=133 * mm))
    story.append(p(
        "Limitations of flattened pixels with a fully connected "
        "network: flattening discards the two-dimensional layout, so "
        "the network does not know which pixels are neighbours and "
        "cannot learn local patterns such as edges or shapes "
        "directly. It is not translation invariant: the same object "
        "shifted by a few pixels looks completely different to it, "
        "so it must relearn the object at every position. It is also "
        "parameter-hungry - 3,072 inputs times the hidden width is a "
        "very large first layer for only 8,000 training images - "
        "which invites overfitting. Convolutional networks address "
        "all three by sharing weights across positions and preserving "
        "locality, but they are outside what this assignment asks for.", body))

    story.append(p("7. Reproducibility", h1))
    story.append(p(
        "Dependencies are listed in requirements.txt. Random seeds: "
        "data split 42, model initialisation 123, shuffling 123. The "
        "notebook runs from beginning to end and reproduces the "
        "numerical results within floating-point tolerance; only "
        "execution times are expected to vary. The test suite "
        "(tests/test_mlp.py) can be run with "
        "<i>python -m pytest tests/ -v</i> and checks the forward "
        "pass, softmax, parameter counts, gradients, both optimisers "
        "and the exact data split.", body))

    doc.build(story)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpretation-file", default=None)
    args = parser.parse_args()
    if args.interpretation_file:
        text = Path(args.interpretation_file).read_text(encoding="utf-8")
    else:
        text = ("[Interpretation to be filled in from the experiment "
                "results.]")
    build(text)
