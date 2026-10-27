# Assignment 3: Neural Networks

**Course:** CS4101 | CS6103
**Team:** Ashish Kumar Maurya (2611AI26), Rajeev Kumar (2611AI44), Aman Raj (2611AI59)

This is the text version of the submitted PDF report
(`Assignment3_Report.pdf`, in the same folder and in the submission
ZIP). The PDF and this file contain the same results.

## 1. Introduction

A feedforward neural network and backpropagation implemented from
scratch in NumPy to classify CIFAR-10 images. Only fully connected
layers are used. No convolutional networks, pretrained models,
transfer learning, automatic differentiation or ready-made
model-training components are used anywhere. Keras is used only to
load CIFAR-10, as permitted, and Pandas, Matplotlib, Seaborn and
scikit-learn are used only for tables, plots and evaluation metrics,
as permitted.

## 2. Data and preprocessing

CIFAR-10 contains 50,000 training images and 10,000 test images, each
32 x 32 pixels with three colour channels, in ten classes
(0 Airplane, 1 Automobile, 2 Bird, 3 Cat, 4 Deer, 5 Dog, 6 Frog,
7 Horse, 8 Ship, 9 Truck). The split follows the assignment
literally: one generator, `np.random.default_rng(42)`; classes in
order 0-9; each class's original indices in ascending order; 1,000
chosen per class with `rng.choice(..., size=1000, replace=False)`;
the first 800 for training and the remaining 200 for validation; the
generator never reset between classes; subsets concatenated in class
order. Result: 8,000 training and 2,000 validation images, reused
for every experiment. Pixels are converted to float and divided by
255 only (range [0, 1]); no further normalisation. All colour
channels are retained and each image is flattened to 3,072 features.
All 10,000 official test images are reserved for the final
evaluation and are not used for hyperparameter tuning.

## 3. Question 1: Forward propagation

The network accepts 3,072 input features, supports one or two hidden
layers of any specified width, supports sigmoid and ReLU hidden
activations, and uses ten output neurons with a numerically stable
softmax (the largest score in each row is subtracted before
exponentiating). For a batch of B images the output is a (B, 10)
matrix of probabilities, and every row sums to 1.

Parameter initialisation: He, scale sqrt(2 / fan_in), for ReLU hidden
layers; Xavier, scale sqrt(1 / fan_in), for sigmoid hidden layers
and for the output layer. All biases start at zero.

| Architecture | Parameters |
|---|---|
| 3072 -> 32 -> 10 | 98,666 |
| 3072 -> 64 -> 10 | 197,322 |
| 3072 -> 128 -> 10 | 394,634 |

For 3072 -> 64 -> 10 in detail: W1 is (3072, 64) and b1 is (64,),
giving 196,672 parameters; W2 is (64, 10) and b2 is (10,), giving
650 parameters; total **197,322**.

## 4. Question 2: Backpropagation and training

Loss: mean categorical cross-entropy, computed stably by clipping
probabilities away from zero before the logarithm. Gradients are
derived by hand for every weight matrix and bias vector. For
softmax combined with cross-entropy, the gradient at the output
layer is (predicted probability - true one-hot label) / batch size;
earlier layers follow by the chain rule. Correctness is checked by
a numerical gradient check in the test suite (agreement within
1e-5; the executed notebook reports 4.83e-10).

Both plain mini-batch SGD and momentum SGD are implemented.
Momentum uses v = 0.9 v + gradient and theta = theta -
learning_rate * v, with velocities starting at zero and kept across
batches and epochs within a run. Training data is shuffled before
each epoch with one generator (seed 123) that keeps advancing.
After each epoch the model is evaluated on the complete training
and validation subsets. Model initialisation uses seed 123.

Demonstrated training configuration (all comparison runs): one
hidden layer, momentum SGD, learning rate 0.02, batch size 128,
20 epochs.

## 5. Question 3: Experiments and interpretation

Three widths (32, 64, 128) x both activations (sigmoid, ReLU) = six
configurations, all other hyperparameters fixed. Training time was
measured with time.perf_counter() around each run's whole train()
call, including the after-epoch training and validation
evaluations.

| Run | Width | Activation | Parameters | Best val. acc. | Best epoch | Time (s) |
|---|---|---|---|---|---|---|
| 0 | 32 | sigmoid | 98,666 | 0.3920 | 19 | 10.3 |
| 1 | 32 | relu | 98,666 | 0.3685 | 20 | 7.9 |
| 2 | 64 | sigmoid | 197,322 | 0.4015 | 15 | 12.0 |
| 3 | 64 | relu | 197,322 | 0.3785 | 18 | 12.6 |
| 4 | 128 | sigmoid | 394,634 | 0.3970 | 19 | 16.0 |
| 5 | 128 | relu | 394,634 | 0.4105 | 18 | 16.2 |

(Times are from the second, reproducibility run; the assignment
allows execution times to vary. Accuracies reproduced exactly.)

Selection rule: highest validation accuracy, then fewer
parameters, then the earlier run. Selected model:
**run5_width128_relu**, best validation accuracy 0.4105 at epoch
18, earliest checkpoint achieving that accuracy.

Weights & Biases: every run was tracked in the project
`cifar10-mlp-from-scratch` (configuration, per-epoch loss and
accuracy, best validation accuracy, training time). In the
environment where these runs were first executed, no Weights &
Biases account was connected, so the runs were logged in offline
mode and no public project link exists yet. After `wandb login`,
running `wandb sync` on the offline run folders after `wandb login` uploads them; the project link is:
**[paste after online sync - see README]**. No link has been
invented here.

Interpretation: sigmoid beat ReLU at the two smaller widths
(32: 39.20% vs 36.85%; 64: 40.15% vs 37.85%), but ReLU won at width
128 (41.05% vs 39.70%) and took the overall best result. So the
ranking flips with width: there is no simple "one activation is
always better" story in these results, and the differences are
small. With only 2,000 validation images, one percentage point is
just 20 images, so a gap of one or two points should not be
over-interpreted. For ReLU, accuracy rose steadily with width
(36.85% to 37.85% to 41.05%), with the clear jump at 128. For
sigmoid, accuracy peaked at width 64 (40.15%) and was essentially
flat afterwards (39.70% at 128). Width on its own therefore did not
guarantee improvement; it interacted with the activation. Training
time grew mainly with width, from about 8 seconds at width 32 to
about 16 seconds at width 128, while the activation made little
consistent difference to time, which matches the first weight
matrix (3,072 x width) dominating the computation.

Underfitting or overfitting: the curves show underfitting more
than overfitting. Training accuracy itself only reaches about
43-49% across runs, far from 100%, so the models have not
memorised the training set; the task with flattened pixels and
only 8,000 training images is simply hard for this model. The
train-validation gap is modest, about 5-6 percentage points for
most runs, and for the sigmoid runs validation loss is still at or
near its best at epoch 20, suggesting they had not yet turned
downwards. The selected ReLU run is the exception that proves the
checkpoint rule matters: its training accuracy peaks at 49.1% and
validation at 41.05% at epoch 18, then both fall at epochs 19-20
(validation to 37.2%, validation loss rising from 1.699 to 1.802).
Restoring the epoch-18 checkpoint, rather than the final weights,
is what keeps the 41.05% result; using the final epoch would have
lost 3.8 points.

## 6. Question 4: Evaluation of the selected model

The selected checkpoint was restored into a fresh model and
evaluated once on the official 10,000-image test set. Test accuracy
is **0.4096** (test loss 1.6832). No model or hyperparameter was
changed based on the test result.

The 10 x 10 confusion matrix (in the PDF and notebook, with class
names and labelled true/predicted axes) shows vehicles mostly
confused with other vehicles (automobile with truck, airplane with
ship) and animals with other animals (cat with dog, deer with frog
and horse). Horse is the best-recognised class (636 of 1,000
correct) and cat the worst (155 of 1,000 correct).

Limitations of flattened pixels with a fully connected network:
flattening discards the two-dimensional layout, so the network
does not know which pixels are neighbours and cannot learn local
patterns such as edges or shapes directly. It is not translation
invariant: the same object shifted by a few pixels looks completely
different to it, so it must relearn the object at every position.
It is also parameter-hungry - 3,072 inputs times the hidden width
is a very large first layer for only 8,000 training images - which
invites overfitting. Convolutional networks address all three by
sharing weights across positions and preserving locality, but they
are outside what this assignment asks for.

## 7. Reproducibility

Dependencies are listed in requirements.txt. Random seeds: data
split 42, model initialisation 123, shuffling 123. The notebook
runs from beginning to end and reproduces the numerical results
within floating-point tolerance; only execution times are
expected to vary. The test suite (tests/test_mlp.py) can be run
with `python -m pytest tests/ -v` and checks the forward pass,
softmax, parameter counts, gradients, both optimisers and the
exact data split.
