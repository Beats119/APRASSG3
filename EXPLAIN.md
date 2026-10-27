# Explain it simply (viva guide)

Read this once and you can answer "what, how and why" for every part.
Each answer is one short paragraph — that is the whole point of keeping
the code simple.

## 1. What did you build?

A feedforward neural network from scratch in NumPy that classifies
CIFAR-10 images into 10 classes. "From scratch" means we wrote the
forward pass, the loss, the gradients and the weight updates ourselves.
No ready-made model and no automatic differentiation.

## 2. Why fully connected only, and why flattened pixels?

Because the assignment says so, and because it is the point of the
exercise: to understand backpropagation, you build the simplest network
that can still learn. A 32×32×3 image becomes a list of 3,072 numbers.
Every input number connects to every hidden unit. That is why the
first weight matrix is so big: 3,072 × hidden width.

## 3. Why divide by 255 and nothing else?

Pixel values are 0–255. Dividing by 255 puts them in [0, 1], so the
first layer's inputs are all on a similar, small scale and gradient
steps behave sensibly. The assignment forbids further normalisation,
so we do not subtract a mean or divide by a standard deviation, even
though that sometimes helps in real projects.

## 4. How was the data split? Why exactly this way?

One generator, `np.random.default_rng(42)`. Go through classes 0 to 9
in order. For each class, take its original indices in ascending
order and choose 1,000 without replacement. First 800 become training,
the other 200 validation. Never reset the generator between classes,
or every class would get the same "random" pattern. Concatenating in
class order gives 8,000 train + 2,000 validation, and the same split
is reused for all six experiments so the comparison is fair.

## 5. Why is the test set never touched until the end?

If you pick a model because it scores well on the test set, the test
set has secretly become training data and its score stops being an
honest measure. So: tune on validation, then evaluate the chosen
model on the test set exactly once, and do not change anything based
on the result.

## 6. What happens in the forward pass?

For each layer: multiply the incoming values by the weight matrix,
add the bias, then apply the activation function. Hidden layers use
ReLU (negatives become 0) or sigmoid (squashed to 0–1). The last layer
has 10 scores, one per class, and softmax turns them into
probabilities that sum to 1. Softmax is made stable by subtracting
the largest score in each row before taking exponentials — this
changes nothing mathematically but stops overflow on large scores.

## 7. How many parameters does 3072→64→10 have?

First layer: 3,072 × 64 weights + 64 biases = 196,672.
Output layer: 64 × 10 weights + 10 biases = 650.
Total: **197,322**. The same calculation gives 98,666 for width 32
and 394,634 for width 128.

## 8. How are weights initialised, and why?

Hidden layers: He initialisation for ReLU, `scale = sqrt(2 / fan_in)`,
because ReLU discards about half its inputs, so weights start a bit
larger to keep the signal alive. Xavier initialisation for sigmoid,
`scale = sqrt(1 / fan_in)`, because larger weights would push sigmoid
into its flat ends where its slope is nearly zero and learning stops.
Output layer: Xavier in both cases, so starting probabilities are
close to uniform. All biases start at zero.

## 9. What is the loss, and why cross-entropy?

Mean categorical cross-entropy: for each image, take
−log(probability given to the true class) and average over the batch.
It punishes confident wrong answers heavily, which is exactly what
you want for classification. It is computed stably by clipping the
probability away from 0 before taking the log.

## 10. How does backpropagation work here?

It is the chain rule, applied backwards. For softmax + cross-entropy
together, the gradient at the output is beautifully simple:
`(predicted probability − true one-hot label) / batch size`.
From there, each earlier layer gets its gradient by passing the error
back through that layer's weights and then through its activation
function's slope. That is all of `MLP.backward()` — no magic.
We proved it is correct with a numerical gradient check in the tests:
nudge each weight by a tiny amount, measure the loss slope, and
compare it with the analytic gradient. They match to within 1e-5.

## 11. SGD vs momentum SGD — what is the difference?

Plain SGD: `weight = weight − learning_rate × gradient`.
Momentum SGD: keep a velocity per weight, `velocity = 0.9 × velocity
+ gradient`, then `weight = weight − learning_rate × velocity`.
Velocity is like momentum in physics: it smooths out noisy
mini-batch gradients and keeps moving in a consistent downhill
direction. Velocities start at zero and are kept across batches and
epochs within a run, as the assignment requires. Our experiments use
momentum SGD.

## 12. Why shuffle before each epoch?

Without shuffling, the model would see the same mini-batches in the
same order every epoch and could memorise the order instead of the
patterns. One shuffle generator (seed 123) keeps advancing across
epochs, so every epoch has a new order. The same seed is used for
every run, so runs differ only in width and activation.

## 13. How do you choose the best model?

Highest validation accuracy wins. Within a run, keep the earliest
checkpoint that reached that accuracy (saved only on strict
improvement, so the file is automatically the earliest best). If two
runs tie, prefer fewer parameters, then the earlier run. The test set
plays no part in this choice.

## 14. What do ReLU and sigmoid do differently, in practice?

ReLU's slope is 1 for positive inputs, so gradients flow well and
training is usually faster and reaches higher accuracy. Sigmoid's
slope is at most 0.25 and shrinks towards 0 for large inputs, so deep
or wide sigmoid networks can learn slowly (the vanishing-gradient
problem). Wider hidden layers can fit more, but with only 8,000
training images they also overfit more easily — that is the trade-off
the experiments measure. In our actual results, sigmoid beat ReLU at
widths 32 and 64, ReLU won at 128 and took the overall best
(41.05%), and sigmoid peaked at width 64 — so the ranking flipped
with width, and small gaps (1–2 points on 2,000 validation images)
should not be over-interpreted.

## 15. How do you tell underfitting from overfitting in the curves?

Underfitting: both training and validation accuracy stay low and the
curves are close together — the model is too small or undertrained.
Overfitting: training accuracy keeps climbing while validation
accuracy flattens or falls and the gap between them grows — the model
is memorising the training set. Our selected run shows underfitting
more than overfitting: training accuracy itself only reaches 49.1%,
so it has not memorised the training data, but after its best epoch
(18) both accuracies fall and validation loss rises, which is mild
late overfitting — exactly why we restore the epoch-18 checkpoint
instead of the final weights.

## 16. Why is a flattened fully connected network limited on CIFAR-10?

Three reasons. First, flattening throws away the 2D layout: the
network does not know which pixels are neighbours, so it cannot learn
local patterns like edges or shapes directly. Second, it is not
translation invariant: the same object shifted by a few pixels looks
completely different to it, so it must relearn the object at every
position. Third, it is parameter-hungry: 3,072 inputs × hidden units
is a huge first layer for only 8,000 training images, which invites
overfitting. That is why convolutional networks, which share weights
across positions and preserve locality, do much better — but they are
forbidden here, and that limitation is part of what this assignment
is teaching.

## 17. What did you NOT use, and why does that matter?

No convolutional layers, no pretrained weights, no transfer learning,
no automatic differentiation, no ready-made trainer. Keras is used
only to load the data, which the assignment explicitly permits.
Pandas, Matplotlib, Seaborn and scikit-learn are used only for tables,
plots and evaluation metrics, which the assignment explicitly permits.
If an examiner asks "did you just call a library model?", the answer
is: the model, the loss, the gradients and the optimiser are all in
`mlp.py`, in plain NumPy, and the tests prove the gradients are right.
