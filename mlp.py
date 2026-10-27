"""A small feedforward neural network, written from scratch in NumPy.

Plain-language idea
-------------------
An image is flattened into a list of 3,072 numbers (32 x 32 pixels x 3
colours). The network multiplies those numbers by weights, adds biases,
passes the result through a simple activation function, and repeats that
for the next layer. The last layer turns its 10 scores into
probabilities with softmax. Training means: measure how wrong the
probabilities are (cross-entropy loss), work out how each weight
contributed to that error (backpropagation, i.e. the chain rule), and
nudge every weight a little in the direction that reduces the error.

Nothing here uses a ready-made model, a ready-made trainer, or
automatic differentiation. Every gradient is written out by hand below.
"""

import numpy as np

NUM_CLASSES = 10


# ------------------------------------------------------------------ helpers
def relu(z):
    """ReLU: keep positive values, replace negatives with 0."""
    return np.maximum(0, z)


def relu_derivative(z):
    """Slope of ReLU: 1 where the input was positive, else 0."""
    return (z > 0).astype(float)


def sigmoid(z):
    """Sigmoid: squashes any number into the range (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Slope of sigmoid, written using the sigmoid value itself."""
    s = sigmoid(z)
    return s * (1 - s)


def softmax(logits):
    """Turn scores into probabilities that sum to 1 in each row.

    Subtracting the largest score in each row first is the stability
    trick: it changes nothing mathematically (exp(a - c) / sum is the
    same as exp(a) / sum), but it stops exp() from overflowing on
    large scores.
    """
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    """Mean cross-entropy loss.

    For each image we take -log(probability given to its true class)
    and average over the batch. Clipping keeps log() away from log(0).
    """
    n = probs.shape[0]
    true_probs = probs[np.arange(n), y]
    return float(-np.mean(np.log(np.clip(true_probs, 1e-15, 1.0))))


def count_parameters(input_dim, hidden_widths, num_classes=NUM_CLASSES):
    """Count every weight and bias, layer by layer."""
    sizes = [input_dim] + list(hidden_widths) + [num_classes]
    total = 0
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        total += fan_in * fan_out + fan_out  # weights + biases
    return total


# -------------------------------------------------------------------- model
class MLP:
    """A fully connected network with one or two hidden layers.

    hidden_widths is a list: [64] means one hidden layer of 64 units,
    [64, 64] means two hidden layers of 64 units each.
    """

    def __init__(self, input_dim=3072, hidden_widths=(64,), activation="relu",
                 seed=123):
        if activation not in ("relu", "sigmoid"):
            raise ValueError("activation must be 'relu' or 'sigmoid'")
        self.activation_name = activation
        self.activation = relu if activation == "relu" else sigmoid
        self.activation_derivative = (
            relu_derivative if activation == "relu" else sigmoid_derivative
        )

        sizes = [input_dim] + list(hidden_widths) + [NUM_CLASSES]
        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        self.velocities_w, self.velocities_b = [], []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            is_output_layer = i == len(sizes) - 2
            # Initialisation, documented in the report:
            # - Hidden layers: He scale sqrt(2 / fan_in) for ReLU, because
            #   ReLU throws away about half its inputs, so weights need
            #   to start a little larger to keep signals alive.
            #   Xavier scale sqrt(1 / fan_in) for sigmoid, because larger
            #   weights would push sigmoid into its flat, saturated ends
            #   where learning almost stops.
            # - Output layer: Xavier scale in both cases, to keep the
            #   starting softmax probabilities close to uniform.
            if is_output_layer or activation == "sigmoid":
                scale = np.sqrt(1.0 / fan_in)
            else:
                scale = np.sqrt(2.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))
            # Momentum velocities always start at exactly zero.
            self.velocities_w.append(np.zeros((fan_in, fan_out)))
            self.velocities_b.append(np.zeros(fan_out))

    # ------------------------------------------------------------ forward
    def forward(self, x):
        """Return (probabilities, cache).

        cache stores, for every layer, the input that layer saw and its
        pre-activation scores. Backpropagation needs exactly those two
        things, so forward saves them instead of recomputing them.
        """
        cache = []
        current = x
        for layer in range(len(self.weights) - 1):
            scores = current @ self.weights[layer] + self.biases[layer]
            cache.append((current, scores))
            current = self.activation(scores)
        logits = current @ self.weights[-1] + self.biases[-1]
        cache.append((current, logits))
        return softmax(logits), cache

    # ---------------------------------------------------------- backward
    def backward(self, cache, probs, y):
        """Hand-written gradients for every weight and bias.

        The key step: for softmax + cross-entropy together, the gradient
        at the output is simply (predicted probability - true label)
        divided by the batch size. Everything before that is the chain
        rule, layer by layer, backwards.
        """
        batch_size = probs.shape[0]
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch_size), y] = 1.0
        # Gradient of the loss with respect to the output scores.
        dz = (probs - one_hot) / batch_size

        grads = [None] * len(self.weights)
        for layer in reversed(range(len(self.weights))):
            layer_input, _scores = cache[layer]
            dw = layer_input.T @ dz
            db = dz.sum(axis=0)
            grads[layer] = (dw, db)
            if layer > 0:
                # Pass the error back through this layer's weights,
                # then through the previous layer's activation function.
                da = dz @ self.weights[layer].T
                dz = da * self.activation_derivative(cache[layer - 1][1])
        return grads

    # ---------------------------------------------------------- optimiser
    def step(self, grads, lr, optimizer="momentum"):
        """Update every parameter once, from one mini-batch's gradients.

        Plain SGD:      weight = weight - lr * gradient
        Momentum (0.9): velocity = 0.9 * velocity + gradient
                        weight   = weight - lr * velocity
        Velocities live on the model, so they survive across batches
        and epochs within a run, exactly as the assignment requires.
        """
        for layer, (dw, db) in enumerate(grads):
            if optimizer == "sgd":
                self.weights[layer] -= lr * dw
                self.biases[layer] -= lr * db
            elif optimizer == "momentum":
                self.velocities_w[layer] = 0.9 * self.velocities_w[layer] + dw
                self.velocities_b[layer] = 0.9 * self.velocities_b[layer] + db
                self.weights[layer] -= lr * self.velocities_w[layer]
                self.biases[layer] -= lr * self.velocities_b[layer]
            else:
                raise ValueError("optimizer must be 'sgd' or 'momentum'")

    # ---------------------------------------------------------- evaluation
    def predict(self, x, batch_size=1000):
        """Predicted class for each row, in batches to save memory."""
        predictions = []
        for start in range(0, x.shape[0], batch_size):
            probs, _ = self.forward(x[start:start + batch_size])
            predictions.append(np.argmax(probs, axis=1))
        return np.concatenate(predictions)

    def evaluate(self, x, y, batch_size=1000):
        """Return (mean loss, accuracy) over a whole dataset."""
        total_loss, correct = 0.0, 0
        for start in range(0, x.shape[0], batch_size):
            xb, yb = x[start:start + batch_size], y[start:start + batch_size]
            probs, _ = self.forward(xb)
            total_loss += cross_entropy_loss(probs, yb) * len(yb)
            correct += int(np.sum(np.argmax(probs, axis=1) == yb))
        return total_loss / len(y), correct / len(y)

    def save(self, path):
        """Save weights and biases so a checkpoint can be restored."""
        np.savez(path, **{f"w{i}": w for i, w in enumerate(self.weights)},
                 **{f"b{i}": b for i, b in enumerate(self.biases)})

    def load(self, path):
        """Restore weights and biases saved by save()."""
        data = np.load(path)
        self.weights = [data[f"w{i}"] for i in range(len(self.weights))]
        self.biases = [data[f"b{i}"] for i in range(len(self.biases))]


# ---------------------------------------------------------------- data split
def build_subset(images, labels, seed=42, per_class=1000, train_per_class=800):
    """Build the assignment's exact train/validation split.

    For each class, in label order 0-9: take that class's original
    indices in ascending order, choose 1,000 with one shared generator
    (seed 42, never reset between classes), use the first 800 for
    training and the remaining 200 for validation. Concatenating in
    class order gives 8,000 training and 2,000 validation images.
    """
    rng = np.random.default_rng(seed)
    train_images, train_labels, val_images, val_labels = [], [], [], []
    for label in range(10):
        class_indices = np.where(labels == label)[0]  # ascending order
        chosen = rng.choice(class_indices, size=per_class, replace=False)
        train_images.append(images[chosen[:train_per_class]])
        train_labels.append(labels[chosen[:train_per_class]])
        val_images.append(images[chosen[train_per_class:]])
        val_labels.append(labels[chosen[train_per_class:]])
    return (np.concatenate(train_images), np.concatenate(train_labels),
            np.concatenate(val_images), np.concatenate(val_labels))


# ------------------------------------------------------------------ training
def train(model, x_train, y_train, x_val, y_val, epochs, batch_size, lr,
          optimizer="momentum", shuffle_seed=123, save_best_to=None):
    """Train for a number of epochs; return per-epoch history.

    Before each epoch the training order is shuffled with one
    generator that keeps advancing (never reset per epoch). After
    each epoch the model is evaluated on the full training and
    validation sets, and both loss and accuracy are recorded.

    If save_best_to is given, the weights are saved there every time
    a new best validation accuracy is reached. Because the save only
    happens on a strict improvement, the file at the end holds the
    earliest checkpoint that achieved the run's best accuracy.
    """
    shuffle_rng = np.random.default_rng(shuffle_seed)
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    n = len(y_train)
    for _epoch in range(epochs):
        order = shuffle_rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = order[start:start + batch_size]
            xb, yb = x_train[batch_idx], y_train[batch_idx]
            probs, cache = model.forward(xb)
            grads = model.backward(cache, probs, yb)
            model.step(grads, lr=lr, optimizer=optimizer)
        train_loss, train_acc = model.evaluate(x_train, y_train)
        val_loss, val_acc = model.evaluate(x_val, y_val)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if save_best_to is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(save_best_to)
    return history
