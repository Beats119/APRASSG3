"""Tests written FIRST (test-driven development).

Run:  python -m pytest tests/ -v

Each test states one rule from the assignment, in plain words, so a
failure tells us exactly which rule is broken. No test uses PyTorch,
TensorFlow, Keras or autograd - the implementation must be NumPy only.
"""

import numpy as np

from mlp import (
    MLP,
    build_subset,
    count_parameters,
    cross_entropy_loss,
    softmax,
    train,
)


# ---------------------------------------------------------------- shapes
def test_forward_returns_probabilities_of_shape_batch_by_10():
    model = MLP(input_dim=20, hidden_widths=[8], activation="relu", seed=123)
    x = np.random.default_rng(0).normal(size=(5, 20))
    probs, _cache = model.forward(x)
    assert probs.shape == (5, 10)


def test_each_probability_row_sums_to_one():
    model = MLP(input_dim=20, hidden_widths=[8], activation="sigmoid", seed=123)
    x = np.random.default_rng(0).normal(size=(7, 20))
    probs, _cache = model.forward(x)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_softmax_is_stable_for_very_large_logits():
    # Subtracting the row max must happen, otherwise exp(1000) overflows.
    probs = softmax(np.array([[1000.0, 1000.0, 999.0]]))
    assert np.all(np.isfinite(probs))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_weight_and_bias_shapes_for_3072_64_10():
    model = MLP(input_dim=3072, hidden_widths=[64], activation="relu", seed=123)
    # W1: (inputs, hidden), b1: (hidden,), W2: (hidden, classes), b2: (classes,)
    assert model.weights[0].shape == (3072, 64)
    assert model.biases[0].shape == (64,)
    assert model.weights[1].shape == (64, 10)
    assert model.biases[1].shape == (10,)


def test_parameter_count_for_3072_64_10():
    # (3072*64 + 64) + (64*10 + 10) = 196672 + 650 = 197322
    assert count_parameters(3072, [64], 10) == 197322
    assert count_parameters(3072, [32], 10) == 98666
    assert count_parameters(3072, [128], 10) == 394634


def test_two_hidden_layers_use_the_same_width_in_both():
    model = MLP(input_dim=20, hidden_widths=[8, 8], activation="relu", seed=1)
    assert model.weights[0].shape == (20, 8)
    assert model.weights[1].shape == (8, 8)
    assert model.weights[2].shape == (8, 10)


def test_both_activations_run():
    for activation in ("relu", "sigmoid"):
        model = MLP(input_dim=20, hidden_widths=[8], activation=activation, seed=1)
        probs, _ = model.forward(np.zeros((2, 20)))
        assert probs.shape == (2, 10)


# ------------------------------------------------------------- gradients
def test_backprop_matches_numerical_gradients():
    """The gold-standard check: analytic gradients must match the
    slope measured by nudging each parameter a tiny amount."""
    rng = np.random.default_rng(0)
    model = MLP(input_dim=4, hidden_widths=[5], activation="sigmoid", seed=3)
    x = rng.normal(size=(3, 4))
    y = np.array([0, 2, 1])
    probs, cache = model.forward(x)
    grads = model.backward(cache, probs, y)

    eps = 1e-6
    for layer in range(len(model.weights)):
        for param, grad in ((model.weights[layer], grads[layer][0]),
                            (model.biases[layer], grads[layer][1])):
            it = np.nditer(param, flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                old = param[idx]
                param[idx] = old + eps
                loss_plus = cross_entropy_loss(model.forward(x)[0], y)
                param[idx] = old - eps
                loss_minus = cross_entropy_loss(model.forward(x)[0], y)
                param[idx] = old
                numeric = (loss_plus - loss_minus) / (2 * eps)
                assert abs(numeric - grad[idx]) < 1e-5, (layer, idx, numeric, grad[idx])
                it.iternext()


def test_cross_entropy_loss_is_correct_on_a_known_case():
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    y = np.array([0, 1])
    expected = -(np.log(0.7) + np.log(0.8)) / 2
    assert abs(cross_entropy_loss(probs, y) - expected) < 1e-9


# ------------------------------------------------------------- optimiser
def test_momentum_velocities_start_at_zero_and_then_move():
    model = MLP(input_dim=6, hidden_widths=[4], activation="relu", seed=1)
    assert all(np.all(v == 0) for v in model.velocities_w)
    x = np.random.default_rng(0).normal(size=(4, 6))
    y = np.array([0, 1, 2, 3])
    probs, cache = model.forward(x)
    grads = model.backward(cache, probs, y)
    w_before = model.weights[0].copy()
    model.step(grads, lr=0.1, optimizer="momentum")
    # First step: v = 0.9*0 + grad = grad, so W moves by exactly lr*grad.
    assert np.allclose(w_before - model.weights[0], 0.1 * grads[0][0])
    assert not np.all(model.velocities_w[0] == 0)


def test_plain_sgd_moves_by_lr_times_gradient():
    model = MLP(input_dim=6, hidden_widths=[4], activation="relu", seed=1)
    x = np.random.default_rng(0).normal(size=(4, 6))
    y = np.array([0, 1, 2, 3])
    probs, cache = model.forward(x)
    grads = model.backward(cache, probs, y)
    w_before = model.weights[0].copy()
    model.step(grads, lr=0.1, optimizer="sgd")
    assert np.allclose(w_before - model.weights[0], 0.1 * grads[0][0])


# ------------------------------------------------------------ training
def test_train_records_history_and_saves_the_best_checkpoint(tmp_path):
    """A checkpoint saved during training must, when restored, score
    exactly the run's best validation accuracy - not the final epoch's."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 20))
    # Labels from a fixed teacher, so the problem is learnable.
    y = np.argmax(x @ rng.normal(size=(20, 10)), axis=1)
    x_tr, y_tr, x_val, y_val = x[:160], y[:160], x[160:], y[160:]
    model = MLP(input_dim=20, hidden_widths=[16], activation="relu", seed=1)
    checkpoint = str(tmp_path / "best.npz")
    history = train(model, x_tr, y_tr, x_val, y_val, epochs=5,
                    batch_size=32, lr=0.05, optimizer="momentum",
                    shuffle_seed=5, save_best_to=checkpoint)
    assert len(history["val_acc"]) == 5
    restored = MLP(input_dim=20, hidden_widths=[16], activation="relu", seed=1)
    restored.load(checkpoint)
    _loss, acc = restored.evaluate(x_val, y_val)
    assert abs(acc - max(history["val_acc"])) < 1e-9


# ------------------------------------------------------------ data split
def test_subset_follows_the_assignment_rule_exactly():
    # Fake data: 10 classes, 1500 images each, labels already in order.
    n_per_class = 1500
    labels = np.repeat(np.arange(10), n_per_class)
    images = np.arange(10 * n_per_class)  # image id == its original index
    x_tr, y_tr, x_val, y_val = build_subset(images, labels, seed=42)
    assert x_tr.shape[0] == 8000 and x_val.shape[0] == 2000
    # Class order is kept: first 800 are class 0, next 800 class 1, ...
    assert np.all(y_tr[:800] == 0) and np.all(y_tr[800:1600] == 1)
    assert np.all(y_val[:200] == 0) and np.all(y_val[200:400] == 1)
    # No image is in both sets, and train+val per class = 1000 selected.
    assert len(set(x_tr.tolist()) & set(x_val.tolist())) == 0
    # Same seed -> same split (reproducible).
    again = build_subset(images, labels, seed=42)
    assert np.array_equal(again[0], x_tr) and np.array_equal(again[2], x_val)


# ------------------------------------------------------------ compliance
def test_implementation_uses_numpy_only():
    source = open("mlp.py", encoding="utf-8").read()
    for banned in ("torch", "tensorflow", "keras", "autograd", "jax"):
        assert banned not in source, f"mlp.py must not use {banned}"
