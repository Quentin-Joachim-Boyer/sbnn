import numpy as np
from sbn_predict.learning.enumerate_fit import best_sbf
from sbn_predict.regimes.associative import make_memory, recall


def test_enumerate_fit_learns_and_with_theta():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND -> nécessite theta != 0
    f, err = best_sbf(X, y, allow_theta=True)
    assert err == 0


def test_associative_recall_clean_pattern():
    rng = np.random.default_rng(0)
    patterns = rng.integers(0, 2, size=(2, 20))
    net = make_memory(patterns)
    out = recall(net, patterns[0].copy(), async_order=list(range(20)))
    # un motif propre doit être un point fixe (rappel parfait)
    assert np.array_equal(out, patterns[0])
