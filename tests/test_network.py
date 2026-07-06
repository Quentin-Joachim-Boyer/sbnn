import numpy as np
from sbn_predict.core.network import SBN, random_sbn


def test_zero_state_is_fixed_point_theta0():
    net = random_sbn(6, wb_max=3, rng=np.random.default_rng(1))
    assert net.attractor_sync(np.zeros(6, dtype=int)) == [tuple([0] * 6)]


def test_goles_olivos_sync_symmetric():
    # W symétrique + synchrone -> attracteurs de longueur 1 ou 2 (théorème).
    rng = np.random.default_rng(2)
    net = random_sbn(8, wb_max=3, rng=rng, symmetric=True)
    for _ in range(30):
        x0 = rng.integers(0, 2, size=8)
        cycle = net.attractor_sync(x0, max_steps=500)
        assert len(cycle) in (1, 2)


def test_clamp_holds_value():
    net = random_sbn(5, wb_max=2, rng=np.random.default_rng(3))
    net.clamped = {0: 1}
    _, cycle = net.run_sync(np.zeros(5, dtype=int))
    assert all(state[0] == 1 for state in cycle)
