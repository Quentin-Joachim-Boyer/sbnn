"""Vérifications EXACTES du cœur (niveau 1 du plan de test)."""
from sbn_predict.core.sbf import SBF, wb, count_distinct_sbf


def test_wb_table():
    assert [wb(k) for k in (1, 2, 3, 4, 5)] == [1, 1, 2, 3, 5]
    assert wb(6) == 6 and wb(10) == 10


def test_strict_threshold_zero_input():
    # somme vide / entrée nulle -> 0 (pas > 0). L'état nul est point fixe (§0.2).
    f = SBF(weights=(1, 1, -1))
    assert f((0, 0, 0)) == 0


def test_distinct_sbf_counts_match_project():
    # comptes d'origine du projet PluripotentSBN
    assert count_distinct_sbf(3, allow_theta=False) == 32
    assert count_distinct_sbf(4, allow_theta=False) == 370


def test_theta_enlarges_function_class():
    # seuil != 0 -> strictement plus de fonctions (LTF générales)
    assert count_distinct_sbf(3, allow_theta=True) > 32


def test_theta_equals_bias():
    # AND(x0,x1) réalisable avec theta=1 (= biais -1), impossible avec theta=0.
    f = SBF(weights=(1, 1), theta=1)
    assert [f(x) for x in [(0, 0), (0, 1), (1, 0), (1, 1)]] == [0, 0, 0, 1]
