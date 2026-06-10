"""Poisson scoreline maths (pure, hermetic)."""
from __future__ import annotations

import numpy as np

from src import goals


def test_poisson_pmf_normalized():
    pmf = goals._poisson_pmf(1.5, goals.DEFAULT_MAX_GOALS)
    assert abs(pmf.sum() - 1.0) < 1e-9
    assert (pmf >= 0).all()


def test_score_matrix_sums_to_one():
    m = goals.score_matrix(1.8, 1.1)
    assert m.shape == (goals.DEFAULT_MAX_GOALS + 1, goals.DEFAULT_MAX_GOALS + 1)
    assert abs(m.sum() - 1.0) < 1e-9


def test_most_likely_score_favours_higher_lambda():
    h, a = goals.most_likely_score(2.6, 0.4)
    assert h > a


def test_derive_1x2_sums_to_one_and_orders():
    m = goals.score_matrix(2.5, 0.6)
    p_home, p_draw, p_away = goals.derive_1x2(m)
    assert abs(p_home + p_draw + p_away - 1.0) < 1e-9
    assert p_home > p_away  # strong home favourite


def test_score_by_outcome_matches_predicted_winner():
    # Strong home favourite -> conditional score must be a home win.
    h, a = goals.most_likely_score_by_outcome(2.5, 0.6)
    assert h > a

    # Strong away favourite -> away win.
    h, a = goals.most_likely_score_by_outcome(0.5, 2.4)
    assert a > h


def test_score_by_outcome_consistent_with_argmax_outcome():
    # Whatever outcome the score implies must equal the argmax of the 1X2 probs.
    for lam_h, lam_a in [(1.2, 1.1), (1.6, 0.9), (0.8, 1.7), (1.0, 1.0)]:
        m = goals.score_matrix(lam_h, lam_a)
        p_home, p_draw, p_away = goals.derive_1x2(m)
        expected = max((p_home, "H"), (p_draw, "D"), (p_away, "A"))[1]
        h, a = goals.most_likely_score_by_outcome(lam_h, lam_a)
        got = "H" if h > a else ("A" if a > h else "D")
        assert got == expected
