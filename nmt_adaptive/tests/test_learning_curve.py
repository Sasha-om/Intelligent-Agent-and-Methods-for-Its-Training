"""Тести кривої навчання (core/learning_curve.py, етап 12): відомий синтетичний ефект має знаходитися."""
import random

import numpy as np
import pytest

import config
from core import learning_curve as lc
from core.ability import p_correct


def _synthetic(sessions: int, gain_per_try: float, seed: int) -> list:
    """Учні з θ = 0; на k-й спробі теми θ = gain_per_try·(k − 1); очікування — за θ̂ вхідного тесту (= 0)."""
    rng, out = random.Random(seed), []
    estimator = lc.pre_estimator([])
    for _ in range(sessions):
        answers, tries = [], {}
        for n in range(config.PILOT_TRAIN_ITEMS):
            subtopic, level = list(config.SUBTOPICS)[n % 4], rng.choice((-1, 0, 1))
            tries[subtopic] = tries.get(subtopic, 0) + 1
            p = p_correct(gain_per_try * (tries[subtopic] - 1), level)
            answers.append((subtopic, level, rng.random() < p))
        out.append(lc.session_points(answers, estimator))
    return out


def test_points_count_tries_per_topic() -> None:
    """Спроба рахується окремо для кожної теми; очікувано — 3PL з θ̂ після вхідного тесту."""
    estimator = lc.pre_estimator([("domain", -1, True), ("domain", 1, True), ("monotonic", -1, False),
                                  ("monotonic", 1, False)])
    points = lc.session_points([("domain", 0, True), ("monotonic", 0, False), ("domain", 1, False)], estimator)
    assert [n for n, _, _ in points] == [1, 1, 2] and [ok for _, ok, _ in points] == [1.0, 0.0, 0.0]
    assert points[0][2] == pytest.approx(estimator.p_hat_blended("domain", 0))
    assert points[0][2] > points[1][2]          # domain на вході — правильно, monotonic — ні


def test_no_learning_gives_flat_curve() -> None:
    """Без навчання нахил ≈ 0 і ДІ містить 0."""
    value, low, high = lc.slope_ci(_synthetic(200, 0.0, 1), np.random.default_rng(0), repeats=400)
    assert abs(value) < 0.02 and low < 0 < high


def test_learning_is_found_and_ranked() -> None:
    """Навчання є → нахил > 0, ДІ вище нуля; швидше навчання → більший нахил і значуща різниця."""
    rng = np.random.default_rng(0)
    fast, slow = _synthetic(200, 0.6, 2), _synthetic(200, 0.15, 3)
    value, low, _ = lc.slope_ci(fast, rng, repeats=400)
    assert value > 0.05 and low > 0
    diff, d_low, _ = lc.slope_difference(fast, slow, rng, repeats=400)
    assert diff > 0 and d_low > 0


def test_curve_rows_cap_and_counts() -> None:
    """Спроби понад CURVE_MAX_OPPORTUNITY злиті в останню; кількості відповідей складаються."""
    sessions = _synthetic(30, 0.3, 4)
    rows = lc.curve_rows(sessions)
    assert [r["спроба"] for r in rows] == list(range(1, config.CURVE_MAX_OPPORTUNITY + 1))
    assert sum(r["відповідей"] for r in rows) == 30 * config.PILOT_TRAIN_ITEMS
    assert lc.slope([[(1, 1.0, 0.5)]]) is None and lc.slope_ci([], np.random.default_rng(0)) == (None, None, None)


def test_sums_match_direct_slope_and_jackknife_detects() -> None:
    """Нахил із сум = прямий нахил; jackknife-тест бачить різницю швидкого і повільного навчання."""
    fast, slow = _synthetic(150, 0.6, 5), _synthetic(150, 0.0, 6)
    sums = [lc.session_sums(s) for s in fast]
    assert lc.slope_from_sums(np.sum(sums, axis=0)) == pytest.approx(lc.slope(fast))
    value, se = lc.jackknife(sums)
    assert se is not None and value / se > 3
    diff, significant = lc.difference_test(sums, [lc.session_sums(s) for s in slow])
    assert diff > 0 and significant
    assert lc.difference_test(sums[:2], sums[:2]) == (None, False)
