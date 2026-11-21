"""Тести оцінки знань: формула IRT (3PL), Elo-оновлення, збіжність до справжньої θ."""
import math
import random

import pytest

import config
from core.ability import AbilityEstimator, best_level_for, p_correct, sigmoid

C = config.GUESS_C


# ----------------------------------------------------------------------------- формула
def test_sigmoid_basic_and_stable() -> None:
    """σ(0) = 0,5; великі |z| не ламають обчислення."""
    assert sigmoid(0) == 0.5
    assert sigmoid(1000) == pytest.approx(1.0)
    assert sigmoid(-1000) == pytest.approx(0.0)
    assert sigmoid(2) + sigmoid(-2) == pytest.approx(1.0)


def test_p_correct_at_theta_equal_b() -> None:
    """θ = b: p = c + (1 − c)/2 = 0,625 — середина зони розвитку."""
    assert p_correct(0.7, 0.7) == pytest.approx(C + (1 - C) / 2)
    assert config.ZPD_LOW <= p_correct(0.0, 0.0) <= config.ZPD_HIGH


def test_p_correct_monotone_and_bounded() -> None:
    """p зростає з θ, спадає з b і лежить між c та 1."""
    thetas = [-3 + 0.5 * i for i in range(13)]
    values = [p_correct(t, 0.0) for t in thetas]
    assert all(a < b for a, b in zip(values, values[1:]))
    assert p_correct(0, -1) > p_correct(0, 1)
    assert all(C < v < 1 for v in values)
    assert p_correct(-50, 0) == pytest.approx(C)


# ----------------------------------------------------------------------------- оновлення
def test_correct_raises_wrong_lowers() -> None:
    """Правильна відповідь піднімає θ̂, помилка — опускає."""
    est = AbilityEstimator()
    est.update("domain", 0.0, True)
    assert est.theta("domain") > 0
    est2 = AbilityEstimator()
    est2.update("domain", 0.0, False)
    assert est2.theta("domain") < 0


def test_surprise_matters() -> None:
    """Важке розв'язав — θ̂ росте сильніше; легке не розв'язав — падає сильніше."""
    hard, easy = AbilityEstimator(), AbilityEstimator()
    hard.update("range", 2.0, True)
    easy.update("range", -2.0, True)
    assert hard.theta("range") > easy.theta("range") > 0
    miss_easy, miss_hard = AbilityEstimator(), AbilityEstimator()
    miss_easy.update("range", -2.0, False)
    miss_hard.update("range", 2.0, False)
    assert miss_easy.theta("range") < miss_hard.theta("range") < 0


def test_update_returns_p_hat_before_update() -> None:
    """update повертає p̂ ДО оновлення (саме воно йде в нагороду) і рахує за формулою Elo."""
    est = AbilityEstimator()
    p_before = est.p_hat("monotonic", 1.0)
    returned = est.update("monotonic", 1.0, True)
    assert returned == pytest.approx(p_before)
    assert est.theta("monotonic") == pytest.approx(config.ELO_K * (1 - p_before))


def test_theta_is_clipped() -> None:
    """100 правильних на найважчих завданнях не виводять θ̂ за межу шкали."""
    est = AbilityEstimator()
    for _ in range(100):
        est.update("linear_ineq", 2.0, True)
    assert est.theta("linear_ineq") == pytest.approx(config.THETA_MAX)
    for _ in range(200):
        est.update("linear_ineq", -2.0, False)
    assert est.theta("linear_ineq") == pytest.approx(config.THETA_MIN)


def test_subtopics_are_independent() -> None:
    """Оновлення однієї підтеми не змінює інші."""
    est = AbilityEstimator()
    est.update("domain", 0.0, True)
    assert all(est.theta(s) == config.THETA_PRIOR for s in config.SUBTOPICS if s != "domain")


def test_unknown_subtopic_gives_clear_error() -> None:
    """Невідома підтема — зрозуміла помилка, а не тиха нова тема."""
    with pytest.raises(KeyError, match="Невідома підтема"):
        AbilityEstimator().update("trigonometry", 0.0, True)


# ----------------------------------------------------------------------------- збіжність
def _simulate(true_theta: float, steps: int, seed: int) -> list:
    """Учень зі справжньою θ відповідає на випадкові завдання; повертаємо траєкторію θ̂."""
    rng = random.Random(seed)
    est = AbilityEstimator()
    path = []
    for _ in range(steps):
        b = rng.choice(config.DIFFICULTY_LEVELS)
        est.update("domain", b, rng.random() < p_correct(true_theta, b))
        path.append(est.theta("domain"))
    return path


@pytest.mark.parametrize("true_theta", [1.0, -1.0, 0.0])
def test_estimate_converges_to_true_theta(true_theta: float) -> None:
    """За 300 відповідей середнє θ̂ на останніх 100 кроках — у межах ±0,4 від справжньої θ."""
    for seed in range(5):
        tail = _simulate(true_theta, 300, seed)[-100:]
        assert abs(sum(tail) / len(tail) - true_theta) < 0.4, seed


# ----------------------------------------------------------------------------- допоміжне
def test_overall_uses_answered_subtopics() -> None:
    """overall — середнє по підтемах з відповідями; без відповідей — стартова θ."""
    est = AbilityEstimator()
    assert est.overall() == config.THETA_PRIOR
    est.update("domain", 0.0, True)
    est.update("range", 0.0, False)
    expected = (est.theta("domain") + est.theta("range")) / 2
    assert est.overall() == pytest.approx(expected)


def test_weakest_subtopic() -> None:
    """Найслабша — з найменшою θ̂ серед тих, де були відповіді; без відповідей — випадкова, але відтворювана."""
    est = AbilityEstimator()
    est.update("domain", 0.0, True)
    est.update("range", 0.0, False)
    est.update("monotonic", -2.0, False)
    assert est.weakest_subtopic() == "monotonic"
    fresh = AbilityEstimator()
    first = fresh.weakest_subtopic(random.Random(3))
    assert first in config.SUBTOPICS
    assert first == fresh.weakest_subtopic(random.Random(3))


def test_snapshot_is_a_copy() -> None:
    """snapshot — копія: зміна оцінок після неї не змінює збережене."""
    est = AbilityEstimator()
    snap = est.snapshot()
    est.update("domain", 0.0, True)
    assert snap["domain"] == config.THETA_PRIOR
    assert set(snap) == set(config.SUBTOPICS)


def test_best_level_for() -> None:
    """Найближчий рівень складності з −2…2."""
    assert best_level_for(0.3) == 0.0
    assert best_level_for(1.7) == 2.0
    assert best_level_for(-9.0) == -2.0
    assert best_level_for(-0.6) == -1.0
    assert math.isclose(best_level_for(0.5), 0.0) or math.isclose(best_level_for(0.5), 1.0)
