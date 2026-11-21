"""Тести віртуального учня (симуляція): відтворюваність, типи, навчання в зоні, втома, відмова, самооцінка."""
import random
import statistics

import pytest

import config
from core.ability import p_correct
from sim.student_sim import Response, VirtualStudent, clone, dropout_chance, make_population

TYPES = list(config.STUDENT_TYPES)


def _student(type_key: str = "strong_enduring", seed: int = 0) -> VirtualStudent:
    """Один учень із фіксованим seed."""
    return VirtualStudent(type_key, random.Random(seed))


# ----------------------------------------------------------------------------- відтворюваність
def test_same_seed_same_student_and_answers() -> None:
    """Той самий seed → ті самі приховані параметри й ті самі відповіді."""
    a, b = _student(seed=5), _student(seed=5)
    assert a.true_theta() == b.true_theta() and a.fatigue() == b.fatigue()
    for _ in range(15):
        assert a.answer("domain", 0.0) == b.answer("domain", 0.0)


def test_different_seed_different_student() -> None:
    """Інший seed → інший учень."""
    assert _student(seed=1).true_theta() != _student(seed=2).true_theta()


def test_unknown_type_gives_clear_error() -> None:
    """Невідомий тип учня — зрозуміла помилка."""
    with pytest.raises(KeyError, match="Невідомий тип"):
        VirtualStudent("genius", random.Random(0))


# ----------------------------------------------------------------------------- типи учнів
def test_types_differ_in_knowledge_and_fatigue_rate() -> None:
    """Сильні знають більше; стомлювані втомлюються швидше (на 200 учнях кожного типу)."""
    def mean_theta(t: str) -> float:
        """Середня справжня θ учнів типу t."""
        return statistics.mean(statistics.mean(s.true_theta().values()) for s in make_population(t, 200, 0))

    def mean_rate(t: str) -> float:
        """Середня швидкість втоми учнів типу t."""
        return statistics.mean(s.fatigue_rate for s in make_population(t, 200, 0))

    assert mean_theta("strong_enduring") > mean_theta("weak_enduring") + 1.0
    assert mean_rate("strong_tiring") > mean_rate("strong_enduring") + 0.03
    assert all(s.fatigue_rate >= config.SIM_MIN_FATIGUE_RATE for s in make_population("weak_enduring", 300, 1))


def test_population_is_reproducible_and_varied() -> None:
    """make_population: n учнів, той самий seed → ті самі учні, учні між собою різні."""
    first, again = make_population("weak_tiring", 20, 3), make_population("weak_tiring", 20, 3)
    assert len(first) == 20
    assert [s.true_theta() for s in first] == [s.true_theta() for s in again]
    assert len({round(s.true_theta()["domain"], 6) for s in first}) == 20


# ----------------------------------------------------------------------------- модель відповіді
def test_true_probability_uses_effective_theta() -> None:
    """P = p_correct(θ − k·F, b): втома знижує ефективні знання."""
    s = _student()
    theta, fatigue = s.true_theta()["range"], s.fatigue()
    assert s.true_probability("range", 1.0) == pytest.approx(p_correct(theta - config.SIM_FATIGUE_K * fatigue, 1.0))


def test_easy_items_answered_more_often() -> None:
    """Частка правильних на b = −2 більша, ніж на b = 2 (500 відповідей, учні без втоми)."""
    def share(b: float) -> float:
        """Частка правильних відповідей 500 різних слабких учнів на завдання складності b."""
        hits = 0
        for i in range(500):
            s = _student("weak_enduring", i)
            hits += s.answer("domain", b).correct
        return hits / 500
    assert share(-2.0) > share(2.0) + 0.3


def test_learning_only_in_zone() -> None:
    """θ підтеми росте на SIM_LEARN_RATE лише коли P у [0,5; 0,85]; інші підтеми не змінюються."""
    s = _student("strong_enduring", 7)
    theta = s.true_theta()["domain"]
    b_zone = theta - config.SIM_FATIGUE_K * s.fatigue() - 0.3        # P ≈ 0,7 — у зоні
    assert config.SIM_LEARN_LOW <= s.true_probability("domain", b_zone) <= config.SIM_LEARN_HIGH
    before = s.true_theta()
    s.answer("domain", b_zone)
    after = s.true_theta()
    assert after["domain"] == pytest.approx(before["domain"] + config.SIM_LEARN_RATE)
    assert all(after[k] == before[k] for k in before if k != "domain")
    far = after["range"] + 6                                         # дуже важке: P ≈ c — поза зоною
    s.answer("range", far)
    assert s.true_theta()["range"] == after["range"]


# ----------------------------------------------------------------------------- втома і перерва
def test_fatigue_grows_and_break_reduces() -> None:
    """Кожне завдання додає втому; перерва знімає SIM_BREAK_RECOVERY частину; втома в [0; 1]."""
    s = _student("strong_tiring", 2)
    start = s.fatigue()
    s.answer("domain", 0.0)
    assert s.fatigue() == pytest.approx(min(1.0, start + s.fatigue_rate))
    for _ in range(40):
        s.answer("domain", 0.0)
    assert s.fatigue() == 1.0
    s.take_break()
    assert s.fatigue() == pytest.approx(1.0 - config.SIM_BREAK_RECOVERY)


def test_time_grows_with_fatigue() -> None:
    """Втомлений учень у середньому відповідає повільніше (ті самі завдання, 300 учнів)."""
    fresh, tired = [], []
    for i in range(300):
        s = _student("strong_enduring", i)
        fresh.append(s.answer("domain", 0.0).time_sec)
        s._fatigue = 1.0                                             # штучно «втомили» для перевірки
        tired.append(s.answer("domain", 0.0).time_sec)
    assert statistics.mean(tired) > 1.3 * statistics.mean(fresh)
    assert min(fresh + tired) > 0


# ----------------------------------------------------------------------------- відмова
def test_dropout_chance_formula() -> None:
    """Шанс відмови росте від серії помилок і високої втоми; не більший за межу."""
    base = dropout_chance(0, 0.0)
    assert base == pytest.approx(config.SIM_DROPOUT_BASE)
    assert dropout_chance(1, 0.0) == pytest.approx(base)             # одна помилка ще не серія
    assert dropout_chance(4, 0.0) == pytest.approx(base + 3 * config.SIM_DROPOUT_STREAK)
    assert dropout_chance(0, 0.9) == pytest.approx(base + config.SIM_DROPOUT_FATIGUE)
    assert dropout_chance(100, 1.0) == config.SIM_DROPOUT_MAX


def test_wrong_streak_resets_after_correct() -> None:
    """Серія помилок рахується підряд і скидається правильною відповіддю."""
    s = _student("weak_tiring", 4)
    while s.answer("domain", 2.5).correct:
        pass
    assert s.wrong_streak >= 1
    while not s.answer("domain", -3.0).correct:
        pass
    assert s.wrong_streak == 0


# ----------------------------------------------------------------------------- самооцінка
def test_self_report_and_checkin_in_range() -> None:
    """Самооцінка й чек-ін — цілі 1–5."""
    s = _student()
    assert s.self_report() in range(1, 6)
    assert set(s.checkin()) == set(config.CHECKIN_QUESTIONS)
    assert all(v in range(1, 6) for v in s.checkin().values())


def test_self_report_falls_with_fatigue() -> None:
    """Самооцінка — шумна версія втоми: кореляція «втома ↔ самооцінка» від'ємна (1 000 вибірок)."""
    rng = random.Random(0)
    fatigues, reports = [], []
    for i in range(1000):
        s = _student("strong_enduring", i)
        s._fatigue = rng.random()
        fatigues.append(s.fatigue())
        reports.append(s.self_report())
    assert statistics.correlation(fatigues, reports) < -0.6


# ----------------------------------------------------------------------------- clone
def test_clone_gives_identical_behaviour() -> None:
    """clone(учень, seed) → та сама θ і ті самі відповіді на ті самі завдання (для чесного порівняння стратегій)."""
    original = _student("weak_tiring", 11)
    a, b = clone(original, 99), clone(original, 99)
    assert a.true_theta() == original.true_theta()
    answers_a = [a.answer("domain", 0.0) for _ in range(20)]
    answers_b = [b.answer("domain", 0.0) for _ in range(20)]
    assert answers_a == answers_b
    assert original.fatigue() == clone(original, 1).fatigue()        # оригінал не змінився


def test_response_fields() -> None:
    """Відповідь має правильність, час і ознаку відмови."""
    r = _student().answer("domain", 0.0)
    assert isinstance(r, Response)
    assert isinstance(r.correct, bool) and isinstance(r.dropped, bool) and r.time_sec > 0
