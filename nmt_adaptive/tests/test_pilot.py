"""Тести пілоту-порівняння (етап 10): форми тестів, бали, фіксований тест, стратегії, статистика."""
import random

import numpy as np
import pytest

import config
from app import pilot
from app.session import ItemPicker, LiveSession
from app.strategies import CATStrategy, FixedStrategy, QStrategy, make_strategy
from core.agent import QAgent
from core.items import is_correct, load_bank
from core.pilot_design import fixed_slots, exam_slots


class FakeClock:
    """Підставний годинник."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(scope="module")
def bank() -> list:
    """Банк завдань."""
    return load_bank()


def test_forms_same_for_everyone_and_parallel(bank) -> None:
    """Форма однакова при кожному виклику; А і Б — ті самі слоти, різні числа."""
    a1, a2, b = pilot.make_form("A", bank), pilot.make_form("A", bank), pilot.make_form("B", bank)
    assert [it["question"] for it in a1] == [it["question"] for it in a2]
    assert [(it["subtopic"], int(it["difficulty"])) for it in a1] == exam_slots()
    assert [(it["subtopic"], int(it["difficulty"])) for it in b] == exam_slots()
    assert all(x["params"] != y["params"] for x, y in zip(a1, b))


def test_forms_do_not_repeat_bank(bank) -> None:
    """Жодне завдання тестів не збігається числами з банком (тест не «підглядає» тренування)."""
    bank_keys = {pilot.item_key(it) for it in bank}
    tests = pilot.make_form("A", bank) + pilot.make_form("B", bank)
    assert not {pilot.item_key(it) for it in tests} & bank_keys
    assert len({it["id"] for it in tests}) == 12


def test_fixed_plan_same_for_all(bank) -> None:
    """Фіксований тест: однаковий список, слоти з core/pilot_design, без повторів і без тестових завдань."""
    tests = pilot.make_form("A", bank) + pilot.make_form("B", bank)
    plan = pilot.fixed_plan(bank, tests)
    assert [it["id"] for it in plan] == [it["id"] for it in pilot.fixed_plan(bank, tests)]
    assert [(it["subtopic"], int(it["difficulty"])) for it in plan] == fixed_slots()
    assert len({it["id"] for it in plan}) == config.PILOT_TRAIN_ITEMS
    assert not {pilot.item_key(it) for it in plan} & {pilot.item_key(it) for it in tests}


def test_normalized_gain() -> None:
    """Нормалізована зміна від −1 до 1: зріс — /(6 − вхід); упав — /вхід; 0→0 і 6→6 — None."""
    assert pilot.normalized_gain(2, 4) == pytest.approx(0.5)
    assert pilot.normalized_gain(5, 0) == pytest.approx(-1)
    assert pilot.normalized_gain(3, 1) == pytest.approx(-2 / 3)
    assert pilot.normalized_gain(3, 3) == 0
    assert pilot.normalized_gain(6, 6) is None and pilot.normalized_gain(0, 0) is None


def test_pilot_test_scores_and_times(bank) -> None:
    """Тест: 6 завдань, бал = кількість правильних, час кожної відповіді окремо, після кінця — помилка."""
    clock = FakeClock()
    test = pilot.PilotTest("pre", "A", pilot.make_form("A", bank), clock)
    records = []
    for k in range(6):
        clock.now += 10 + k
        item = test.current
        chosen = item["answer"] if k % 2 == 0 else (item["answer"] + 1) % 4
        records.append(test.answer(chosen))
    assert test.done and test.score == 3 and test.current is None
    assert [r["time_sec"] for r in records] == [10, 11, 12, 13, 14, 15]
    assert [r["slot"] for r in records] == [1, 2, 3, 4, 5, 6] and records[0]["correct"] is True
    with pytest.raises(RuntimeError):
        test.answer(0)


def test_item_picker_excludes_tests(bank) -> None:
    """Тренування не дає завдань з тими самими числами, що в тестах, навіть коли банк вичерпано."""
    tests = pilot.make_form("A", bank) + pilot.make_form("B", bank)
    picker, rng = ItemPicker(bank, tests), random.Random(0)
    keys = {pilot.item_key(it) for it in tests}
    got = [picker(s, lvl, rng) for s, lvl in exam_slots() * 5]
    assert not {pilot.item_key(it) for it in got} & keys


def _session(arm: str, bank: list, seed: int = 0) -> LiveSession:
    """Навчальна сесія пілоту в режимі arm з «вхідним тестом» (2 правильні з 6)."""
    tests = pilot.make_form("A", bank) + pilot.make_form("B", bank)
    warm = [(s, lvl, k < 2) for k, (s, lvl) in enumerate(exam_slots())]
    return LiveSession(make_strategy(arm, bank, tests), bank, {}, seed, clock=FakeClock(),
                       session_items=config.PILOT_TRAIN_ITEMS, warm=warm, exclude=tests)


@pytest.mark.parametrize("arm", config.PILOT_ARMS)
def test_every_arm_gives_same_number_of_items(arm, bank) -> None:
    """Кожен режим: рівно PILOT_TRAIN_ITEMS навчальних завдань; перерви — лише в Q-агента."""
    session, rng, items, breaks = _session(arm, bank), random.Random(1), 0, 0
    while not session.done:
        decision = session.decide()
        if decision["action"] == "short_break":
            breaks += 1
            session.take_break(None)
            continue
        session.answer(rng.randrange(4))
        items += 1
    assert items == config.PILOT_TRAIN_ITEMS
    assert breaks == 0 or arm == "q_agent"


def test_fixed_arm_same_items_for_two_students(bank) -> None:
    """Звичайний тест: два різні учні (різні seed) отримують ті самі завдання в тому самому порядку."""
    def ids(seed: int) -> list:
        session, out = _session("fixed", bank, seed), []
        while not session.done:
            session.decide()
            out.append(session.item["id"])
            session.answer(seed % 4)
        return out
    assert ids(1) == ids(2)


def test_only_q_explains(bank) -> None:
    """«Чому» і «що бачив агент» — лише в Q-агента; в інших режимах — тема і складність."""
    q = _session("q_agent", bank).decide()
    while q["action"] == "short_break":
        q = _session("q_agent", bank, 5).decide()
    cat = _session("cat", bank).decide()
    fixed = _session("fixed", bank).decide()
    assert q["reason"] and q["state_view"]
    for d in (cat, fixed):
        assert d["reason"] == "" and d["state_view"] is None and d["text"].startswith("Наступне завдання: тема «")
    assert fixed["action"] == "fixed"


def test_warm_start_moves_estimate(bank) -> None:
    """Вхідний тест задає стартову θ̂: 6 з 6 → оцінка вища, ніж 0 з 6."""
    good, bad = (LiveSession(CATStrategy(), bank, {}, 0, warm=[(s, lvl, ok) for s, lvl in exam_slots()])
                 for ok in (True, False))
    assert good.env.estimator.general() > 0 > bad.env.estimator.general()


def test_strategies_reject_unknown_and_unfrozen(bank) -> None:
    """Невідомий режим і агент, що вчиться, — помилка."""
    with pytest.raises(ValueError):
        make_strategy("magic", bank)
    with pytest.raises(ValueError):
        QStrategy(QAgent(np.random.default_rng(0)))
    assert isinstance(make_strategy("fixed", bank), FixedStrategy)


def test_answers_are_checked_like_bank(bank) -> None:
    """Правильна відповідь тестового завдання — справді правильна (той самий is_correct)."""
    for item in pilot.make_form("B", bank):
        assert is_correct(item, item["answer"]) and not is_correct(item, (item["answer"] + 1) % 4)


# ----------------------------------------------------------------------------- статистика
def test_t_critical_close_to_table() -> None:
    """t-критичні значення близькі до таблиці: df=10 → 2,228; df=30 → 2,042; великий df → 1,96."""
    assert pilot.t_critical(3, 0.05) == pytest.approx(3.182, abs=0.001)
    assert pilot.t_critical(5, 0.025) == pytest.approx(3.163, abs=0.001)
    assert pilot.t_critical(1, 0.025) == pytest.approx(25.452, abs=0.001)
    assert pilot.t_critical(40, 0.05) == pytest.approx(2.021, abs=0.005)      # поза таблицею — наближення
    assert pilot.t_critical(10, 0.05) == pytest.approx(2.228, abs=0.01)
    assert pilot.t_critical(30, 0.05) == pytest.approx(2.042, abs=0.005)
    assert pilot.t_critical(10_000, 0.05) == pytest.approx(1.960, abs=0.002)


def test_ancova_recovers_known_effect() -> None:
    """Синтетичні дані: вихід = 1 + 0,5·вхід + 2·[Q] + шум → ефект ≈ 2, ДІ містить 2."""
    rng = np.random.default_rng(0)
    pre = rng.integers(0, 7, 200)
    group = np.arange(200) % 2 == 0
    post = 1 + 0.5 * pre + 2 * group + rng.normal(0, 0.5, 200)
    res = pilot.ancova(pre, post, group)
    assert res["effect"] == pytest.approx(2, abs=0.15) and res["low"] < 2 < res["high"]
    assert pilot.ancova([1, 2], [1, 2], [True, False]) is None


def test_bootstrap_and_effect_size() -> None:
    """ДІ середнього містить середнє; різниця і Hedges g — правильного знаку; замало даних → None."""
    rng = np.random.default_rng(0)
    mean, low, high = pilot.mean_ci([1, 2, 3, 4, 5], rng)
    assert low < mean == 3 < high
    diff, dlow, dhigh = pilot.diff_ci([5, 6, 7, 6], [1, 2, 3, 2], rng)
    assert diff == pytest.approx(4) and dlow > 0
    assert pilot.hedges_g([5, 6, 7], [1, 2, 3]) > 1 and pilot.hedges_g([1], [2, 3]) is None
    assert pilot.mean_ci([], rng) == (None, None, None)
