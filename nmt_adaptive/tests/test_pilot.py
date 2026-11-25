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
from core.pilot_design import CoverageGuard, exam_slots, fixed_slots, exam_subtopics


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
    assert len({it["id"] for it in tests}) == 2 * len(config.PILOT_TEST_SLOTS)


def test_fixed_plan_same_for_all(bank) -> None:
    """Фіксований тест: однаковий список, слоти з core/pilot_design, без повторів і без тестових завдань."""
    tests = pilot.make_form("A", bank) + pilot.make_form("B", bank)
    plan = pilot.fixed_plan(bank, tests)
    assert [it["id"] for it in plan] == [it["id"] for it in pilot.fixed_plan(bank, tests)]
    assert [(it["subtopic"], int(it["difficulty"])) for it in plan] == fixed_slots()
    assert len({it["id"] for it in plan}) == config.PILOT_TRAIN_ITEMS
    assert not {pilot.item_key(it) for it in plan} & {pilot.item_key(it) for it in tests}


def test_normalized_gain() -> None:
    """Нормалізована зміна від −1 до 1: зріс — /(макс − вхід); упав — /вхід; 0→0 і макс→макс — None."""
    assert pilot.normalized_gain(2, 4, total=6) == pytest.approx(0.5)
    assert pilot.normalized_gain(2, 4) == pytest.approx(2 / (len(config.PILOT_TEST_SLOTS) - 2))
    assert pilot.normalized_gain(5, 0) == pytest.approx(-1)
    assert pilot.normalized_gain(3, 1) == pytest.approx(-2 / 3)
    assert pilot.normalized_gain(3, 3) == 0
    top = len(config.PILOT_TEST_SLOTS)
    assert pilot.normalized_gain(top, top) is None and pilot.normalized_gain(0, 0) is None


def test_pilot_test_scores_and_times(bank) -> None:
    """Тест: усі слоти, бал = кількість правильних, час кожної відповіді окремо, після кінця — помилка."""
    clock, n = FakeClock(), len(config.PILOT_TEST_SLOTS)
    test = pilot.PilotTest("pre", "A", pilot.make_form("A", bank), clock)
    records = []
    for k in range(n):
        clock.now += 10 + k
        item = test.current
        chosen = item["answer"] if k % 2 == 0 else (item["answer"] + 1) % 4
        records.append(test.answer(chosen))
    assert test.done and test.score == n // 2 and test.current is None
    assert [r["time_sec"] for r in records] == [10 + k for k in range(n)]
    assert [r["slot"] for r in records] == list(range(1, n + 1)) and records[0]["correct"] is True
    with pytest.raises(RuntimeError):
        test.answer(0)


def test_pilot_test_review(bank) -> None:
    """Розбір тесту: для кожної відповіді — завдання, обраний варіант, правильно чи ні; до кінця — лише відповідені."""
    test = pilot.PilotTest("post", "B", pilot.make_form("B", bank), FakeClock())
    first, second = test.current, test.items[1]
    test.answer(first["answer"])
    test.answer((second["answer"] + 1) % 4)
    review = test.review()
    assert [(it["id"], ok) for it, _, ok in review] == [(first["id"], True), (second["id"], False)]
    assert review[1][1] == (second["answer"] + 1) % 4


def test_easier_slots_first() -> None:
    """Тест етапу 12: 6 тем × 2 рівні, спершу всі легші, теми ті самі в обох половинах."""
    slots = list(config.PILOT_TEST_SLOTS)
    half = len(slots) // 2
    assert len(slots) == 12 and len(set(slots)) == 12
    assert all(lvl < 0 for _, lvl in slots[:half]) and all(lvl > 0 for _, lvl in slots[half:])
    assert [s for s, _ in slots[:half]] == [s for s, _ in slots[half:]]


def test_item_picker_excludes_tests(bank) -> None:
    """Тренування не дає завдань з тими самими числами, що в тестах, навіть коли банк вичерпано."""
    tests = pilot.make_form("A", bank) + pilot.make_form("B", bank)
    picker, rng = ItemPicker(bank, tests), random.Random(0)
    keys = {pilot.item_key(it) for it in tests}
    got = [picker(s, lvl, rng) for s, lvl in exam_slots() * 5]
    assert not {pilot.item_key(it) for it in got} & keys


def _session(arm: str, bank: list, seed: int = 0, right: int = 2, cover: bool = True) -> LiveSession:
    """Навчальна сесія пілоту в режимі arm з «вхідним тестом» (перші right правильні) і охопленням тем тесту."""
    tests = pilot.make_form("A", bank) + pilot.make_form("B", bank)
    warm = [(s, lvl, k < right) for k, (s, lvl) in enumerate(exam_slots())]
    return LiveSession(make_strategy(arm, bank, tests), bank, {}, seed, clock=FakeClock(),
                       session_items=config.PILOT_TRAIN_ITEMS, warm=warm, exclude=tests,
                       must_cover=exam_subtopics() if cover else None)


def _run_session(session: LiveSession, rng: random.Random) -> list:
    """Проходить сесію (перерви — пропускає); повертає рішення для кожного завдання."""
    decisions = []
    while not session.done:
        decision = session.decide()
        if decision["action"] == "short_break":
            session.take_break(None)
            continue
        decisions.append((decision["action"], session.item["subtopic"]))
        session.answer(rng.randrange(4))
    return decisions


def test_coverage_guard_forces_only_at_the_end() -> None:
    """Тема примусова лише тоді, коли пропущених тем стільки ж, скільки лишилося завдань."""
    guard = CoverageGuard(["a", "b"])
    assert guard.forced(0, 14) is None and guard.forced(11, 14) is None
    assert guard.forced(12, 14) == "a"
    guard.seen("a")
    assert guard.forced(12, 14) is None and guard.forced(13, 14) == "b"
    guard.seen("b")
    assert guard.forced(13, 14) is None and exam_subtopics() == list(dict.fromkeys(s for s, _ in exam_slots()))


@pytest.mark.parametrize("arm", config.PILOT_ARMS)
def test_every_arm_covers_tested_topics(arm, bank) -> None:
    """Етап 12: у кожному режимі кожна тема вихідного тесту трапляється в тренуванні (різні учні, різні вхідні бали)."""
    for seed in range(12):
        decisions = _run_session(_session(arm, bank, seed, right=seed % 13), random.Random(seed))
        assert len(decisions) == config.PILOT_TRAIN_ITEMS
        assert set(exam_subtopics()) <= {subtopic for _, subtopic in decisions}


def test_forced_topic_is_explained(bank) -> None:
    """Примусова тема має дію «coverage» і пояснення, однакове для всіх режимів; без охоплення — її може не бути."""
    forced, rng = [], random.Random(0)
    for seed in range(20):
        session = _session("q_agent", bank, seed, right=6)
        while not session.done:
            decision = session.decide()
            if decision["action"] == "short_break":
                session.take_break(None)
                continue
            if decision["action"] == "coverage":
                forced.append(decision)
            session.answer(session.item["answer"] if rng.random() < 0.6 else (session.item["answer"] + 1) % 4)
    assert forced, "за 20 сесій охоплення мало хоч раз спрацювати"
    assert all("однакове для всіх режимів" in d["reason"] and d["state_view"] is None for d in forced)


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
