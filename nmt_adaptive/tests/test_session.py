"""Тести логіки сесії живого учня (app/session.py): секундомір, завдання, розбір, перерва, підсумок."""
import random

import numpy as np
import pytest

import config
from app.session import ItemPicker, LiveSession, LiveStudent, Stopwatch, feedback
from core.agent import QAgent
from core.env import BREAK, AdaptiveEnv
from core.items import load_bank, solve


class FakeClock:
    """Підставний годинник: час іде лише тоді, коли тест скаже."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture(scope="module")
def agent() -> QAgent:
    """Навчена заморожена політика з data/policy.json (етап 7)."""
    return QAgent.load(config.POLICY_PATH, np.random.default_rng(0))


@pytest.fixture(scope="module")
def bank() -> list:
    """Банк завдань."""
    return load_bank()


def _until_item(session: LiveSession) -> dict:
    """Рішення агента; якщо перерва — «повертаємось» і просимо знову, доки не буде завдання."""
    decision = session.decide()
    while decision["action"] == "short_break":
        session.take_break(None)
        decision = session.decide()
    return decision


def test_stopwatch_measures_and_caps() -> None:
    """Секундомір рахує різницю часу; понад ANSWER_TIME_CAP_SEC — обрізає."""
    clock = FakeClock()
    watch = Stopwatch(clock)
    watch.start()
    clock.now += 37.5
    assert watch.stop() == pytest.approx(37.5)
    watch.start()
    clock.now += 10_000
    assert watch.stop() == config.ANSWER_TIME_CAP_SEC
    with pytest.raises(RuntimeError):
        watch.stop()


def test_review_time_not_counted(agent, bank) -> None:
    """Час, поки учень читає розбір, не потрапляє в час наступної відповіді (секундомір — від показу завдання)."""
    clock = FakeClock()
    session = LiveSession(agent, bank, {}, seed=1, clock=clock)
    _until_item(session)
    clock.now += 30
    first = session.answer(0)["record"]
    clock.now += 500          # довго читає розбір — до відповіді не рахується
    _until_item(session)
    clock.now += 20
    second = session.answer(0)["record"]
    assert first["time_sec"] == pytest.approx(30) and second["time_sec"] == pytest.approx(20)


def test_session_needs_frozen_agent(bank) -> None:
    """Агент, що вчиться, у застосунок не допускається."""
    with pytest.raises(ValueError):
        LiveSession(QAgent(np.random.default_rng(0)), bank, {}, seed=0)


def test_answer_record_has_all_fields(agent, bank) -> None:
    """Запис відповіді містить усе, що чекає storage.save_answer, і правильні значення."""
    from app.storage import ANSWER_FIELDS
    session = LiveSession(agent, bank, {"energy": 4}, seed=2, clock=FakeClock())
    _until_item(session)
    item = session.item
    out = session.answer(item["answer"])
    record = out["record"]
    assert set(ANSWER_FIELDS) <= set(record)
    assert record["correct"] is True and record["n"] == 1 and record["item_id"] == item["id"]
    assert record["action"] in config.ACTIONS and set(record["theta_hat"]) == set(config.SUBTOPICS)
    assert isinstance(record["fallback"], bool) and 0 < record["p_hat"] < 1


def test_wrong_answer_gets_steps(agent, bank) -> None:
    """Помилка → покроковий розв'язок, правильна відповідь і типова помилка; правильно → один рядок."""
    session = LiveSession(agent, bank, {}, seed=3, clock=FakeClock())
    _until_item(session)
    item = session.item
    wrong = (item["answer"] + 1) % len(item["options"])
    fb = session.answer(wrong)["feedback"]
    assert fb["correct"] is False and fb["steps"] == solve(item)["steps"] and fb["typical_error"]
    assert fb["chosen"] == item["options"][wrong]
    assert fb["answer"] == item["options"][item["answer"]]
    good = feedback(item, True)
    assert good["correct"] is True and good["text"] == item["explanation"]
    assert good["answer"] == item["options"][item["answer"]]
    assert good["walkthrough"][-1]["title"] == "Відповідь" and fb["walkthrough"] == good["walkthrough"]


def test_answer_without_item_fails(agent, bank) -> None:
    """Відповісти, коли завдання немає, не можна."""
    session = LiveSession(agent, bank, {}, seed=0, clock=FakeClock())
    with pytest.raises(RuntimeError):
        session.answer(0)


def test_full_session_reaches_end(agent, bank) -> None:
    """Повна сесія: SESSION_ITEMS завдань, перерви не більше MAX_BREAKS, підсумок сходиться."""
    session = LiveSession(agent, bank, {"energy": 2}, seed=4, clock=FakeClock())
    rng = random.Random(0)
    while not session.done:
        _until_item(session)
        session.answer(rng.randrange(4))
    summary, stats = session.summary(), session.stats()
    assert summary["items"] == config.SESSION_ITEMS
    assert sum(done for done, _ in summary["by_topic"].values()) == config.SESSION_ITEMS
    assert 0 <= stats["breaks"] <= config.MAX_BREAKS and len(summary["weak"]) <= 2
    assert set(summary["good"]).isdisjoint(summary["weak"])


def test_break_resets_fatigue_and_takes_report(agent, bank) -> None:
    """Перерва: лічильник «без перерви» = 0, самооцінка після перерви потрапляє в стан."""
    session = LiveSession(agent, bank, {}, seed=5, clock=FakeClock())
    for _ in range(3):
        _until_item(session)
        session.answer(0)
    session.decision = {"action": "short_break", "fallback": False}   # наче агент запропонував перерву
    session.take_break(5)
    assert session.env.since_break == 0 and session.env.breaks == 1 and session.env.self_report == 5
    with pytest.raises(RuntimeError):
        session.take_break(None)          # друга перерва без пропозиції агента


def test_items_from_bank_first_then_generated(bank) -> None:
    """Спершу — не бачені завдання банку потрібної підтеми й рівня, потім — нові з шаблону."""
    picker, rng = ItemPicker(bank), random.Random(0)
    in_bank = [it for it in bank if it["subtopic"] == "domain" and it["difficulty"] == 0]
    got = [picker("domain", 0, rng) for _ in range(len(in_bank) + 2)]
    assert {it["id"] for it in got[:len(in_bank)]} == {it["id"] for it in in_bank}
    assert all(it["id"].startswith("gen") and it["difficulty"] == 0 for it in got[len(in_bank):])
    assert len({it["id"] for it in got}) == len(got)


def test_live_student_report_once() -> None:
    """Самооцінка після перерви віддається один раз; пропущений чек-ін — порожній."""
    student = LiveStudent({"mood": None, "energy": 3})
    assert student.checkin() == {"energy": 3} and LiveStudent(None).checkin() == {}
    student.next_report = 2
    assert student.self_report() == 2 and student.self_report() is None


def test_frozen_agent_never_learns(agent, bank) -> None:
    """Після сесії Q-таблиця агента не змінилась."""
    before = agent.q.copy()
    session = LiveSession(agent, bank, {}, seed=7, clock=FakeClock())
    for _ in range(5):
        _until_item(session)
        session.answer(1)
    assert np.array_equal(before, agent.q)
    assert BREAK == config.ACTIONS.index("short_break")


def test_decision_text_is_plain(agent, bank) -> None:
    """Текст рішення: з ACTION_TEXT_UA; «повторення» на старті не каже про «найслабшу» тему."""
    for seed in range(20):
        session = LiveSession(agent, bank, {}, seed=seed, clock=FakeClock())
        first = session.decide()
        if first["action"] == "review_weak_topic":
            assert first["text"].startswith("Починаємо з теми «") and "найслабша" not in first["text"]
        elif first["action"] != "short_break":
            assert first["text"].startswith(config.ACTION_TEXT_UA[first["action"]])
        assert session.item is None or session.item["subtopic_name"] in first["text"]


def test_review_text_is_honest(agent, bank) -> None:
    """«Повторення»: на старті — «Починаємо з теми», для не баченої теми — «якої ще не було», інакше — «найслабша»."""
    texts = []
    for seed in range(40):
        session = LiveSession(agent, bank, {}, seed=seed, clock=FakeClock())
        for _ in range(6):
            decision = session.decide()
            if decision["action"] == "short_break":
                session.take_break(None)
                continue
            seen = session.env.estimator.answers(session.item["subtopic"]) > 0
            if decision["action"] == "review_weak_topic" and session.env.items_done > 0:
                assert ("найслабша" in decision["text"]) == seen
                texts.append(decision["text"])
            session.answer(0)
    assert any("якої ще не було" in t for t in texts) and any("найслабша" in t for t in texts)


def test_fallback_explained_honestly(bank) -> None:
    """Запасна дія (стан, якого агент не бачив) пояснюється як просте правило, а не «агент урахував»."""
    agent = QAgent.load(config.POLICY_PATH, np.random.default_rng(0))
    agent.visits[:] = 0                         # наче агент не бачив жодного стану
    session = LiveSession(agent, bank, {}, seed=0, clock=FakeClock())
    decision = session.decide()
    assert decision["fallback"] is True and "запасним правилом" in decision["reason"]
