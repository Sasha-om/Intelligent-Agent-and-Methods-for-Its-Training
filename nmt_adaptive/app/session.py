"""Сесія живого учня — логіка сторінки учня без streamlit (тому її можна тестувати pytest).

Те саме середовище AdaptiveEnv, що й у симуляції, але «учень» — людина: відповідь приходить з інтерфейсу.
Агент заморожений (ε = 0, без навчання): він лише застосовує політику, вивчену в симуляторі.
Секундомір стартує, коли завдання показали, і зупиняється при відповіді — час розбору не рахується.
"""
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import config
from core.agent import QAgent
from app.explain import headline, state_view, why_text
from core.env import BREAK, AdaptiveEnv
from core.items import is_correct, make_item, solve


@dataclass
class LiveResponse:
    """Відповідь живого учня у форматі, який розуміє AdaptiveEnv.apply_answer."""
    correct: bool
    time_sec: float
    dropped: bool = False


class LiveStudent:
    """Живий учень для AdaptiveEnv: чек-ін і самооцінку дає інтерфейс; відповіді — через LiveSession.answer."""

    def __init__(self, checkin: dict | None) -> None:
        """checkin — {"mood", "energy", "sleep"} зі значеннями 1–5 або None; {} — учень пропустив чек-ін."""
        self._checkin = {k: v for k, v in (checkin or {}).items() if v is not None}
        self.next_report = None   # самооцінка після перерви (якщо учень її вказав)

    def checkin(self) -> dict:
        """Чек-ін на початку сесії (середовище бере з нього енергію)."""
        return dict(self._checkin)

    def take_break(self) -> None:
        """Перерву учень робить сам; середовище лише скидає лічильник утоми."""

    def self_report(self) -> int | None:
        """Самооцінка після перерви (одноразово) або None — тоді середовище лишає попередню."""
        report, self.next_report = self.next_report, None
        return report

    def answer(self, subtopic: str, b: float) -> LiveResponse:
        """Живий учень відповідає через інтерфейс, а не через модель."""
        raise RuntimeError("Відповідь живого учня передається через LiveSession.answer().")


class Stopwatch:
    """Секундомір відповіді: start — коли показали завдання, stop — коли натиснули «Відповісти»."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        """clock — джерело часу в секундах (у тестах — підставний годинник)."""
        self.clock, self.started = clock, None

    def start(self) -> None:
        """Запускає відлік."""
        self.started = self.clock()

    def stop(self) -> float:
        """Зупиняє відлік і повертає секунди (не більше ANSWER_TIME_CAP_SEC: учень міг відійти)."""
        if self.started is None:
            raise RuntimeError("Секундомір не запущено.")
        elapsed, self.started = self.clock() - self.started, None
        return min(max(0.0, elapsed), float(config.ANSWER_TIME_CAP_SEC))


class ItemPicker:
    """Джерело завдань для середовища: спершу ще не бачені завдання банку, потім — нові з того самого шаблону."""

    def __init__(self, bank: list) -> None:
        """bank — список завдань з data/items.json."""
        self.bank, self.used, self.generated = bank, set(), 0

    def __call__(self, subtopic: str, level: int, rng: random.Random) -> dict:
        """Завдання підтеми subtopic рівня level (−2…2)."""
        fresh = [it for it in self.bank if it["subtopic"] == subtopic and it["difficulty"] == level
                 and it["id"] not in self.used]
        if fresh:
            item = rng.choice(fresh)
        else:
            self.generated += 1
            item = make_item(subtopic, level, rng, f"gen{self.generated:03d}")
        self.used.add(item["id"])
        return item


class LiveSession:
    """Одна сесія учня: рішення агента → завдання → відповідь → розбір → … → підсумок."""

    def __init__(self, agent: QAgent, bank: list, checkin: dict | None, seed: int,
                 clock: Callable[[], float] = time.monotonic) -> None:
        """agent — заморожена політика; seed — випадковість середовища (порядок тем, вибір завдань)."""
        if not agent.frozen:
            raise ValueError("У застосунку агент має бути замороженим (QAgent.load).")
        self.agent, self.stopwatch = agent, Stopwatch(clock)
        self.env = AdaptiveEnv(LiveStudent(checkin), random.Random(seed), item_factory=ItemPicker(bank))
        self.state = self.env.reset()
        self.item, self.decision, self.results, self.break_fallbacks = None, None, [], 0

    @property
    def done(self) -> bool:
        """Сесія закінчилась (SESSION_ITEMS завдань)."""
        return self.env.done

    def decide(self) -> dict:
        """Агент обирає дію. Якщо це завдання — воно готується і запускається секундомір.

        Повертає рішення: action, state, fallback, after_items, text (що далі), reason (чому), state_view.
        """
        action = self.agent.choose_action(self.state, self.env.valid_actions())
        name, fallback = config.ACTIONS[action], bool(self.agent.last_was_fallback)
        self.item = None
        if action != BREAK:
            self.item = self.env.next_item(action)
            self.stopwatch.start()
        self.decision = {"action": name, "state": self.state, "fallback": fallback,
                         "after_items": self.env.items_done, "text": headline(name, self.env, self.item),
                         "reason": why_text(name, self.env, self.item, self.topic_counts(), fallback),
                         "state_view": state_view(self.state)}
        return self.decision

    def topic_counts(self) -> dict:
        """Скільки завдань і правильних у кожній підтемі за цю сесію: {підтема: (усього, правильно)}."""
        counts = {}
        for subtopic, ok in self.results:
            done, right = counts.get(subtopic, (0, 0))
            counts[subtopic] = (done + 1, right + ok)
        return counts

    def answer(self, chosen: int) -> dict:
        """Учень обрав варіант chosen: зупиняє секундомір, оновлює оцінку знань, повертає запис і розбір."""
        if self.item is None:
            raise RuntimeError("Спершу decide(): зараз немає завдання.")
        time_sec = self.stopwatch.stop()
        item, correct = self.item, is_correct(self.item, chosen)
        self.state, _reward, _done, info = self.env.apply_answer(LiveResponse(correct, time_sec))
        self.item = None
        self.results.append((item["subtopic"], correct))
        record = {"n": self.env.items_done, "item_id": item["id"], "subtopic": item["subtopic"],
                  "difficulty": item["difficulty"], "params": item["params"], "chosen": chosen,
                  "correct": correct, "time_sec": round(time_sec, 1), "action": self.decision["action"],
                  "p_hat": info["p_hat"], "theta_hat": self.env.estimator.snapshot(),
                  "state": self.decision["state"], "fallback": self.decision["fallback"]}
        return {"record": record, "feedback": feedback(item, correct, chosen)}

    def take_break(self, report: int | None = None) -> None:
        """Учень повернувся з перерви; report — самооцінка енергії 1–5 (або None)."""
        if self.decision is None or self.decision["action"] != "short_break":
            raise RuntimeError("Агент зараз не пропонував перерву.")
        self.env.student.next_report = report
        self.state, *_ = self.env.step(BREAK)
        self.break_fallbacks += self.decision["fallback"]
        self.decision = None

    def stats(self) -> dict:
        """Числа для end_session: приріст оцінки θ̂ за сесію, перерви, запасні дії на перервах."""
        return {"theta_gain": self.env.session_gain(), "breaks": self.env.breaks,
                "break_fallbacks": self.break_fallbacks}

    def summary(self) -> dict:
        """Підсумок для учня: скільки розв'язано, що вдалося, над чим попрацювати."""
        by_topic = self.topic_counts()
        good = [s for s, (done, right) in by_topic.items() if done == right]
        shaky = sorted((s for s, (done, right) in by_topic.items() if right < done),
                       key=self.env.estimator.theta_blended)[:config.SUMMARY_WEAK_TOPICS]
        names = {s: config.SUBTOPICS[s][1] for s in by_topic}
        return {"items": len(self.results), "right": sum(ok for _, ok in self.results), "by_topic": by_topic,
                "names": names, "good": good, "weak": shaky}


def feedback(item: dict, correct: bool, chosen: int | None = None) -> dict:
    """Розбір: правильно — один рядок; помилка — твоя і правильна відповіді, кроки розв'язку, типова помилка."""
    if correct:
        return {"correct": True, "text": item["explanation"]}
    sol = solve(item)
    mine = None if chosen is None else item["options"][chosen]
    return {"correct": False, "chosen": mine, "answer": sol["answer"], "steps": sol["steps"],
            "typical_error": sol["typical_error"]}
