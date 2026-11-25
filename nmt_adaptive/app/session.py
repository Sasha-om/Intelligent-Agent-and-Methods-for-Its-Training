"""Сесія живого учня — логіка сторінки учня без streamlit (тому її можна тестувати pytest).

Те саме середовище AdaptiveEnv, що й у симуляції, але «учень» — людина: відповідь приходить з інтерфейсу.
Агент заморожений (ε = 0, без навчання): він лише застосовує політику, вивчену в симуляторі.
Секундомір стартує, коли завдання показали, і зупиняється при відповіді — час розбору не рахується.
"""
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass

import config
from core.agent import QAgent
from app.explain import headline, plain_headline, state_view, why_text
from core.env import BREAK, AdaptiveEnv
from core.pilot_design import CoverageGuard
from core.items import is_correct, make_item, solve
from core.walkthrough import walkthrough


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

    def __init__(self, bank: list, exclude: list = ()) -> None:
        """bank — завдання з data/items.json; exclude — завдання, яких не можна повторювати (тести пілоту)."""
        self.bank, self.used, self.generated = bank, set(), 0
        self.excluded = {_numbers(it) for it in exclude}

    def __call__(self, subtopic: str, level: int, rng: random.Random) -> dict:
        """Завдання підтеми subtopic рівня level (−2…2), що не збігається з виключеними."""
        fresh = [it for it in self.bank if it["subtopic"] == subtopic and it["difficulty"] == level
                 and it["id"] not in self.used and _numbers(it) not in self.excluded]
        if fresh:
            item = rng.choice(fresh)
        else:
            item = self._generate(subtopic, level, rng)
        self.used.add(item["id"])
        return item

    def _generate(self, subtopic: str, level: int, rng: random.Random) -> dict:
        """Нове завдання з шаблону (повторюємо, якщо числа збіглися з виключеними)."""
        while True:
            self.generated += 1
            item = make_item(subtopic, level, rng, f"gen{self.generated:03d}")
            if _numbers(item) not in self.excluded:
                return item


def _numbers(item: dict) -> tuple:
    """«Ті самі числа»: (підтема, параметри) — як app/pilot.item_key."""
    return item["subtopic"], json.dumps(item["params"], sort_keys=True, default=str)


class LiveSession:
    """Одна сесія учня: рішення агента → завдання → відповідь → розбір → … → підсумок."""

    def __init__(self, strategy: object, bank: list, checkin: dict | None, seed: int,
                 clock: Callable[[], float] = time.monotonic, session_items: int | None = None,
                 warm: list | None = None, exclude: list = (), must_cover: list | None = None) -> None:
        """strategy — режим (app/strategies.py) або заморожений QAgent; seed — випадковість середовища.

        session_items — довжина сесії (у пілоті 14); warm — відповіді вхідного тесту для стартової θ̂;
        exclude — завдання тестів, які не можна повторювати в тренуванні;
        must_cover — теми, які мають хоч раз трапитися в тренуванні (теми вихідного тесту), або None.
        """
        if isinstance(strategy, QAgent):
            from app.strategies import QStrategy     # тут, щоб не було циклічного імпорту
            strategy = QStrategy(strategy)
        self.strategy, self.stopwatch = strategy, Stopwatch(clock)
        self.env = AdaptiveEnv(LiveStudent(checkin), random.Random(seed), item_factory=ItemPicker(bank, exclude),
                               session_items=session_items)
        self.state = self.env.reset()
        if warm:
            self.state = self.env.warm_start(warm)
        strategy.start(self.env)
        self.guard = CoverageGuard(must_cover) if must_cover is not None else None
        self.item, self.decision, self.results, self.break_fallbacks = None, None, [], 0

    @property
    def done(self) -> bool:
        """Сесія закінчилась (env.session_items завдань)."""
        return self.env.done

    def decide(self) -> dict:
        """Режим обирає дію. Якщо це завдання — воно готується і запускається секундомір.

        Повертає рішення: action, state, fallback, after_items, text (що далі), reason (чому), state_view.
        Пояснення «чому» і «що бачив агент» — лише в Q-агента (strategy.explain); в інших — тема і складність.
        """
        state, counts = self.state, self.topic_counts()
        forced = self.guard.forced(self.env.items_done, self.env.session_items) if self.guard else None
        if forced is not None:
            return self._decide_forced(forced, state)
        action, self.item, fallback = self.strategy.next(self.env, state)
        name = config.ACTIONS[action]
        if self.item is not None:
            self.stopwatch.start()
            if self.guard:
                self.guard.seen(self.item["subtopic"])
        self.decision = {"action": name if self.strategy.name != "fixed" else "fixed", "state": state,
                         "fallback": fallback, "after_items": self.env.items_done}
        if self.strategy.explain:
            self.decision.update(text=headline(name, self.env, self.item), state_view=state_view(state),
                                 reason=why_text(name, self.env, self.item, counts, fallback))
        else:
            self.decision.update(text=plain_headline(self.item), reason="", state_view=None)
        return self.decision

    def _decide_forced(self, subtopic: str, state: int) -> dict:
        """Обов'язкове охоплення: тема з вихідного тесту, якої ще не було; рівень — «той самий» (під учня)."""
        self.item = self.env.next_item(config.ACTIONS.index("same"), subtopic=subtopic)
        self.guard.seen(subtopic)
        self.stopwatch.start()
        self.decision = {"action": "coverage", "state": state, "fallback": False, "after_items": self.env.items_done,
                         "text": plain_headline(self.item), "state_view": None,
                         "reason": f"Теми «{self.item['subtopic_name']}» ще не було, а вона є у вихідному тесті — "
                                   "тому даємо її зараз (це правило однакове для всіх режимів)."}
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
    """Розбір: правильно — один рядок + покроковий розв'язок «за бажанням»; помилка — твоя і правильна відповіді,
    покроковий розв'язок (з формулами, малюнками і перевіркою на прикладі) і типова помилка."""
    steps = walkthrough(item)["steps"]
    if correct:
        return {"correct": True, "text": item["explanation"], "answer": item["options"][item["answer"]],
                "walkthrough": steps}
    sol = solve(item)
    mine = None if chosen is None else item["options"][chosen]
    return {"correct": False, "chosen": mine, "answer": sol["answer"], "steps": sol["steps"],
            "typical_error": sol["typical_error"], "walkthrough": steps}
