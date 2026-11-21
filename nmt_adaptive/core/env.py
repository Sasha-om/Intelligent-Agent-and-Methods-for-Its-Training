"""Середовище Q-навчання: навчальна сесія як послідовність станів, дій і нагород.

Аналогія з тренажером (лабіринт):
    стан    — не клітинка, а 4 спостереження про учня → номер 0…134;
    дія     — не «вгору/вниз», а легше / те саме / складніше / повторення слабкої теми / перерва;
    епізод  — сесія до 20 завдань (або доки учень не кине).

Середовище не знає, з ким працює: з віртуальним учнем (sim/student_sim.py) чи з живим (app/).
Учня воно бачить лише через методи checkin(), answer(підтема, b), take_break(), self_report().
Справжні знання (true_theta) потрібні лише для бонусу в симуляції (gain_source="true").
"""
import random
import statistics
from collections.abc import Callable
from bisect import bisect_right
from collections import deque

import config
from core.ability import AbilityEstimator, target_level_for

BREAK = config.ACTIONS.index("short_break")
MIN_LEVEL, MAX_LEVEL = int(min(config.DIFFICULTY_LEVELS)), int(max(config.DIFFICULTY_LEVELS))
GAIN_SOURCES = ("estimate", "true")


# ----------------------------------------------------------------------------- частини стану
def ability_level(theta: float) -> int:
    """Відро знань 0…4 за межами ABILITY_EDGES (значення на межі — у верхнє відро)."""
    return bisect_right(config.ABILITY_EDGES, theta)


def fatigue_level(items_since_break: int, times: list) -> int:
    """Втома 0…2 з того, що видно: завдань без перерви (6 і 12 — пороги) + уповільнення відповідей.

    Уповільнення: середній час останніх 3 відповідей > TIME_DRIFT_RATIO × середній час перших 3 сесії → +1.
    Останні 3 мають бути ПІСЛЯ останньої перерви — інакше перерва не «знімала» б уповільнення.
    """
    level = (items_since_break >= config.FATIGUE_ITEMS_MID) + (items_since_break >= config.FATIGUE_ITEMS_HIGH)
    base_n = config.TIME_BASELINE_ITEMS
    if len(times) >= 2 * base_n and items_since_break >= base_n:
        baseline = statistics.mean(times[:base_n])
        recent = statistics.mean(times[-base_n:])
        level += recent > config.TIME_DRIFT_RATIO * baseline
    return min(config.N_FATIGUE - 1, int(level))


def accuracy_level(recent: list) -> int:
    """Точність останніх відповідей: 0 — низька (<0,4), 1 — середня, 2 — висока (≥0,8). Без відповідей — 1."""
    if not recent:
        return 1
    return bisect_right(config.ACCURACY_EDGES, sum(recent) / len(recent))


def self_level(report: int | None) -> int:
    """Самооцінка 1–5 → 0 (1–2, погано), 1 (3), 2 (4–5, добре); немає відповіді → SELF_REPORT_DEFAULT."""
    if report is None:
        return config.SELF_REPORT_DEFAULT
    return 0 if report <= 2 else (1 if report == 3 else 2)


def encode_state(ability: int, fatigue: int, accuracy: int, self_rep: int) -> int:
    """4 частини стану → один номер 0…134 (як номер рядка Q-таблиці)."""
    return ((ability * config.N_FATIGUE + fatigue) * config.N_ACCURACY + accuracy) * config.N_SELF_REPORT + self_rep


def decode_state(state: int) -> tuple:
    """Номер стану → (знання, втома, точність, самооцінка)."""
    state, self_rep = divmod(state, config.N_SELF_REPORT)
    state, accuracy = divmod(state, config.N_ACCURACY)
    ability, fatigue = divmod(state, config.N_FATIGUE)
    return ability, fatigue, accuracy, self_rep


# ----------------------------------------------------------------------------- середовище
class AdaptiveEnv:
    """Одна навчальна сесія учня. API як у тренажері: reset() → стан; step(дія) → (стан, нагорода, кінець, інфо)."""

    def __init__(self, student: object, rng: random.Random, gain_source: str = "estimate", use_fatigue: bool = True,
                 use_self_report: bool = True, allow_break: bool = True, item_factory: Callable | None = None) -> None:
        """student — віртуальний чи реальний учень; rng — випадковість середовища; прапорці — для абляцій.

        item_factory(підтема, рівень, rng) → повне завдання (для застосунку); без нього — лише підтема й рівень.
        """
        if gain_source not in GAIN_SOURCES:
            raise ValueError(f"gain_source має бути одним із {GAIN_SOURCES}, а не {gain_source!r}")
        self.student, self.rng, self.gain_source = student, rng, gain_source
        self.use_fatigue, self.use_self_report, self.allow_break = use_fatigue, use_self_report, allow_break
        self.item_factory = item_factory
        self.subtopics = list(config.SUBTOPICS)

    # ------------------------------------------------------------------ початок сесії
    def reset(self) -> int:
        """Нова сесія: чек-ін, θ̂ = стартова, лічильники = 0. Повертає початковий стан."""
        self.estimator = AbilityEstimator()
        checkin = self.student.checkin()
        self.self_report = None if not checkin else checkin.get("energy")
        self.level = None
        self.pointer = self.rng.randrange(len(self.subtopics))
        self.items_done, self.breaks, self.steps, self.since_break = 0, 0, 0, 0
        self.times, self.recent = [], deque(maxlen=config.ACCURACY_WINDOW)
        self.last_correct, self.done, self.pending = None, False, None
        self.last_subtopic, self.repeat = None, 0
        self.start_potential = self._potential()
        return self.state()

    def _potential(self) -> float:
        """«Скільки знає учень» для бонусу за приріст.

        "true" — середні справжні знання віртуального учня (лише симуляція): агент не може «накрутити» оцінку;
        "estimate" — загальна θ̂ (працює і з живим учнем).
        """
        if self.gain_source == "true":
            values = self.student.true_theta().values()
            return sum(values) / len(values)
        return self.estimator.general()

    def session_gain(self) -> float:
        """Приріст знань від початку сесії (у тих самих одиницях, що й бонус)."""
        return self._potential() - self.start_potential

    # ------------------------------------------------------------------ стан
    def state_parts(self) -> dict:
        """Частини стану, як їх бачить агент (з урахуванням абляцій). Для пояснень учню і звітів."""
        fatigue = fatigue_level(self.since_break, self.times) if self.use_fatigue else 0
        report = self_level(self.self_report) if self.use_self_report else config.SELF_REPORT_DEFAULT
        return {"ability": ability_level(self.estimator.general()), "fatigue": fatigue,
                "accuracy": accuracy_level(list(self.recent)), "self_report": report}

    def state(self) -> int:
        """Поточний номер стану 0…134."""
        p = self.state_parts()
        return encode_state(p["ability"], p["fatigue"], p["accuracy"], p["self_report"])

    def valid_actions(self) -> list:
        """Дозволені дії: перерва — лише якщо дозволена, не вичерпано MAX_BREAKS і після останньої перерви
        (або початку сесії) було хоча б MIN_ITEMS_BEFORE_BREAK завдань."""
        actions = list(range(config.N_ACTIONS))
        too_soon = self.since_break < config.MIN_ITEMS_BEFORE_BREAK
        if not self.allow_break or self.breaks >= config.MAX_BREAKS or too_soon:
            actions.remove(BREAK)
        return actions

    # ------------------------------------------------------------------ крок
    def step(self, action: int) -> tuple:
        """Один крок сесії з віртуальним учнем: дія → (новий стан, нагорода, кінець, інфо)."""
        if action == BREAK:
            return self._take_break()
        item = self.next_item(action)
        return self.apply_answer(self.student.answer(item["subtopic"], item["difficulty"]))

    def _check_action(self, action: int) -> None:
        """Помилка, якщо сесія завершена або дія недозволена."""
        if self.done:
            raise RuntimeError("Сесія вже завершена — викличте reset().")
        if action not in self.valid_actions():
            raise ValueError(f"Дія {config.ACTIONS[action]!r} зараз недоступна.")

    def _take_break(self) -> tuple:
        """Перерва: завдання немає, втома в стані скидається, самооцінку питаємо знову."""
        self._check_action(BREAK)
        self.student.take_break()
        self.breaks += 1
        self.steps += 1
        self.since_break = 0
        report = self.student.self_report()
        self.self_report = report if report is not None else self.self_report
        parts = {"zpd": 0, "two_wrong": 0, "fatigue": 0, "dropout": 0, "gain": 0}
        return self.state(), 0.0, self.done, self._info(BREAK, None, parts)

    def next_item(self, action: int) -> dict:
        """Обирає наступне завдання за дією (для застосунку — окремо від відповіді).

        Складність рахується ВІД ВИДИМОЇ оцінки: base = рівень «під учня» (p̂ ≈ 0,7) для обраної підтеми;
        easier → base − 1, same → base, harder → base + 1. Тоді та сама дія в тому самому стані
        означає те саме завдання «відносно учня» (агент не мусить пам'ятати прихований рівень).
        """
        self._check_action(action)
        name = config.ACTIONS[action]
        if name == "review_weak_topic":
            subtopic = self._review_subtopic()
        else:
            subtopic = self.peek_subtopic()
            self.pointer += 1
        self.level = self._level(subtopic, name)
        item = {"subtopic": subtopic, "difficulty": float(self.level)}
        if self.item_factory is not None:
            item = self.item_factory(subtopic, self.level, self.rng)
        fatigue_before = fatigue_level(self.since_break, self.times)
        self.pending = {"action": action, "item": item, "fatigue_before": fatigue_before,
                        "p_hat": self.estimator.p_hat_blended(subtopic, self.level), "potential": self._potential()}
        return item

    def peek_subtopic(self) -> str:
        """Підтема, яку отримають easier/same/harder на наступному кроці (лише читання, без змін)."""
        return self.subtopics[self.pointer % len(self.subtopics)]

    def _level(self, subtopic: str, name: str) -> int:
        """Рівень завдання: рівень «під учня» (p̂ ≈ 0,7) для підтеми ± 1 залежно від дії, у межах −2…2."""
        base = int(target_level_for(self.estimator.theta_blended(subtopic)))
        return max(MIN_LEVEL, min(MAX_LEVEL, base + {"easier": -1, "harder": 1}.get(name, 0)))

    def level_if(self, action: int) -> int:
        """Який рівень дасть дія easier/same/harder на наступному кроці (лише читання; для базових стратегій)."""
        name = config.ACTIONS[action]
        if name not in ("easier", "same", "harder"):
            raise ValueError("level_if — лише для easier/same/harder")
        return self._level(self.peek_subtopic(), name)

    def _review_subtopic(self) -> str:
        """Найслабша підтема за змішаною θ̂; ту саму — не більше REVIEW_MAX_REPEAT разів поспіль."""
        exclude = self.last_subtopic if self.repeat >= config.REVIEW_MAX_REPEAT else None
        return self.estimator.weakest_blended(self.rng, exclude=exclude)

    def apply_answer(self, response: object) -> tuple:
        """Приймає відповідь на завдання з next_item: оновлює θ̂ і лічильники, рахує нагороду."""
        if self.pending is None:
            raise RuntimeError("Спершу next_item(дія).")
        pend, self.pending = self.pending, None
        item = pend["item"]
        self.estimator.update(item["subtopic"], item["difficulty"], response.correct)
        parts = self._reward_parts(pend, response)
        self.items_done += 1
        self.steps += 1
        self.since_break += 1
        self.times.append(response.time_sec / (2.0 - pend["p_hat"]))   # час, «очищений» від складності
        self.recent.append(response.correct)
        self.last_correct = response.correct
        self.repeat = self.repeat + 1 if item["subtopic"] == self.last_subtopic else 1
        self.last_subtopic = item["subtopic"]
        self.done = response.dropped or self.items_done >= config.SESSION_ITEMS
        parts["gain"] = config.R_GAIN_COEF * (self._potential() - pend["potential"])
        info = self._info(pend["action"], item, parts, response, pend["p_hat"])
        return self.state(), sum(parts.values()), self.done, info

    # ------------------------------------------------------------------ нагорода
    def _reward_parts(self, pend: dict, response: object) -> dict:
        """Доданки нагороди за завдання (усі числа з config); бонус за приріст знань додається окремо (gain)."""
        in_zpd = config.ZPD_LOW <= pend["p_hat"] <= config.ZPD_HIGH
        tired = pend["fatigue_before"] == config.N_FATIGUE - 1
        return {"zpd": config.R_ZPD if in_zpd else 0,
                "two_wrong": config.R_TWO_WRONG if (not response.correct and self.last_correct is False) else 0,
                "fatigue": config.R_FATIGUE_NO_BREAK if tired else 0,
                "dropout": config.R_DROPOUT if response.dropped else 0}

    def _info(self, action: int, item: dict | None, parts: dict, response: object | None = None,
              p_hat: float | None = None) -> dict:
        """Що сталося на кроці — для журналу, звіту, пояснення учню й тестів."""
        return {"action_name": config.ACTIONS[action], "item": item, "p_hat": p_hat,
                "in_zpd": bool(parts["zpd"]), "correct": None if response is None else response.correct,
                "time_sec": None if response is None else response.time_sec,
                "dropped": bool(response and response.dropped), "theta_hat": self.estimator.general(),
                "reward_parts": parts, "items_done": self.items_done, "breaks": self.breaks}
