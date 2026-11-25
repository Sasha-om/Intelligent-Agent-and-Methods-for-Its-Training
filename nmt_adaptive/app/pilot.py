"""Пілот-порівняння режимів (етап 10) без streamlit: тести А/Б, бали, план фіксованого тесту, статистика.

Вхідний і вихідний тест — PILOT_TEST_SLOTS завдань (етап 12: 12) з тих самих слотів (core/pilot_design.py), форми А і Б різняться лише числами.
Форми однакові для всіх учнів, тож бали можна порівнювати між режимами.
"""
import json
import random
import time
from collections.abc import Callable

import config
from app.session import Stopwatch
from core.items import is_correct, make_item
from core.stats import ancova, diff_ci, hedges_g, mean_ci, t_critical   # noqa: F401 (для app/)
from core.pilot_design import fixed_slots, exam_slots


def item_key(item: dict) -> tuple:
    """Ключ «ті самі числа»: (підтема, параметри) — щоб тести не перетиналися з навчальними завданнями."""
    return item["subtopic"], json.dumps(item["params"], sort_keys=True, default=str)


def make_form(form: str, bank: list) -> list:
    """Форма тесту ("A" або "B"): завдання слотів exam_slots(); числа — з seed форми; не збігаються з банком."""
    taken = {item_key(it) for it in bank}
    items = []
    for k, (subtopic, level) in enumerate(exam_slots(), 1):
        for attempt in range(config.PILOT_FORM_TRIES):
            rng = random.Random(config.PILOT_FORM_SEEDS[form] * 1000 + k * 10 + attempt)
            item = make_item(subtopic, level, rng, f"test{form}{k}")
            if item_key(item) not in taken:
                break
        taken.add(item_key(item))
        items.append(item)
    return items


def fixed_plan(bank: list, exclude: list = ()) -> list:
    """Фіксований тест: однаковий для всіх список завдань із банку (перше за id для кожного слота)."""
    taken = {item_key(it) for it in exclude}
    used, plan = set(), []
    for i, (subtopic, level) in enumerate(fixed_slots()):
        fits = sorted((it for it in bank if it["subtopic"] == subtopic and it["difficulty"] == level
                       and it["id"] not in used and item_key(it) not in taken), key=lambda it: it["id"])
        item = fits[0] if fits else make_item(subtopic, level, random.Random(config.PILOT_FIXED_SEED + i), f"fix{i + 1:02d}")
        used.add(item["id"])
        plan.append(item)
    return plan


def normalized_gain(pre: int, post: int, total: int = len(config.PILOT_TEST_SLOTS)) -> float | None:
    """Нормалізована зміна (Marx, Cummings 2007): від −1 до 1.

    Зріс: (вихід − вхід) / (максимум − вхід); упав: (вихід − вхід) / вхід; не змінився: 0;
    вхід і вихід обидва 0 або обидва максимум — None (зміну виміряти неможливо).
    """
    if pre == post:
        return None if pre in (0, total) else 0.0
    return (post - pre) / (total - pre) if post > pre else (post - pre) / pre


class PilotTest:
    """Вхідний або вихідний тест: без розбору; секундомір іде від показу завдання до відповіді."""

    def __init__(self, phase: str, form: str, items: list, clock: Callable[[], float] = time.monotonic) -> None:
        """phase — "pre" або "post"; form — "A"/"B"; items — make_form(form, bank)."""
        self.phase, self.form, self.items = phase, form, items
        self.results, self.chosen, self.stopwatch = [], [], Stopwatch(clock)
        self.stopwatch.start()

    @property
    def done(self) -> bool:
        """Усі завдання тесту розв'язано."""
        return len(self.results) >= len(self.items)

    @property
    def current(self) -> dict | None:
        """Поточне завдання (або None, якщо тест завершено)."""
        return None if self.done else self.items[len(self.results)]

    def answer(self, chosen: int) -> dict:
        """Відповідь на поточне завдання → запис для storage.save_test_answer; секундомір — на наступне."""
        item, time_sec = self.current, self.stopwatch.stop()
        if item is None:
            raise RuntimeError("Тест уже завершено.")
        correct = is_correct(item, chosen)
        self.results.append((item["subtopic"], int(item["difficulty"]), correct))
        self.chosen.append(chosen)
        if not self.done:
            self.stopwatch.start()
        return {"phase": self.phase, "form": self.form, "slot": len(self.results), "item_id": item["id"],
                "subtopic": item["subtopic"], "difficulty": item["difficulty"], "chosen": chosen,
                "correct": correct, "time_sec": round(time_sec, 1)}

    def review(self) -> list:
        """Розбір тесту після завершення: [(завдання, обраний варіант, правильно?)] для відповідей, що вже є."""
        return [(item, chosen, ok) for item, chosen, (_, _, ok) in zip(self.items, self.chosen, self.results)]

    @property
    def score(self) -> int:
        """Скільки правильних."""
        return sum(ok for _, _, ok in self.results)
