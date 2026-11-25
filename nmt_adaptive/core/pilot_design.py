"""Дизайн порівняння режимів (етап 10): які завдання у вхідному/вихідному тесті і в фіксованому тесті.

Лише «підтема + рівень» з config — без конкретних чисел. Конкретні завдання будує app/pilot.py,
а симуляція пілоту (sim/pilot_sim.py) використовує ці самі слоти, тож урок і симуляція однакові.
"""
import config


def exam_slots() -> list:
    """Слоти вхідного/вихідного тесту: [(підтема, рівень)] — однакові для форм А і Б."""
    return [(subtopic, int(level)) for subtopic, level in config.PILOT_TEST_SLOTS]


def fixed_slots(n: int = config.PILOT_TRAIN_ITEMS) -> list:
    """Фіксований тест: n слотів, підтеми по колу (порядок SUBTOPICS), рівні по колу (PILOT_FIXED_LEVELS)."""
    subtopics, levels = list(config.SUBTOPICS), config.PILOT_FIXED_LEVELS
    return [(subtopics[i % len(subtopics)], int(levels[i % len(levels)])) for i in range(n)]


def exam_subtopics() -> list:
    """Теми вхідного/вихідного тесту (без повторів, у порядку слотів)."""
    return list(dict.fromkeys(subtopic for subtopic, _ in exam_slots()))


class CoverageGuard:
    """Обов'язкове охоплення тем: кожна тема з тесту хоча б раз трапляється в тренуванні.

    Поки завдань лишається більше, ніж непоказаних тем, режим обирає сам. Коли непоказаних тем стає стільки ж,
    скільки лишилось завдань, — наступне завдання з непоказаної теми (рівень «той самий», тобто під учня).
    """

    def __init__(self, must: list | None = None) -> None:
        """must — теми, які треба показати (за замовчуванням — теми тесту)."""
        self.missing = list(must if must is not None else exam_subtopics())

    def seen(self, subtopic: str) -> None:
        """Завдання з цієї теми вже було."""
        if subtopic in self.missing:
            self.missing.remove(subtopic)

    def forced(self, items_done: int, total: int) -> str | None:
        """Тема, яку треба дати зараз, або None (режим обирає сам)."""
        left = total - items_done
        return self.missing[0] if self.missing and len(self.missing) >= left else None
