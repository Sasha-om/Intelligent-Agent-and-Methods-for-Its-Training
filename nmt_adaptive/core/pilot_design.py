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
