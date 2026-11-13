"""qlearn — бекенд тренажера Q-навчання (без GUI).

Модулі:
    environment — середовище-сітка (граф клітинок), винагорода, досяжність, оптимальна вартість;
    agent       — базовий клас агента: спільний контракт для таблиці та DQN;
    tabular     — табличне Q-навчання (перенесене з q_learning_trainer.py без зміни поведінки).
"""
from .environment import DEFAULT_ENV, DIRS, GOAL_BONUS, GridEnv
from .tabular import TabularAgent

__all__ = ["DEFAULT_ENV", "DIRS", "GOAL_BONUS", "GridEnv", "TabularAgent"]
