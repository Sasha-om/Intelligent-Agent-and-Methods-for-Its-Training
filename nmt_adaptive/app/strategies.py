"""Три режими навчальної сесії пілоту (етап 10): Q-агент, CAT, фіксований тест. Без streamlit.

Однаковий метод next(env, state) → (дія, завдання або None, запасна?). Усі йдуть через той самий AdaptiveEnv,
тож θ̂, втома і журнал рахуються однаково; різниться лише вибір завдань.
"""
import numpy as np

import config
from app.pilot import fixed_plan
from core.agent import QAgent
from core.baselines import CATPolicy
from core.env import BREAK, AdaptiveEnv

SAME = config.ACTIONS.index("same")


class QStrategy:
    """Наш Q-агент: заморожена політика; пояснення і перерви — як навчений."""

    name, explain = "q_agent", True

    def __init__(self, agent: QAgent) -> None:
        """agent — QAgent.load(...) (заморожений)."""
        if not agent.frozen:
            raise ValueError("У застосунку агент має бути замороженим (QAgent.load).")
        self.agent = agent

    def start(self, env: AdaptiveEnv) -> None:
        """Нічого готувати не треба."""

    def next(self, env: AdaptiveEnv, state: int) -> tuple:
        """Дія агента; для завдання — env.next_item, для перерви — None."""
        action = self.agent.choose_action(state, env.valid_actions())
        item = None if action == BREAK else env.next_item(action)
        return action, item, bool(self.agent.last_was_fallback)


class CATStrategy:
    """Класичний адаптивний тест: рівень, найближчий до θ̂ підтеми (core/baselines.CATPolicy). Без перерв."""

    name, explain = "cat", False

    def start(self, env: AdaptiveEnv) -> None:
        """CAT читає θ̂ із середовища."""
        self.policy = CATPolicy()
        self.policy.reset(env)

    def next(self, env: AdaptiveEnv, state: int) -> tuple:
        """Дія з потрібним рівнем → завдання."""
        action = self.policy.choose_action(state, env.valid_actions())
        return action, env.next_item(action), False


class FixedStrategy:
    """Звичайний тест: однаковий для всіх список завдань (app/pilot.fixed_plan). Без перерв."""

    name, explain = "fixed", False

    def __init__(self, items: list) -> None:
        """items — fixed_plan(bank, exclude=тестові завдання)."""
        self.items = items

    def start(self, env: AdaptiveEnv) -> None:
        """Список уже готовий."""

    def next(self, env: AdaptiveEnv, state: int) -> tuple:
        """Наступне завдання списку."""
        return SAME, env.next_given_item(self.items[env.items_done]), False


def make_strategy(arm: str, bank: list, exclude: list = ()) -> object:
    """Стратегія за назвою режиму (config.PILOT_ARMS)."""
    if arm == "q_agent":
        return QStrategy(QAgent.load(config.POLICY_PATH, np.random.default_rng(0)))
    if arm == "cat":
        return CATStrategy()
    if arm == "fixed":
        return FixedStrategy(fixed_plan(bank, exclude))
    raise ValueError(f"Невідомий режим {arm!r}; є {config.PILOT_ARMS}")
