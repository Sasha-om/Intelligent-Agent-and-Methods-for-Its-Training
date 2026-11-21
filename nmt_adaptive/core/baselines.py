"""Базові стратегії для порівняння з Q-агентом (усі — з тим самим інтерфейсом).

Інтерфейс:
    reset(env)                       — перед сесією (env — лише для читання: наступна підтема, рівень дії);
    choose_action(state, valid)      — номер дії серед дозволених;
    observe(info)                    — що сталося після кроку (Staircase запам'ятовує правильність).

Середовище дає дії лише відносно рівня «під учня» (етап 5), тому Staircase і CAT обирають
ту з дій easier / same / harder, що дає рівень, найближчий до бажаного (env.level_if).
"""
import numpy as np

import config
from core.ability import best_level_for
from core.agent import QAgent

ACTION = {name: i for i, name in enumerate(config.ACTIONS)}
LEVEL_ACTIONS = [ACTION["easier"], ACTION["same"], ACTION["harder"]]
MIN_LEVEL, MAX_LEVEL = int(min(config.DIFFICULTY_LEVELS)), int(max(config.DIFFICULTY_LEVELS))


def action_for_level(env: object, wanted: float) -> int:
    """Та з дій easier/same/harder, після якої рівень завдання найближчий до wanted (при рівних — same)."""
    order = [ACTION["same"], ACTION["easier"], ACTION["harder"]]
    return min(order, key=lambda a: abs(env.level_if(a) - wanted))


class RandomPolicy:
    """Випадкова дозволена дія (нижня межа: «без жодної стратегії»)."""

    name = "random"

    def __init__(self, rng: np.random.Generator) -> None:
        """rng — генератор випадковості стратегії."""
        self.rng = rng

    def reset(self, env: object) -> None:
        """Нічого не запам'ятовує."""

    def choose_action(self, state: int, valid: list) -> int:
        """Випадкова дія серед дозволених."""
        return int(valid[int(self.rng.integers(len(valid)))])

    def observe(self, info: dict) -> None:
        """Нічого не запам'ятовує."""


class StaircasePolicy:
    """«Драбинка»: власний рівень −2…2; правильно → +1, помилка → −1. Без оцінки знань і без перерв."""

    name = "staircase"

    def reset(self, env: object) -> None:
        """Старт із середнього рівня 0."""
        self.env, self.level = env, 0

    def choose_action(self, state: int, valid: list) -> int:
        """Дія, що дає рівень, найближчий до рівня драбинки."""
        return action_for_level(self.env, self.level)

    def observe(self, info: dict) -> None:
        """Після відповіді рівень драбинки зсувається на 1 (після перерви — без змін)."""
        if info.get("item") is None:
            return
        step = 1 if info["correct"] else -1
        self.level = max(MIN_LEVEL, min(MAX_LEVEL, self.level + step))


class CATPolicy:
    """Класичний адаптивний тест: завдання зі складністю, найближчою до оцінки знань θ̂ підтеми. Без перерв."""

    name = "cat"

    def reset(self, env: object) -> None:
        """Запам'ятовує середовище (звідти береться θ̂)."""
        self.env = env

    def choose_action(self, state: int, valid: list) -> int:
        """Дія з рівнем, найближчим до θ̂ наступної підтеми."""
        theta = self.env.estimator.theta_blended(self.env.peek_subtopic())
        return action_for_level(self.env, best_level_for(theta))

    def observe(self, info: dict) -> None:
        """Оцінку знань оновлює саме середовище."""


class RuleBreakPolicy:
    """Просте правило: «те саме», а перерва — щойно k завдань без перерви (якщо вона ще доступна)."""

    def __init__(self, k: int) -> None:
        """k — після скількох завдань без перерви робити перерву."""
        self.k, self.name = k, f"rule_{k}"

    def reset(self, env: object) -> None:
        """Запам'ятовує середовище (звідти — лічильник завдань без перерви)."""
        self.env = env

    def choose_action(self, state: int, valid: list) -> int:
        """Перерва, якщо пора і можна; інакше «те саме»."""
        if self.env.since_break >= self.k and ACTION["short_break"] in valid:
            return ACTION["short_break"]
        return ACTION["same"]

    def observe(self, info: dict) -> None:
        """Нічого не запам'ятовує."""


class WithBreaks:
    """Будь-яка стратегія + перерва після k завдань без перерви (чесніша база: перерви — не «привілей» Q-агента)."""

    def __init__(self, inner: object, k: int) -> None:
        """inner — стратегія без перерв (CAT, Staircase); k — після скількох завдань перерва."""
        self.inner, self.k, self.name = inner, k, f"{inner.name}_break_{k}"

    def reset(self, env: object) -> None:
        """Скидає внутрішню стратегію й запам'ятовує середовище."""
        self.env = env
        self.inner.reset(env)

    def choose_action(self, state: int, valid: list) -> int:
        """Перерва, якщо пора і можна; інакше — дія внутрішньої стратегії."""
        if self.env.since_break >= self.k and ACTION["short_break"] in valid:
            return ACTION["short_break"]
        return self.inner.choose_action(state, valid)

    def observe(self, info: dict) -> None:
        """Передає подію внутрішній стратегії (Staircase зсуває рівень)."""
        self.inner.observe(info)


class QPolicy:
    """Заморожений Q-агент у тому самому інтерфейсі."""

    def __init__(self, agent: QAgent, name: str, ablation: str = "none") -> None:
        """agent — завантажена (заморожена) політика; name — назва в таблицях;
        ablation — у яких умовах агент навчався (так само його треба й оцінювати: env_kwargs)."""
        self.agent, self.name = agent, name
        self.env_kwargs = config.ABLATIONS[ablation]

    def reset(self, env: object) -> None:
        """Агент нічого не запам'ятовує між кроками (усе — у стані)."""

    def choose_action(self, state: int, valid: list) -> int:
        """Найкраща дія з таблиці Q (або запасна — у невідомому стані)."""
        return self.agent.choose_action(state, valid)

    def observe(self, info: dict) -> None:
        """Заморожений агент не вчиться."""


def make_policies(rng: np.random.Generator, ablations: bool = True) -> list:
    """Усі стратегії: 3 класичні бази, 2 правила, CAT і Staircase з перервою, Q-агент і (за бажання) абляції."""
    policies = [RandomPolicy(rng), StaircasePolicy(), CATPolicy(), RuleBreakPolicy(6), RuleBreakPolicy(5),
                WithBreaks(CATPolicy(), 6), WithBreaks(StaircasePolicy(), 6)]
    files = {"none": ("q_agent", config.POLICY_PATH)}
    if ablations:
        files.update({a: (f"q_{a}", config.DATA_DIR / f"policy_{a}.json") for a in config.ABLATIONS if a != "none"})
    for ablation, (name, path) in files.items():
        if path.exists():
            policies.append(QPolicy(QAgent.load(path, np.random.default_rng(0)), name, ablation))
    return policies
