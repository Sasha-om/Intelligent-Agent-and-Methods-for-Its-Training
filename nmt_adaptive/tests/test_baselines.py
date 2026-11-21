"""Тести базових стратегій: однаковий інтерфейс, лише дозволені дії, поведінка Staircase / CAT / правила."""
import random

import numpy as np
import pytest

import config
from core.ability import best_level_for
from core.agent import QAgent
from core.baselines import (CATPolicy, QPolicy, RandomPolicy, RuleBreakPolicy, StaircasePolicy, WithBreaks,
                            make_policies)
from core.env import AdaptiveEnv
from sim.student_sim import Response, VirtualStudent

A = {name: i for i, name in enumerate(config.ACTIONS)}


def _env(seed: int = 0) -> AdaptiveEnv:
    """Середовище з віртуальним учнем."""
    return AdaptiveEnv(VirtualStudent("weak_tiring", random.Random(seed)), random.Random(seed), gain_source="true")


def _play(policy: object, seed: int = 0) -> list:
    """Одна сесія стратегії; повертає список (дія, інфо)."""
    env = _env(seed)
    state, done, log = env.reset(), False, []
    policy.reset(env)
    while not done:
        action = policy.choose_action(state, env.valid_actions())
        assert action in env.valid_actions()
        state, _r, done, info = env.step(action)
        policy.observe(info)
        log.append((action, info))
    return log


@pytest.mark.parametrize("policy", [RandomPolicy(np.random.default_rng(0)), StaircasePolicy(), CATPolicy(),
                                    RuleBreakPolicy(6), RuleBreakPolicy(5)])
def test_common_interface_and_valid_actions(policy: object) -> None:
    """Кожна стратегія має name, reset(env), choose_action, observe і обирає лише дозволені дії."""
    assert isinstance(policy.name, str) and policy.name
    for seed in range(5):
        _play(policy, seed)


def test_staircase_moves_level_by_answers() -> None:
    """Staircase: старт 0; правильно → +1, помилка → −1 (у межах −2…2); перерв не дає."""
    policy = StaircasePolicy()
    policy.reset(_env())
    assert policy.level == 0
    policy.observe({"item": {"subtopic": "domain"}, "correct": True})
    policy.observe({"item": {"subtopic": "domain"}, "correct": True})
    policy.observe({"item": {"subtopic": "domain"}, "correct": True})
    assert policy.level == 2
    policy.observe({"item": {"subtopic": "domain"}, "correct": False})
    assert policy.level == 1
    policy.observe({"item": None, "correct": None})                  # перерва рівень не змінює
    assert policy.level == 1
    assert all(a != A["short_break"] for a, _i in _play(StaircasePolicy()))


def test_staircase_picks_action_closest_to_its_level() -> None:
    """Staircase обирає ту з дій easier/same/harder, чий рівень найближчий до його власного рівня."""
    env = _env()
    env.reset()
    policy = StaircasePolicy()
    policy.reset(env)
    policy.level = -2
    action = policy.choose_action(env.state(), env.valid_actions())
    options = {a: env.level_if(A[a]) for a in ("easier", "same", "harder")}
    assert abs(env.level_if(action) + 2) == min(abs(lv + 2) for lv in options.values())


def test_cat_targets_level_nearest_to_estimate() -> None:
    """CAT: рівень завдання — найближчий до θ̂ підтеми (класичний адаптивний тест); перерв не дає."""
    for action, info in _play(CATPolicy(), seed=3):
        assert action != A["short_break"]
    env = _env(1)
    env.reset()
    cat = CATPolicy()
    cat.reset(env)
    want = best_level_for(env.estimator.theta_blended(env.peek_subtopic()))
    chosen = env.level_if(cat.choose_action(env.state(), env.valid_actions()))
    assert abs(chosen - want) == min(abs(env.level_if(A[a]) - want) for a in ("easier", "same", "harder"))


def test_rule_break_policy() -> None:
    """Правило-k: «same», а перерва — щойно k завдань без перерви (якщо перерва ще доступна)."""
    log = _play(RuleBreakPolicy(5))
    breaks_after = [info["items_done"] for a, info in log if a == A["short_break"]]
    assert breaks_after[:3] == [5, 10, 15] or len(log) < 16
    assert all(a in (A["same"], A["short_break"]) for a, _i in log)


def test_q_policy_wraps_frozen_agent(tmp_path) -> None:
    """QPolicy — заморожений агент з тим самим інтерфейсом; observe нічого не робить."""
    agent = QAgent(np.random.default_rng(0))
    policy = QPolicy(QAgent.load(agent.save(tmp_path / "p.json"), np.random.default_rng(0)), "q_agent")
    assert policy.agent.frozen
    _play(policy)


def test_make_policies_from_files() -> None:
    """make_policies: 4 бази + 2 правила + Q-агент і абляції з data/ (ті файли, що є)."""
    policies = make_policies(np.random.default_rng(0))
    names = [p.name for p in policies]
    assert names[:7] == ["random", "staircase", "cat", "rule_6", "rule_5", "cat_break_6", "staircase_break_6"]
    assert "q_agent" in names and len(names) == len(set(names))


def test_random_policy_is_reproducible() -> None:
    """Random з тим самим seed → ті самі дії."""
    a = [x for x, _ in _play(RandomPolicy(np.random.default_rng(4)))]
    b = [x for x, _ in _play(RandomPolicy(np.random.default_rng(4)))]
    assert a == b
    assert Response(True, 1.0, False).correct


def test_with_breaks_adds_breaks_to_any_policy() -> None:
    """WithBreaks(CAT, 6): перерва після 6 завдань без перерви, інакше — дія CAT; observe доходить до внутрішньої."""
    policy = WithBreaks(StaircasePolicy(), 6)
    log = _play(policy, seed=2)
    breaks_after = [info["items_done"] for a, info in log if a == A["short_break"]]
    assert breaks_after and breaks_after[0] == 6 and policy.name == "staircase_break_6"
    assert any(a in (A["easier"], A["harder"]) for a, _i in log)          # драбинка всередині працює
