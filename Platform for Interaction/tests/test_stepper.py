"""Юніт-тести qlearn.stepper.EpisodeStepper — навчання порціями за часом для «Швидкого навчання» (етап 6).

Навіщо: епізод DQN може тривати секунди (≈1.2 мс на крок), і цілий епізод «заморозив» би вікно.
Stepper робить стільки кроків, скільки встигає за бюджет часу, і продовжує з того самого місця наступного разу.
Для таблиці він виконує цілі епізоди через run_episode — точно як раніше (NFR-1).
"""
import random
import time

import pytest

from qlearn.environment import GridEnv
from qlearn.stepper import EpisodeStepper
from qlearn.tabular import TabularAgent


def test_tabular_same_as_run_episode(maps):
    """Для таблиці: ті самі виклики run_episode у тому самому порядку → та сама Q, ті самі винагороди."""
    a_env, b_env = GridEnv(maps["K3_rough"]), GridEnv(maps["K3_rough"])
    random.seed(0)
    direct = TabularAgent(a_env)
    expected = [direct.run_episode()["reward"] for _ in range(50)]
    random.seed(0)
    agent = TabularAgent(b_env)
    stepper = EpisodeStepper(agent)
    got = []
    while len(got) < 50:
        got += stepper.train_for(1.0, max_episodes=50 - len(got))
    assert got == expected and agent.q == direct.q


def test_max_episodes_respected(maps):
    random.seed(1)
    stepper = EpisodeStepper(TabularAgent(GridEnv(maps["K1_default"])))
    assert len(stepper.train_for(5.0, max_episodes=7)) == 7


def test_zero_max_episodes(maps):
    stepper = EpisodeStepper(TabularAgent(GridEnv(maps["K1_default"])))
    assert stepper.train_for(1.0, max_episodes=0) == []


def test_reset_returns_to_start(maps):
    env = GridEnv(maps["K1_default"])
    stepper = EpisodeStepper(TabularAgent(env))
    stepper.state, stepper.steps, stepper.ep_reward = 5, 3, -3.0
    stepper.reset()
    assert (stepper.state, stepper.steps, stepper.ep_reward) == (env.start_i, 0, 0.0)


class TestDQNStepper:
    @pytest.fixture(autouse=True)
    def _torch(self):
        pytest.importorskip("torch")

    def make(self, env_dict):
        from qlearn.dqn import DQNAgent
        return DQNAgent(GridEnv(env_dict))

    def test_stepwise_same_as_run_episode(self, maps):
        """Покрокове навчання малими порціями = цілі епізоди run_episode (та сама мережа)."""
        random.seed(2)
        direct = self.make(maps["K1_default"])
        expected = [direct.run_episode()["reward"] for _ in range(6)]
        random.seed(2)
        agent = self.make(maps["K1_default"])
        stepper = EpisodeStepper(agent)
        got = []
        while len(got) < 6:
            got += stepper.train_for(0.003, max_episodes=6 - len(got))   # дрібні порції — епізоди рвуться посередині
        assert got == pytest.approx(expected)
        assert agent.q_snapshot() == direct.q_snapshot()

    def test_budget_respected(self, maps):
        random.seed(3)
        stepper = EpisodeStepper(self.make(maps["R1_10x8"]))
        t0 = time.perf_counter()
        stepper.train_for(0.05, max_episodes=1000)
        assert time.perf_counter() - t0 < 0.05 + 0.25          # перевищення — не більше одного кроку
        assert stepper.steps > 0 or stepper.episodes_done > 0

    def test_episode_ends_at_step_cap(self, maps):
        random.seed(4)
        agent = self.make(maps["K1_default"])
        agent.epsilon = 1.0
        agent.env.step_cap = lambda: 3
        stepper = EpisodeStepper(agent)
        rewards = stepper.train_for(5.0, max_episodes=2)
        assert len(rewards) == 2 and rewards[0] == pytest.approx(-3.0)


def test_last_results_have_steps_and_goal(maps):
    """Етап 7: для запису метрик потрібні не лише винагороди, а й кроки та фініш кожного епізоду."""
    random.seed(5)
    agent = TabularAgent(GridEnv(maps["K1_default"]))
    stepper = EpisodeStepper(agent)
    rewards = stepper.train_for(5.0, max_episodes=4)
    results = stepper.last_results
    assert [r["reward"] for r in results] == rewards
    assert all(set(r) == {"reward", "steps", "goal"} and r["steps"] > 0 for r in results)


def test_last_results_dqn_stepwise(maps):
    pytest.importorskip("torch")
    from qlearn.dqn import DQNAgent
    random.seed(6)
    agent = DQNAgent(GridEnv(maps["S_3x3"]))
    stepper = EpisodeStepper(agent)
    got = []
    while len(got) < 3:
        stepper.train_for(0.002, max_episodes=3 - len(got))
        got += stepper.last_results
    random.seed(6)
    direct = DQNAgent(GridEnv(maps["S_3x3"]))
    expected = [direct.run_episode() for _ in range(3)]
    assert [(r["steps"], r["goal"]) for r in got] == [(e["steps"], e["goal"]) for e in expected]
    assert [r["reward"] for r in got] == pytest.approx([e["reward"] for e in expected])
