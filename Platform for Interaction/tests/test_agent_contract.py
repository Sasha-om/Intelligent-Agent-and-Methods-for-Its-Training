"""Контракт агента: вимоги, однакові для БУДЬ-ЯКОГО агента (таблиця — етап 2, DQN — етап 3).

Щоб перевірити нового агента, достатньо створити клас Test<Агент>Contract(AgentContract)
з фікстурою make_agent(env_dict, epsilon) → агент.
"""
import math
import random

import pytest

from qlearn.environment import GridEnv
from qlearn.tabular import TabularAgent

STEP_KEYS = {"to", "key", "old_q", "new_q", "reward", "cost", "dir", "goal"}


class AgentContract:
    """Набір тестів-вимог. Нащадок має надати фікстуру make_agent."""

    def test_has_name_and_params(self, make_agent, maps):
        agent = make_agent(maps["K1_default"])
        assert isinstance(agent.name, str) and agent.name
        params = agent.params()
        assert {"gamma", "epsilon"} <= set(params)

    def test_choose_action_only_allowed(self, make_agent, maps):
        agent = make_agent(maps["K2_wall_gap"], epsilon=1.0)
        env = agent.env
        random.seed(1)
        for i in env.free:
            allowed = set(env.nb[i])
            for _ in range(20):
                assert agent.choose_action(i) in allowed

    def test_learn_step_fields_and_update(self, make_agent, maps):
        agent = make_agent(maps["K1_default"])
        random.seed(2)
        res = agent.learn_step(agent.env.start_i)
        assert STEP_KEYS <= set(res)
        assert res["to"] in {o[0] for o in agent.env.nb[agent.env.start_i]}
        assert math.isfinite(res["new_q"])

    def test_run_episode_terminates(self, make_agent, maps):
        agent = make_agent(maps["K1_default"])
        random.seed(3)
        out = agent.run_episode()
        assert {"reward", "steps", "goal"} <= set(out)
        assert 0 < out["steps"] <= agent.env.step_cap()
        assert out["goal"] in (True, False)

    def test_q_snapshot_shape_and_finite(self, make_agent, maps):
        agent = make_agent(maps["R1_10x8"])
        random.seed(4)
        for _ in range(20):
            agent.run_episode()
        q = agent.q_snapshot()
        assert len(q) == agent.env.n_keys
        assert all(math.isfinite(v) for v in q)

    def test_greedy_path_valid(self, make_agent, maps):
        agent = make_agent(maps["K1_default"])
        random.seed(5)
        for _ in range(30):
            agent.run_episode()
        path = agent.greedy_path()
        assert path[0] == agent.env.start_i
        assert len(path) == len(set(path))                          # без циклів
        for a, b in zip(path, path[1:]):
            assert b in {o[0] for o in agent.env.nb[a]}

    def test_reset_clears_learning(self, make_agent, maps):
        agent = make_agent(maps["K1_default"])
        random.seed(6)
        for _ in range(10):
            agent.run_episode()
        agent.reset()
        assert all(v == 0.0 for v in agent.q_snapshot())


class TestTabularContract(AgentContract):
    @pytest.fixture
    def make_agent(self):
        def _make(env_dict, epsilon=None):
            agent = TabularAgent(GridEnv(env_dict))
            if epsilon is not None:
                agent.epsilon = epsilon
            return agent
        return _make
