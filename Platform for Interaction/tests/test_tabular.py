"""Юніт-тести qlearn.tabular.TabularAgent (етап 2): ті самі ручні формули, що й для старого QModel,
плюс рівність «агент напряму» ↔ «фасад QModel у тренажері»."""
import random

import pytest

import helpers
from qlearn.environment import GridEnv
from qlearn.tabular import TabularAgent


def make(env_dict, epsilon=None):
    agent = TabularAgent(GridEnv(env_dict))
    if epsilon is not None:
        agent.epsilon = epsilon
    return agent


def test_defaults():
    agent = make(None)
    assert (agent.alpha, agent.gamma, agent.epsilon) == (0.35, 0.9, 0.12)
    assert agent.name == "tabular"
    assert agent.params() == {"alpha": 0.35, "gamma": 0.9, "epsilon": 0.12}


def test_reset_zeroes_table(maps):
    agent = make(maps["K1_default"])
    random.seed(1)
    for _ in range(10):
        agent.run_episode()
    agent.reset()
    assert all(v == 0.0 for v in agent.q_snapshot())


def test_update_by_hand(maps):
    agent = make(maps["S_3x3"], epsilon=0.0)
    res = agent.learn_step(agent.env.start_i)
    assert res["dir"] == "ВНИЗ" and res["new_q"] == pytest.approx(-0.35)


def test_terminal_by_hand():
    env = {"cols": 3, "rows": 3, "obstacles": [], "rough": [], "start": (2, 1), "finish": (2, 2)}
    agent = make(env, epsilon=0.0)
    e = agent.env
    agent.q[e.edge_key(e.idx((2, 1)), e.idx((2, 0)))] = -1.0
    agent.q[e.edge_key(e.idx((2, 1)), e.idx((1, 1)))] = -1.0
    res = agent.learn_step(e.start_i)
    assert res["goal"] and res["new_q"] == pytest.approx(17.15)


def test_learn_step_and_run_episode_equivalent(maps):
    fast, slow = make(maps["K3_rough"]), make(maps["K3_rough"])
    random.seed(0)
    for _ in range(100):
        fast.run_episode()
    random.seed(0)
    for _ in range(100):
        s, steps = slow.env.start_i, 0
        while True:
            res = slow.learn_step(s)
            if res is None:
                break
            steps += 1
            s = res["to"]
            if res["goal"] or steps >= slow.env.step_cap():
                break
    assert fast.q == slow.q


def test_run_episode_reports_goal_and_steps(maps):
    agent = make(maps["K1_default"])
    random.seed(0)
    for _ in range(600):
        agent.run_episode()
    agent.epsilon = 0.0
    out = agent.run_episode()
    assert out == {"reward": pytest.approx(39.0), "steps": 11, "goal": True}


def test_facade_equals_direct_agent(trainer, maps):
    """Фасад QModel і TabularAgent напряму після однакового навчання мають однакову Q."""
    facade = trainer.QModel()
    facade.load_env(maps["R1_10x8"])
    helpers.train(facade, 300, seed=9)
    agent = make(maps["R1_10x8"])
    random.seed(9)
    for _ in range(300):
        agent.run_episode()
    assert facade.q == agent.q_snapshot()


def test_facade_proxies_editor_fields(trainer):
    """Редактор App змінює m.cols, m.obstacles тощо напряму, потім викликає m.rebuild()."""
    m = trainer.QModel()
    m.cols, m.rows = 9, 4
    m.obstacles = {(1, 0)}
    m.rough.add((2, 2))
    m.finish = (8, 3)
    m.rebuild()
    assert (m.env.cols, m.env.rows, m.env.finish) == (9, 4, (8, 3))
    assert (1, 0) in m.env.obstacles and (2, 2) in m.env.rough
    assert len(m.q) == 9 * 4 * 4 and all(v == 0.0 for v in m.q)
    m.alpha = 0.5
    assert m.agent.alpha == 0.5
