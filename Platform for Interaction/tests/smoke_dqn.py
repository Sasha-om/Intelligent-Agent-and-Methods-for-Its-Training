"""Smoke-тест DQN: агент живе, навчається кілька епізодів, нічого не падає, працює через фабрику."""
import math
import random
import time

import pytest

pytest.importorskip("torch")

from qlearn.environment import GridEnv     # noqa: E402
from qlearn.factory import make_agent     # noqa: E402

pytestmark = pytest.mark.torch


def test_dqn_runs_quickly():
    env = GridEnv({"cols": 5, "rows": 5, "obstacles": [(2, 1), (2, 2)], "rough": [(1, 3)],
                   "start": (0, 0), "finish": (4, 4)})
    random.seed(0)
    agent = make_agent("dqn", env)
    t0 = time.perf_counter()
    for _ in range(20):
        out = agent.run_episode()
        assert out["steps"] > 0
    assert time.perf_counter() - t0 < 10
    assert all(math.isfinite(v) for v in agent.q_snapshot())
    path = agent.greedy_path()
    assert path[0] == env.start_i
