"""Регресія збіжності DQN (етап 3, маркер slow — запуск з --runslow).

Критерій: жадібний шлях оптимальний за ДИСКОНТОВАНОЮ винагородою (= V*(старт), ітерація цінностей)
і лишається таким ≥ 100 епізодів. Не «найдешевший шлях»: з γ=0.9 на K3 оптимально йти прямо
через дорогі клітинки (INS-008).

Межі встановлено за вимірами 2026-11-13 (5 seed; найгірший епізод збіжності: K1 40, K2 60, K3 320, R1 60)
з запасом приблизно ×2. Вимога — у ≥ 4 з 5 seed.
"""
import random

import pytest

pytest.importorskip("torch")

import helpers                               # noqa: E402
from qlearn.dqn import DQNAgent              # noqa: E402
from qlearn.environment import GridEnv       # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.slow]

GAMMA = 0.9
SEEDS = range(5)
CASES = [  # (карта, ознаки, межа епізодів)
    ("K1_default", "compact", 150),
    ("K2_wall_gap", "compact", 150),
    ("K3_rough", "compact", 600),
    ("R1_10x8", "compact", 150),
    ("K1_default", "onehot", 150),
]


@pytest.mark.parametrize("map_name, kind, n_max", CASES)
def test_dqn_converges(maps, map_name, kind, n_max):
    results = []
    for seed in SEEDS:
        env = GridEnv(maps[map_name])
        random.seed(seed)
        agent = DQNAgent(env, features=kind)
        results.append(helpers.convergence_episode(env, agent, GAMMA, max_episodes=n_max + 100))
    ok = [c for c in results if c is not None and c <= n_max]
    print(f"{map_name}/{kind}: епізоди збіжності по seed = {results}")
    assert len(ok) >= 4, f"{map_name}/{kind}: зійшлося {len(ok)} з 5 у межах {n_max} епізодів: {results}"


def test_tabular_k3_does_not_find_discounted_optimum(maps):
    """Характеризація (для роботи): табличний агент з ε=0.12 на K3 застрягає в обході —
    ланцюжок дорогих клітинок майже не досліджується (INS-008). Якщо колись зміниться — це помітно."""
    from qlearn.tabular import TabularAgent
    env = GridEnv(maps["K3_rough"])
    random.seed(0)
    agent = TabularAgent(env)
    assert helpers.convergence_episode(env, agent, GAMMA, max_episodes=3000) is None
    assert len(agent.greedy_path()) - 1 == 10          # обхід: найдешевший, але не оптимальний за γ=0.9
