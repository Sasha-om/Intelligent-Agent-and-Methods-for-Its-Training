"""Юніт-тести qlearn.recording.RunRecorder — метрики навчального запуску (етап 4).

Метрики (FR-R1, FR-R2; визначення — docs/PLAN-2-stages-4-6.md, етап 4):
    first_success_episode — перший епізод, у якому агент дійшов до фінішу;
    convergence_episode   — перша оцінка, з якої жадібний шлях оптимальний за V* і лишається таким ≥ hold епізодів;
    return_ratio / return_gap — винагорода фінального жадібного шляху відносно V* (головна метрика, INS-008);
    final_path_cost / cost_ratio — вартість шляху відносно найдешевшого (довідково);
    mean_reward_last_50, success_rate_last_50, total_env_steps, duration_sec.
"""
import random

import pytest

import helpers
from qlearn.environment import GridEnv
from qlearn.recording import RunRecorder, convergence_from_evals
from qlearn.tabular import TabularAgent

K3_STRAIGHT = -3 * (1 + 0.9 + 0.81 + 0.729 + 0.6561) + 0.9 ** 5 * 49      # 16.649
K3_DETOUR = -(1 - 0.9 ** 9) / 0.1 + 0.9 ** 9 * 49                          # 12.858


class FakeAgent:
    """Агент-«сценарій»: повертає заздалегідь задані результати епізодів і жадібні шляхи."""
    name = "fake"

    def __init__(self, env, episodes, paths, gamma=0.9):
        self.env, self.gamma = env, gamma
        self._episodes, self._paths = list(episodes), list(paths)

    def params(self):
        return {"gamma": self.gamma}

    def run_episode(self):
        return self._episodes.pop(0)

    def greedy_path(self):
        return self._paths.pop(0) if len(self._paths) > 1 else self._paths[0]


def straight_path(env):
    return [env.idx((c, 2)) for c in range(7)]


def detour_path(env):
    cells = [(0, 2), (0, 1), (0, 0)] + [(c, 0) for c in range(1, 7)] + [(6, 1), (6, 2)]
    return [env.idx(p) for p in cells]


# ---------------------------------------------------------------- синтетичний запуск
@pytest.fixture
def synthetic(maps):
    """60 епізодів: винагорода k, 3 кроки, перші 4 — без фінішу; оцінки на 10,20,…,60:
    погано, оптимально, погано, оптимально ×3 → збіжність з 40-го (60 − 40 = 20 = hold)."""
    env = GridEnv(maps["K1_default"])
    episodes = [{"reward": float(k), "steps": 3, "goal": k > 4} for k in range(1, 61)]
    bad, good = [env.start_i], env.optimal_path(0.9)
    agent = FakeAgent(env, episodes, [bad, good, bad, good, good, good, good])
    rec = RunRecorder(env, agent, seed=7, eval_every=10, hold=20)
    for _ in range(60):
        rec.on_episode(agent.run_episode())
    return rec


def test_collects_per_episode(synthetic):
    assert synthetic.rewards == [float(k) for k in range(1, 61)]
    assert synthetic.steps == [3] * 60
    assert synthetic.goals == [False] * 4 + [True] * 56


def test_greedy_eval_every_k(synthetic):
    assert [e["episode"] for e in synthetic.evals] == [10, 20, 30, 40, 50, 60]
    assert [e["optimal"] for e in synthetic.evals] == [False, True, False, True, True, True]


def test_summary_on_synthetic_run(synthetic):
    s = synthetic.summary()
    assert s["episodes"] == 60
    assert s["first_success_episode"] == 5
    assert s["convergence_episode"] == 40
    assert s["mean_reward_last_50"] == pytest.approx(35.5)          # середнє 11…60
    assert s["success_rate_last_50"] == pytest.approx(1.0)
    assert s["total_env_steps"] == 180
    assert s["final_reaches_finish"] is True and s["final_optimal"] is True
    assert s["return_ratio"] == pytest.approx(1.0) and s["return_gap"] == pytest.approx(0.0)
    assert s["final_path_cost"] == 11 and s["cost_ratio"] == pytest.approx(1.0)
    assert s["duration_sec"] >= 0


def test_not_converged_is_none():
    assert convergence_from_evals([{"episode": 10, "optimal": True}, {"episode": 20, "optimal": False}], hold=100) is None
    assert convergence_from_evals([], hold=100) is None


def test_convergence_first_long_streak():
    evals = [{"episode": e, "optimal": ok} for e, ok in
             [(10, True), (20, False), (30, True), (40, True), (50, True), (60, False), (70, True)]]
    assert convergence_from_evals(evals, hold=20) == 30     # 30…50 — перша серія довжиною ≥ 20


def test_no_success_at_all(maps):
    env = GridEnv(maps["K1_default"])
    agent = FakeAgent(env, [{"reward": -5.0, "steps": 5, "goal": False}] * 12, [[env.start_i]])
    rec = RunRecorder(env, agent, eval_every=10, hold=100)
    for _ in range(12):
        rec.on_episode(agent.run_episode())
    s = rec.summary()
    assert s["first_success_episode"] is None and s["convergence_episode"] is None
    assert s["final_reaches_finish"] is False
    assert s["return_ratio"] is None and s["final_path_cost"] is None and s["cost_ratio"] is None
    assert s["success_rate_last_50"] == 0.0


# --------------------------------------------- головна й довідкова метрики (INS-008)
def test_k3_detour_is_cheapest_but_not_optimal(maps):
    env = GridEnv(maps["K3_rough"])
    rec = RunRecorder(env, FakeAgent(env, [], [detour_path(env)]))
    s = rec.summary()
    assert s["final_path_cost"] == 10 and s["cost_ratio"] == pytest.approx(1.0)
    assert s["return_ratio"] == pytest.approx(K3_DETOUR / K3_STRAIGHT)
    assert s["return_gap"] == pytest.approx(K3_STRAIGHT - K3_DETOUR)
    assert s["final_optimal"] is False


def test_k3_straight_is_optimal_but_not_cheapest(maps):
    env = GridEnv(maps["K3_rough"])
    s = RunRecorder(env, FakeAgent(env, [], [straight_path(env)])).summary()
    assert s["final_optimal"] is True and s["return_ratio"] == pytest.approx(1.0)
    assert s["final_path_cost"] == 16 and s["cost_ratio"] == pytest.approx(1.6)


def test_return_ratio_none_when_v_star_not_positive():
    """Довгий шлях: V* < 0 — відношення не має сенсу, лишається return_gap."""
    env = GridEnv({"cols": 40, "rows": 3, "obstacles": [], "rough": [], "start": (0, 0), "finish": (39, 2)})
    assert env.optimal_return(0.9) < 0
    s = RunRecorder(env, FakeAgent(env, [], [env.optimal_path(0.9)])).summary()
    assert s["return_ratio"] is None and s["return_gap"] == pytest.approx(0.0) and s["final_optimal"] is True


# ----------------------------------------------------------- справжні агенти
def test_real_tabular_and_helpers_agree(maps):
    """Епізод збіжності з RunRecorder = той, що рахує tests/helpers.convergence_episode (той самий критерій)."""
    env = GridEnv(maps["K1_default"])
    random.seed(0)
    rec = RunRecorder(env, TabularAgent(env), seed=0)
    result = rec.run(400)
    env2 = GridEnv(maps["K1_default"])
    random.seed(0)
    expected = helpers.convergence_episode(env2, TabularAgent(env2), 0.9, max_episodes=400)
    assert result["summary"]["convergence_episode"] == expected is not None


def test_recording_does_not_change_learning(trainer, maps):
    """Запис (оцінки жадібного шляху) не споживає випадкових чисел: Q = golden-знімок етапу 1."""
    import json
    golden = json.loads((helpers.TESTS_DIR / "golden_tabular.json").read_text(encoding="utf-8"))
    env = GridEnv(maps["K1_default"])
    agent = TabularAgent(env)
    random.seed(0)
    RunRecorder(env, agent, seed=0).run(600)
    assert helpers.q_hash(agent.q) == golden["scenarios"]["K1_default_seed0_600"]["snapshot"]["q_sha256"]


def test_same_structure_for_both_agents(maps):
    pytest.importorskip("torch")
    from qlearn.dqn import DQNAgent
    env = GridEnv(maps["S_3x3"])
    random.seed(1)
    tab = RunRecorder(env, TabularAgent(env), seed=1).run(20)
    random.seed(1)
    dqn = RunRecorder(env, DQNAgent(env), seed=1).run(20)
    assert set(tab) == set(dqn) and set(tab["summary"]) == set(dqn["summary"])
    assert tab["agent"] == "tabular" and dqn["agent"] == "dqn"
    assert dqn["versions"]["torch"] and tab["params"]["alpha"] == 0.35
