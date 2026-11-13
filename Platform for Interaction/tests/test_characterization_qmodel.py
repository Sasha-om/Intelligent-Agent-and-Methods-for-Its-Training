"""Характеризаційні юніт-тести QModel (етап 1).

Вони фіксують ПОТОЧНУ поведінку табличного Q-навчання, щоб будь-яка майбутня зміна
(рефакторинг на етапі 2, додавання DQN) не змінила її непомітно.
Ручні формули: Q(s,a) ← Q + α·[r + γ·max Q(s′,·) − Q], α=0.35, γ=0.9, r = −вартість (+50 на фініші).
"""
import math
import random

import pytest

import helpers

ALPHA, GAMMA, GOAL = 0.35, 0.9, 50


def key_of(model, frm_cell, to_cell):
    """Ключ Q для переходу між двома клітинками."""
    return model.edge_key(model.idx(frm_cell), model.idx(to_cell))


# ------------------------------------------------------------------ середовище
def test_default_env(make_model):
    m = make_model()
    assert (m.cols, m.rows) == (7, 6)
    assert m.start == (0, 0) and m.finish == (6, 5)
    assert len(m.free) == 42
    assert len(m.q) == 42 * 4 and len(m.visited) == 42 * 4
    assert all(v == 0.0 for v in m.q)
    assert (m.alpha, m.gamma, m.epsilon) == (0.35, 0.9, 0.12)
    assert m.step_cap() == max(100, 7 * 6 * 3)


@pytest.mark.parametrize("name", ["K1_default", "K2_wall_gap", "K3_rough", "S_3x3", "R1_10x8", "R2_15x15"])
def test_graph_invariants(make_model, maps, name):
    """У кожній дії: ключ = i*4 + слот, сусід поруч, не перешкода, вартість 1 (або 3 для ділянки 3×)."""
    m = make_model(maps[name])
    assert m.finish_reachable()
    for i in m.free:
        c, r = m.cell_of(i)
        for to, key, cost, label in m.nb[i]:
            slot = key - i * 4
            assert 0 <= slot < 4
            dc, dr, dir_label = m_dirs(slot)
            assert label == dir_label
            tc, tr = m.cell_of(to)
            assert (tc - c, tr - r) == (dc, dr)
            assert m.in_bounds(tc, tr) and (tc, tr) not in m.obstacles
            assert cost == (3 if (tc, tr) in m.rough else 1)


def m_dirs(slot):
    return helpers.load_trainer().DIRS[slot]


def test_unreachable_finish_detected(make_model):
    m = make_model(helpers.UNREACHABLE)
    assert m.finish_reachable() is False


def test_optimal_cost_reference(make_model, maps):
    """Еталон Дейкстри, з яким порівнюють жадібний шлях."""
    assert helpers.optimal_cost(make_model(maps["K1_default"])) == 11
    assert helpers.optimal_cost(make_model(maps["S_3x3"])) == 4


# ------------------------------------------------------- формула оновлення
def test_q_update_by_hand(make_model, maps):
    """Сітка 3×3, ε=0, Q=0: перша дія — ВНИЗ (перша дозволена за DIRS). new_q = 0.35·(−1 + 0.9·0 − 0) = −0.35."""
    m = make_model(maps["S_3x3"], epsilon=0.0)
    res = m.apply_step(m.start_i)
    assert res["dir"] == "ВНИЗ" and res["to"] == m.idx((0, 1))
    assert res["reward"] == -1 and res["goal"] is False
    assert res["old_q"] == 0.0
    assert res["new_q"] == pytest.approx(ALPHA * (-1 + GAMMA * 0 - 0))
    assert res["new_q"] == pytest.approx(-0.35)
    assert m.q[res["key"]] == res["new_q"] and m.visited[res["key"]] == 1


def test_q_update_uses_max_next(make_model, maps):
    """Якщо в наступному стані є Q = 10, то new_q = 0.35·(−1 + 0.9·10) = 2.8."""
    m = make_model(maps["S_3x3"], epsilon=0.0)
    m.q[key_of(m, (0, 1), (0, 2))] = 10.0
    res = m.apply_step(m.start_i)
    assert res["to"] == m.idx((0, 1))
    assert res["new_q"] == pytest.approx(ALPHA * (-1 + GAMMA * 10.0))
    assert res["new_q"] == pytest.approx(2.8)


def test_terminal_no_bootstrap(make_model):
    """Крок у фініш: new_q = 0.35·(50 − 1) = 17.15; Q-значення фінішу не враховуються."""
    env = {"cols": 3, "rows": 3, "obstacles": [], "rough": [], "start": (2, 1), "finish": (2, 2)}
    m = make_model(env, epsilon=0.0)
    m.q[key_of(m, (2, 1), (2, 0))] = -1.0          # ВГОРУ гірше
    m.q[key_of(m, (2, 1), (1, 1))] = -1.0          # ВЛІВО гірше
    for _to, key, _c, _l in m.nb[m.finish_i]:      # «приманка»: великі Q у фініші не мають впливати
        m.q[key] = 100.0
    res = m.apply_step(m.start_i)
    assert res["goal"] is True and res["dir"] == "ВНИЗ"
    assert res["reward"] == GOAL - 1
    assert res["new_q"] == pytest.approx(ALPHA * (GOAL - 1))
    assert res["new_q"] == pytest.approx(17.15)


def test_rough_cell_costs_three(make_model):
    """Перехід на ділянку 3×: r = −3, new_q = 0.35·(−3) = −1.05."""
    env = {"cols": 3, "rows": 3, "obstacles": [], "rough": [(0, 1)], "start": (0, 0), "finish": (2, 2)}
    m = make_model(env, epsilon=0.0)
    res = m.apply_step(m.start_i)
    assert res["to"] == m.idx((0, 1)) and res["cost"] == 3 and res["reward"] == -3
    assert res["new_q"] == pytest.approx(-1.05)


# ------------------------------------------------------------- ε-жадібність
def test_epsilon_zero_is_greedy(make_model, maps):
    m = helpers.train(make_model(maps["K1_default"]), 200, seed=3)
    m.epsilon = 0.0
    random.seed(7)
    for i in m.free:
        if m.nb[i]:
            assert m.choose_action(i) == m.best_of(m.nb[i])


def test_epsilon_one_only_allowed_actions(make_model, maps):
    """ε=1: дія завжди випадкова, але тільки з дозволених; з кута (0,0) обидві дії трапляються."""
    m = make_model(maps["K1_default"], epsilon=1.0)
    random.seed(11)
    allowed = set(m.nb[m.start_i])
    seen = set()
    for _ in range(2000):
        a = m.choose_action(m.start_i)
        assert a in allowed
        seen.add(a[3])
    assert seen == {"ВНИЗ", "ВПРАВО"}


# ------------------------------------------------- два шляхи оновлення Q
@pytest.mark.parametrize("name", ["K1_default", "K3_rough", "R1_10x8"])
def test_two_update_paths_equivalent(make_model, maps, name):
    """apply_step (анімація) і run_episode_silent (швидкий цикл) з тим самим seed дають однакову Q (знахідка З1)."""
    episodes = 150
    fast = helpers.train(make_model(maps[name]), episodes, seed=0)

    slow = make_model(maps[name])
    random.seed(0)
    for _ in range(episodes):
        s, steps = slow.start_i, 0
        while True:
            res = slow.apply_step(s)
            if res is None:
                break
            steps += 1
            s = res["to"]
            if res["goal"] or steps >= slow.step_cap():
                break
    assert slow.q == fast.q
    assert bytes(slow.visited) == bytes(fast.visited)


def test_run_episode_returns_reward_sum(make_model, maps):
    """Сума винагород епізоду: на навченій моделі з ε=0 = 50 − вартість оптимального шляху."""
    m = helpers.train(make_model(maps["K1_default"]), 600, seed=0)
    m.epsilon = 0.0
    assert m.run_episode_silent() == pytest.approx(GOAL - 11)


def test_reset_q_clears_learning(make_model, maps):
    m = helpers.train(make_model(maps["K1_default"]), 50, seed=0)
    assert any(v != 0.0 for v in m.q)
    m.reset_q()
    assert all(v == 0.0 for v in m.q) and not any(m.visited)


def test_q_values_finite_after_training(make_model, maps):
    m = helpers.train(make_model(maps["R2_15x15"]), 300, seed=5)
    assert all(math.isfinite(v) for v in m.q)
