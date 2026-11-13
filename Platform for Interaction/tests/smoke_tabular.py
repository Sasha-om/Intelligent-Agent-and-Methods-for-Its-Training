"""Smoke-тести табличного режиму: «програма живе» на різних картах і швидко дає результат."""
import math
import time

import pytest

import helpers


@pytest.mark.parametrize("name", ["K1_default", "K2_wall_gap", "K3_rough", "R1_10x8"])
def test_trains_and_finds_path(make_model, maps, name):
    """600 епізодів: Q скінченні, жадібний шлях доходить до фінішу з допустимих переходів."""
    m = helpers.train(make_model(maps[name]), 600, seed=0)
    assert all(math.isfinite(v) for v in m.q)
    path = m.greedy_path()
    assert path[0] == m.start_i and path[-1] == m.finish_i
    assert helpers.path_cost(m, path) is not None


def test_default_map_path_is_optimal(make_model, maps):
    m = helpers.train(make_model(maps["K1_default"]), 600, seed=0)
    assert helpers.path_cost(m, m.greedy_path()) == helpers.optimal_cost(m) == 11


def test_big_grid_runs_quickly(make_model):
    """Сітка 100×100 (максимум редактора): 5 епізодів без помилок і швидше за 30 с."""
    env = {"cols": 100, "rows": 100, "obstacles": [], "rough": [], "start": (0, 0), "finish": (99, 99)}
    m = make_model(env)
    t0 = time.perf_counter()
    helpers.train(m, 5, seed=0)
    assert time.perf_counter() - t0 < 30
    assert all(math.isfinite(v) for v in m.q)


def test_apply_step_result_fields(make_model):
    res = make_model().apply_step(0)
    assert {"to", "key", "old_q", "new_q", "reward", "cost", "dir", "goal"} <= set(res)
