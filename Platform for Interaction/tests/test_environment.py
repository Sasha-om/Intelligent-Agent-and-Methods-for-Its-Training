"""Юніт-тести qlearn.environment.GridEnv (етап 2).

Еталон графа — tests/golden_env.json, знятий зі старого QModel до рефакторингу.
"""
import json
import os
import subprocess
import sys

import pytest

import helpers
import make_golden_env
from qlearn.environment import DEFAULT_ENV, DIRS, GOAL_BONUS, GridEnv

GOLDEN_ENV = json.loads((helpers.TESTS_DIR / "golden_env.json").read_text(encoding="utf-8"))

# Дочірній Python на Windows пише в кодуванні консолі (cp1251), а ми читаємо UTF-8 — задаємо явно (INS-010).
UTF8_ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def test_constants_unchanged():
    assert DIRS == [(0, -1, "ВГОРУ"), (0, 1, "ВНИЗ"), (-1, 0, "ВЛІВО"), (1, 0, "ВПРАВО")]
    assert GOAL_BONUS == 50
    assert DEFAULT_ENV == {"cols": 7, "rows": 6, "obstacles": [], "rough": [], "start": (0, 0), "finish": (6, 5)}


@pytest.mark.parametrize("name", sorted(GOLDEN_ENV))
def test_graph_matches_legacy(maps, name):
    """Граф сусідів GridEnv збігається з графом старого QModel на кожній тестовій карті."""
    env = GridEnv(maps[name])
    expected = GOLDEN_ENV[name]
    assert make_golden_env.graph_fingerprint(env) == expected["graph_sha256"]
    assert len(env.free) == expected["free"]
    assert env.step_cap() == expected["step_cap"]
    assert env.finish_reachable() == expected["reachable"]


def test_default_construction():
    env = GridEnv()
    assert (env.cols, env.rows, env.start, env.finish) == (7, 6, (0, 0), (6, 5))
    assert env.n_keys == 7 * 6 * 4


def test_reward():
    env = GridEnv()
    assert env.reward(1, False) == -1
    assert env.reward(3, False) == -3
    assert env.reward(1, True) == GOAL_BONUS - 1


def test_unreachable():
    assert GridEnv(helpers.UNREACHABLE).finish_reachable() is False


@pytest.mark.parametrize("name, cost", [("K1_default", 11), ("S_3x3", 4), ("K3_rough", 10), ("K2_wall_gap", 14)])
def test_optimal_cost(maps, name, cost):
    """Вартість найдешевшого шляху (Дейкстра) — еталон для оцінки жадібного шляху."""
    assert GridEnv(maps[name]).optimal_cost() == cost


def test_optimal_cost_unreachable_is_none():
    assert GridEnv(helpers.UNREACHABLE).optimal_cost() is None


def test_idx_cell_of_roundtrip():
    env = GridEnv()
    for i in range(env.cols * env.rows):
        assert env.idx(env.cell_of(i)) == i


def test_edge_key_and_cell_kind(maps):
    env = GridEnv(maps["K3_rough"])
    a, b = env.idx((0, 2)), env.idx((1, 2))
    assert env.edge_key(a, b) == a * 4 + 3        # ВПРАВО — слот 3
    assert env.edge_key(a, env.idx((5, 5))) is None
    assert env.cell_kind((0, 2)) == "старт" and env.cell_kind((6, 2)) == "фініш"
    assert env.cell_kind((1, 1)) == "ділянка 3×" and env.cell_kind((0, 0)) == "звичайна"


def test_to_dict_roundtrip(maps):
    for env_dict in maps.values():
        env = GridEnv(env_dict)
        again = GridEnv(env.to_dict())
        assert make_golden_env.graph_fingerprint(again) == make_golden_env.graph_fingerprint(env)


def test_fingerprint_stable_and_sensitive(maps):
    """Відбиток карти: однаковий для однакових карт (порядок перешкод не важить), інший — для іншої карти."""
    base = dict(maps["R1_10x8"])
    shuffled = dict(base, obstacles=list(reversed(base["obstacles"])))
    assert GridEnv(base).fingerprint() == GridEnv(shuffled).fingerprint()
    changed = dict(base, obstacles=base["obstacles"][1:])
    assert GridEnv(base).fingerprint() != GridEnv(changed).fingerprint()


def test_mutating_fields_then_rebuild():
    """Як робить редактор: змінює поля напряму, потім rebuild()."""
    env = GridEnv()
    env.obstacles.add((1, 0))
    env.rebuild()
    assert all(o[0] != env.idx((1, 0)) for o in env.nb[0])


def test_backend_does_not_import_tkinter():
    """Бекенд не залежить від GUI (NFR-4, NFR-8)."""
    code = ("import sys; import qlearn, qlearn.environment, qlearn.agent, qlearn.tabular; "
            "print('tkinter' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=helpers.APP_DIR,
                         capture_output=True, text=True, encoding="utf-8", check=True, env=UTF8_ENV)
    assert out.stdout.strip() == "False"


# ------------------------------------------- оптимум за дисконтованою винагородою (етап 3)
def test_optimal_return_by_hand_k1(maps):
    """K1: 11 кроків, 10 × (−1) і останній +49: V* = −(1 − 0.9¹⁰)/0.1 + 0.9¹⁰·49 ≈ 10.5720."""
    env = GridEnv(maps["K1_default"])
    expected = -(1 - 0.9 ** 10) / 0.1 + 0.9 ** 10 * 49
    assert env.optimal_return(0.9) == pytest.approx(expected, abs=1e-9)
    assert expected == pytest.approx(10.5720, abs=1e-4)


def test_optimal_policy_k3_goes_through_rough(maps):
    """K3 з γ=0.9: прямо через 5 дорогих клітинок (−3×5, +49) вигідніше за обхід (−1×9, +49):
    16.649 > 12.858. Тобто найдешевший шлях (вартість 10) ≠ оптимальний за винагородою (вартість 16)."""
    env = GridEnv(maps["K3_rough"])
    straight = -3 * (1 + 0.9 + 0.81 + 0.729 + 0.6561) + 0.9 ** 5 * 49
    detour = -(1 - 0.9 ** 9) / 0.1 + 0.9 ** 9 * 49
    assert env.optimal_return(0.9) == pytest.approx(straight, abs=1e-9)
    assert straight > detour
    path = env.optimal_path(0.9)
    assert len(path) - 1 == 6 and env.optimal_cost() == 10


def test_optimal_policy_depends_on_gamma(maps):
    """З γ=0.99 майбутнє важить більше — і на K3 вигідніший уже обхід (вартість 10)."""
    env = GridEnv(maps["K3_rough"])
    assert len(env.optimal_path(0.99)) - 1 == 10


def test_path_return(maps):
    env = GridEnv(maps["K1_default"])
    path = env.optimal_path(0.9)
    assert env.path_return(path, 0.9) == pytest.approx(env.optimal_return(0.9), abs=1e-9)
    assert env.path_return([env.start_i], 0.9) == 0.0


def test_optimal_values_unreachable_is_none():
    assert GridEnv(helpers.UNREACHABLE).optimal_return(0.9) is None


def test_path_cost(maps):
    """Вартість шляху (етап 4): сума вартостей переходів; None для неіснуючого переходу."""
    env = GridEnv(maps["K3_rough"])
    straight = [env.idx((c, 2)) for c in range(7)]
    assert env.path_cost(straight) == 16
    assert env.path_cost([env.start_i]) == 0
    assert env.path_cost([env.idx((0, 2)), env.idx((2, 2))]) is None
