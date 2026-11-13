"""Юніт-тести ознак стану та масок дій qlearn.features (етап 3, без torch)."""
import pytest

from qlearn import features as F
from qlearn.environment import GridEnv


def test_onehot(maps):
    env = GridEnv(maps["K1_default"])
    for i in env.free:
        v = F.onehot(env, i)
        assert len(v) == env.cols * env.rows == F.feature_size(env, "onehot")
        assert v[i] == 1.0 and sum(v) == 1.0


def test_compact_values_and_range(maps):
    env = GridEnv(maps["K3_rough"])
    assert F.feature_size(env, "compact") == 8
    for i in env.free:
        v = F.compact(env, i)
        assert len(v) == 8
        assert all(-1.0 <= x <= 1.0 for x in v)


def test_compact_by_hand(maps):
    """Клітинка (0,2) карти K3: x=0, y=2/5, до фінішу (6,2): dx=6/6=1, dy=0;
    сусіди ВГОРУ (0,1) звичайна → 1, ВНИЗ (0,3) → 1, ВЛІВО — край → 0, ВПРАВО (1,2) ділянка 3× → 1/3."""
    env = GridEnv(maps["K3_rough"])
    v = F.compact(env, env.idx((0, 2)))
    assert v == pytest.approx([0.0, 2 / 5, 1.0, 0.0, 1.0, 1.0, 0.0, 1 / 3])


def test_compact_blocked_neighbour_is_zero(maps):
    env = GridEnv(maps["K2_wall_gap"])
    v = F.compact(env, env.idx((2, 0)))       # праворуч (3,0) — стіна
    assert v[4 + 3] == 0.0


def test_action_mask_matches_graph(maps):
    for name in ("K1_default", "K2_wall_gap", "R1_10x8"):
        env = GridEnv(maps[name])
        for i in env.free:
            mask = F.action_mask(env, i)
            slots = {key - i * 4 for _to, key, _c, _l in env.nb[i]}
            assert mask == [s in slots for s in range(4)]


def test_unknown_kind_raises(maps):
    with pytest.raises(ValueError):
        F.feature_size(GridEnv(maps["K1_default"]), "pixels")


def test_onehot_limited_to_small_grids():
    """onehot для 100×100 — матриця 10 000 × 10 000 (≈400 МБ): дозволено лише до 900 клітинок (30×30)."""
    ok = GridEnv({"cols": 30, "rows": 30, "obstacles": [], "rough": [], "start": (0, 0), "finish": (29, 29)})
    assert F.feature_size(ok, "onehot") == 900
    big = GridEnv({"cols": 31, "rows": 30, "obstacles": [], "rough": [], "start": (0, 0), "finish": (30, 29)})
    with pytest.raises(ValueError, match="compact"):
        F.feature_size(big, "onehot")
    assert F.feature_size(big, "compact") == 8
