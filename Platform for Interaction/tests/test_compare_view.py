"""Юніт-тести qlearn.compare_view — дані для вікна «Порівняння результатів» (етап 8, FR-C4).

Тут немає tkinter: таблиця, криві, кольори й попередження готуються як звичайні списки й рядки,
а вікно (`CompareWindow` у тренажері) лише їх малює.
"""
import copy
import statistics

import pytest

from qlearn import compare_view as cv
from qlearn.comparison import moving_average
from qlearn.recording import load_run

import helpers

FIXTURES = helpers.TRAINER_PATH.parent / "tests"


def fake_run(run_id, agent="tabular", seed=1, rewards=(1.0, 2.0, 3.0), fingerprint="fp-A", cols=7, rows=6,
             gamma=0.9, params=None, summary=None, evals=None, optimal_return=10.0, note="", changed=()):
    """Мінімальний запуск схеми v1 (лише поля, потрібні вікну порівняння)."""
    n = len(rewards)
    base_summary = {"episodes": n, "first_success_episode": 2, "convergence_episode": None, "final_optimal": False,
                    "return_ratio": 0.75, "return_gap": 2.5, "final_path_cost": 14, "cost_ratio": 1.0769,
                    "mean_reward_last_50": 12.345, "success_rate_last_50": 0.8, "total_env_steps": 1234,
                    "duration_sec": 3.21}
    return {"schema_version": 1, "run_id": run_id, "agent": agent, "seed": seed, "gamma": gamma,
            "params": params or {"gamma": gamma, "epsilon": 0.12, "alpha": 0.35}, "note": note,
            "env": {"map": {"cols": cols, "rows": rows}, "fingerprint": fingerprint, "optimal_return": optimal_return},
            "settings": {"eval_every": 10, "changed_during_run": list(changed)},
            "metrics": {"reward_per_episode": list(rewards), "steps_per_episode": [5] * n, "reached_goal": [1] * n},
            "greedy_eval": evals or [], "summary": dict(base_summary, **(summary or {}))}


# ------------------------------------------------------------------ таблиця
def test_rows_for_view():
    run = fake_run("a", note="перший", changed=["gamma"])
    row = cv.rows_for_view([run])[0]
    cells = dict(zip([key for key, _title in cv.TABLE_COLUMNS], row))
    assert cells["agent"] == "Таблиця" and cells["seed"] == "1" and cells["map"] == "7×6"
    assert cells["first_success_episode"] == "2"
    assert cells["convergence_episode"] == "—"                      # не зійшовся (null)
    assert cells["return_ratio"] == "0.750" and cells["cost_ratio"] == "1.077"
    assert cells["mean_reward_last_50"] == "12.3" and cells["success_rate_last_50"] == "80 %"
    assert cells["total_env_steps"] == "1234" and cells["duration_sec"] == "3.2"
    assert cells["note"] == "перший · змінено посеред запуску: γ"
    assert len(row) == len(cv.TABLE_COLUMNS)


def test_rows_for_view_none_everywhere():
    run = fake_run("b", agent="dqn", seed=None,
                   summary={k: None for k in ("first_success_episode", "return_ratio", "cost_ratio",
                                              "mean_reward_last_50", "success_rate_last_50")})
    cells = dict(zip([k for k, _t in cv.TABLE_COLUMNS], cv.rows_for_view([run])[0]))
    assert cells["agent"] == "Нейромережа" and cells["seed"] == "—"
    for key in ("first_success_episode", "return_ratio", "cost_ratio", "mean_reward_last_50", "success_rate_last_50"):
        assert cells[key] == "—"


# ------------------------------------------------------------------- криві
def test_chart_series_downsampling():
    values = [float((i * 37) % 101) for i in range(3000)]
    run = fake_run("c", rewards=values)
    xs, ys = cv.run_series(run, "reward", window=20, max_points=500)
    assert len(xs) == len(ys) <= 500
    assert xs[0] == 1 and xs[-1] == 3000                            # перша й остання точки на місці
    smooth = moving_average(values, 20)
    for x, y in zip(xs, ys):                                       # ковзне середнє — точне в кожній точці
        assert y == pytest.approx(smooth[x - 1])
    assert xs == sorted(set(xs))


def test_short_series_not_downsampled():
    run = fake_run("d", rewards=[1.0, 5.0, 3.0])
    assert cv.run_series(run, "reward", window=1) == ([1, 2, 3], [1.0, 5.0, 3.0])
    assert cv.run_series(run, "steps", window=1)[1] == [5, 5, 5]
    assert cv.run_series(run, "success", window=1)[1] == [1, 1, 1]


def test_optimal_series_from_greedy_eval():
    """«Оптимальність шляху»: винагорода жадібного шляху / V* у кожній оцінці; не дійшов до фінішу — 0."""
    evals = [{"episode": 10, "reaches_finish": False, "path_return": -1.9},
             {"episode": 20, "reaches_finish": True, "path_return": 5.0},
             {"episode": 30, "reaches_finish": True, "path_return": 10.0}]
    run = fake_run("e", evals=evals, optimal_return=10.0)
    assert cv.run_series(run, "optimal", window=1) == ([10, 20, 30], [0.0, 0.5, 1.0])


def test_unknown_metric():
    with pytest.raises(ValueError):
        cv.run_series(fake_run("f"), "speed")


# ------------------------------------------------------------------ кольори
def test_series_colors_stable():
    ids = ["20261119-182423_tabular_7x6_noseed", "20261119-182443_dqn_7x6_noseed", "x", "y"]
    a = cv.assign_colors(ids)
    b = cv.assign_colors(list(reversed(ids)))
    assert a[ids[0]] == cv.assign_colors([ids[0]])[ids[0]]          # той самий запуск — той самий колір
    assert len(set(a.values())) == len(ids)                         # у межах вибору кольори різні
    assert all(c.startswith("#") and len(c) == 7 for c in a.values())
    assert set(a.values()) == set(b.values())


def test_colors_unique_for_many_runs():
    ids = [f"run{k}" for k in range(len(cv.PALETTE) + 3)]
    colors = cv.assign_colors(ids)
    assert len(set(colors[i] for i in ids[:len(cv.PALETTE)])) == len(cv.PALETTE)


# -------------------------------------------------------------- попередження
def test_map_mismatch_banner():
    same = [fake_run("a"), fake_run("b", seed=2)]
    assert cv.mismatch_banner(same) == ""
    other = [fake_run("a"), fake_run("b", fingerprint="fp-B", cols=10, rows=8)]
    text = cv.mismatch_banner(other)
    assert "різних картах" in text and "7x6" in text and "10x8" in text
    assert "різними γ" in cv.mismatch_banner([fake_run("a"), fake_run("b", gamma=0.99)])


# --------------------------------------------------------------- групування
def test_group_by_seed_toggle():
    r1 = fake_run("a", seed=1, rewards=[0.0, 2.0, 4.0])
    r2 = fake_run("b", seed=2, rewards=[2.0, 4.0, 8.0])
    r3 = fake_run("c", agent="dqn", seed=1, rewards=[1.0, 1.0, 1.0],
                  params={"gamma": 0.9, "epsilon": 0.12, "hidden": 64})
    groups = cv.group_series([r1, r2, r3], "reward", window=1)
    assert [g["n_runs"] for g in groups] == [2, 1]                  # одна лінія на конфігурацію
    g = groups[0]
    assert g["xs"] == [1, 2, 3] and g["mean"] == [1.0, 3.0, 6.0]
    sd = [statistics.stdev(p) for p in ([0.0, 2.0], [2.0, 4.0], [4.0, 8.0])]
    assert g["low"] == pytest.approx([m - s for m, s in zip(g["mean"], sd)])
    assert g["high"] == pytest.approx([m + s for m, s in zip(g["mean"], sd)])
    assert "2 seed" in g["label"] and "Таблиця" in g["label"]
    single = groups[1]
    assert single["low"] == single["mean"] == single["high"]       # один запуск — смуги немає (σ = 0)


def test_group_with_different_lengths_and_downsampling():
    r1 = fake_run("a", seed=1, rewards=[1.0] * 1200)
    r2 = fake_run("b", seed=2, rewards=[3.0] * 800)
    g = cv.group_series([r1, r2], "reward", window=5, max_points=300)[0]
    assert len(g["xs"]) <= 300 and g["xs"][0] == 1 and g["xs"][-1] == 1200
    assert g["mean"][0] == 2.0 and g["mean"][-1] == 1.0             # після 800-го епізоду лишився один запуск


# ------------------------------------------------------------ підписи й фікстури
def test_labels():
    run = fake_run("20261119-182423_tabular_7x6_noseed", seed=None, note="автозбереження")
    assert cv.run_label(run) == "Таблиця · 7×6 · без seed"
    item = cv.list_label(run)
    assert "19.11 18:24" in item and "Таблиця" in item and "автозбереження" in item


def test_fixtures_are_valid_runs():
    for name, agent in (("fixture_run_tabular.json", "tabular"), ("fixture_run_dqn.json", "dqn")):
        run = load_run(FIXTURES / name)
        assert run["agent"] == agent and run["seed"] == 1
        rows = cv.rows_for_view([run])
        assert len(rows[0]) == len(cv.TABLE_COLUMNS)
        for metric in cv.METRICS:
            xs, ys = cv.run_series(run, metric, window=3)
            assert len(xs) == len(ys) > 0


def test_inputs_not_modified():
    run = fake_run("a", rewards=[1.0, 2.0, 3.0, 4.0])
    before = copy.deepcopy(run)
    cv.rows_for_view([run])
    cv.run_series(run, "reward", window=2)
    cv.group_series([run], "reward", window=2)
    assert run == before


def test_banner_explains_same_size_different_maps():
    """Дві карти 7×6 з різними перешкодами: «7x6, 7x6» саме по собі незрозуміле — пояснюємо, чим вони різні."""
    text = cv.mismatch_banner([fake_run("a"), fake_run("b", fingerprint="fp-B")])
    assert "різних картах" in text and "розмір однаковий" in text


def test_color_does_not_depend_on_selection():
    """Колір запуску не залежить від того, які ще запуски вибрано і в якому порядку
    (поки його «власний» колір не зайнятий іншим запуском)."""
    alone = {}
    ids = []
    for k in range(200):                      # запуски з різними «власними» кольорами
        rid = f"20261119-1200{k:02d}_tabular_7x6_seed{k}"
        color = cv.assign_colors([rid])[rid]
        if color not in alone.values():
            alone[rid] = color
            ids.append(rid)
        if len(ids) == 4:
            break
    assert len(ids) == 4, "різні запуски мають отримувати різні «власні» кольори"
    for order in (ids, list(reversed(ids)), ids[1:] + ids[:1]):
        colors = cv.assign_colors(order)
        assert all(colors[rid] == alone[rid] for rid in order)
