"""Юніт-тести qlearn.comparison — порівняння збережених запусків (етап 5, FR-C1…FR-C3)."""
import pytest

from qlearn.comparison import (ROW_FIELDS, aggregate, compare, mean_curve, moving_average, run_curve, to_csv,
                               write_csv)


def fake_run(agent="tabular", seed=0, fp="aaaa", gamma=0.9, conv=30, ratio=1.0, rewards=(1.0, 2.0, 3.0),
             cols=7, rows=6, params=None, note=""):
    """Мінімальний запуск у форматі схеми v1 (лише поля, які потрібні порівнянню)."""
    return {
        "run_id": f"20261114-120000_{agent}_{cols}x{rows}_seed{seed}", "agent": agent, "seed": seed, "gamma": gamma,
        "params": dict(params or {"gamma": gamma, "epsilon": 0.12}),
        "env": {"map": {"cols": cols, "rows": rows}, "fingerprint": fp, "optimal_cost": 11, "optimal_return": 10.572},
        "summary": {"episodes": len(rewards), "first_success_episode": 2, "convergence_episode": conv,
                    "final_optimal": conv is not None, "return_ratio": ratio, "return_gap": 0.0,
                    "final_path_cost": 11, "cost_ratio": 1.0, "mean_reward_last_50": sum(rewards) / len(rewards),
                    "success_rate_last_50": 1.0, "total_env_steps": 100, "duration_sec": 0.5},
        "metrics": {"reward_per_episode": list(rewards), "steps_per_episode": [5] * len(rewards),
                    "reached_goal": [1] * len(rewards)},
        "note": note,
    }


# ------------------------------------------------------------- таблиця
def test_compare_one_row_per_run():
    out = compare([fake_run(seed=0), fake_run(agent="dqn", seed=0)])
    assert [r["agent"] for r in out["rows"]] == ["tabular", "dqn"]
    assert list(out["rows"][0]) == ROW_FIELDS
    assert out["rows"][0]["map"] == "7x6" and out["rows"][1]["convergence_episode"] == 30
    assert out["warnings"] == []


def test_warns_on_different_maps_and_gamma():
    out = compare([fake_run(fp="aaaa"), fake_run(fp="bbbb", cols=10, rows=8), fake_run(gamma=0.99)])
    text = " ".join(out["warnings"])
    assert "різних картах" in text and "γ" in text


# ---------------------------------------------------- середнє ± σ по seed
def test_aggregate_mean_std_by_config():
    runs = [fake_run(seed=0, conv=30, ratio=1.0), fake_run(seed=1, conv=50, ratio=1.0),
            fake_run(seed=2, conv=None, ratio=0.8), fake_run(agent="dqn", seed=0, conv=20)]
    groups = aggregate(runs)
    assert [g["agent"] for g in groups] == ["tabular", "dqn"]
    tab = groups[0]
    assert tab["n_runs"] == 3 and tab["seeds"] == [0, 1, 2]
    assert tab["converged"] == "2 з 3"          # не «2/3»: Excel перетворив би на дату
    assert tab["convergence_episode_mean"] == pytest.approx(40.0)
    assert tab["convergence_episode_std"] == pytest.approx(14.1421356, abs=1e-6)     # вибіркове σ з [30, 50]
    assert tab["return_ratio_mean"] == pytest.approx(0.9333333, abs=1e-6)
    assert tab["return_ratio_std"] == pytest.approx(0.1154701, abs=1e-6)
    assert groups[1]["n_runs"] == 1 and groups[1]["convergence_episode_std"] == 0.0


def test_aggregate_separates_different_params_and_maps():
    runs = [fake_run(seed=0), fake_run(seed=1, params={"gamma": 0.9, "epsilon": 0.3}), fake_run(seed=2, fp="bbbb")]
    assert len(aggregate(runs)) == 3


def test_aggregate_all_none():
    g = aggregate([fake_run(conv=None), fake_run(seed=1, conv=None)])[0]
    assert g["converged"] == "0 з 2" and g["convergence_episode_mean"] is None and g["convergence_episode_std"] is None


# --------------------------------------------------------------- криві
def test_moving_average_exact():
    assert moving_average([1, 2, 3, 4, 5], 2) == pytest.approx([1, 1.5, 2.5, 3.5, 4.5])
    assert moving_average([1, 2, 3, 4, 5], 3) == pytest.approx([1, 1.5, 2, 3, 4])
    assert moving_average([4, 8], 1) == [4, 8]
    assert moving_average([], 5) == []


def test_mean_curve_different_lengths():
    assert mean_curve([[1, 2, 3], [3, 4]]) == pytest.approx([2, 3, 3])


def test_run_curve_metrics():
    run = fake_run(rewards=(2.0, 4.0, 6.0))
    assert run_curve(run, "reward", window=2) == pytest.approx([2, 3, 5])
    assert run_curve(run, "steps") == [5, 5, 5]
    assert run_curve(run, "success") == [1, 1, 1]
    with pytest.raises(ValueError):
        run_curve(run, "loss")


# ----------------------------------------------------------------- CSV
def test_csv_excel_format():
    rows = [{"agent": "tabular", "ratio": 0.5, "conv": None, "ok": True, "n": 3}]
    text = to_csv(rows, ["agent", "ratio", "conv", "ok", "n"])
    assert text == "agent;ratio;conv;ok;n\r\ntabular;0,5;;так;3"


def test_write_csv_has_bom(tmp_path):
    path = write_csv(tmp_path / "c.csv", [{"агент": "таблиця"}], ["агент"])
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and "таблиця".encode("utf-8") in raw
