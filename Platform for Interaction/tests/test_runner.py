"""Юніт-тести пакетного запуску qlearn.runner і вбудованих карт qlearn.maps (етап 5, FR-C3)."""
import json

import pytest

import helpers
from qlearn.maps import BUILTIN_MAPS, load_map
from qlearn.recording import list_runs, load_run
from qlearn.runner import main, parse_args, run_experiment


# --------------------------------------------------------------- карти
@pytest.mark.parametrize("name, test_map", [("default", "K1_default"), ("wall_gap", "K2_wall_gap"), ("rough", "K3_rough")])
def test_builtin_maps_match_test_maps(maps, name, test_map):
    assert set(BUILTIN_MAPS) >= {"default", "wall_gap", "rough"}
    a, b = load_map(name), maps[test_map]
    assert (a["cols"], a["rows"], tuple(a["start"]), tuple(a["finish"])) == (b["cols"], b["rows"], b["start"], b["finish"])
    assert {tuple(p) for p in a["obstacles"]} == set(b["obstacles"]) and {tuple(p) for p in a["rough"]} == set(b["rough"])


def test_load_map_from_gui_export(tmp_path):
    """JSON-експорт тренажера: клітинки записані рядками "c,r"."""
    export = {"environment": {"cols": 5, "rows": 4, "start": "0,0", "finish": "4,3",
                              "obstacles": ["2,0", "2,1"], "rough": ["1,3"]}}
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    m = load_map(str(path))
    assert (m["cols"], m["rows"], m["start"], m["finish"]) == (5, 4, (0, 0), (4, 3))
    assert m["obstacles"] == [(2, 0), (2, 1)] and m["rough"] == [(1, 3)]


def test_load_map_from_run_file(tmp_path, maps):
    run = {"env": {"map": {"cols": 7, "rows": 6, "obstacles": [[3, 0]], "rough": [], "start": [0, 0], "finish": [6, 5]}}}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run), encoding="utf-8")
    m = load_map(str(path))
    assert m["obstacles"] == [(3, 0)] and m["finish"] == (6, 5)


def test_load_map_unknown():
    with pytest.raises(ValueError, match="карт"):
        load_map("lava")


# ------------------------------------------------------------- аргументи
def test_parse_args_full():
    a = parse_args(["--agent", "both", "--episodes", "50", "--seeds", "0", "1", "--map", "rough",
                    "--set", "lr=0.002", "--set", "features=onehot", "--set", "hidden=32",
                    "--alpha", "0.5", "--gamma", "0.99", "--note", "тест"])
    assert a.agent == "both" and a.episodes == 50 and a.seeds == [0, 1] and a.map == "rough"
    assert a.dqn_params == {"lr": 0.002, "features": "onehot", "hidden": 32}
    assert (a.alpha, a.gamma, a.epsilon, a.note) == (0.5, 0.99, 0.12, "тест")


def test_parse_args_defaults():
    a = parse_args([])
    assert (a.agent, a.episodes, a.seeds, a.map, a.out) == ("both", 600, [0, 1, 2, 3, 4], "default", None)


def test_parse_args_bad_set():
    with pytest.raises(SystemExit):
        parse_args(["--set", "lr"])


# ------------------------------------------------------------ запуск
def test_run_experiment_deterministic():
    def summary():
        r = run_experiment("tabular", load_map("default"), seed=3, episodes=80)
        s = dict(r["summary"])
        s.pop("duration_sec")
        return s, r["metrics"]
    assert summary() == summary()


def test_run_experiment_params_reach_agent():
    r = run_experiment("tabular", load_map("default"), seed=0, episodes=5, alpha=0.5, gamma=0.95, epsilon=0.2)
    assert r["params"] == {"alpha": 0.5, "gamma": 0.95, "epsilon": 0.2} and r["gamma"] == 0.95 and r["seed"] == 0


def test_main_tabular_writes_runs_and_csv(tmp_path, capsys):
    code = main(["--agent", "tabular", "--episodes", "30", "--seeds", "0", "1", "--out", str(tmp_path), "--note", "пробний"])
    assert code == 0
    runs = list_runs(tmp_path)
    assert len(runs) == 2 and all(load_run(p)["note"] == "пробний" for p in runs)
    csvs = sorted(p.name for p in tmp_path.glob("*.csv"))
    assert len(csvs) == 2 and any(n.startswith("comparison-") for n in csvs) and any(n.startswith("summary-") for n in csvs)
    out = capsys.readouterr().out
    assert "Підсумок" in out and "tabular" in out


def test_main_bad_map_returns_error(tmp_path, capsys):
    assert main(["--agent", "tabular", "--map", "lava", "--out", str(tmp_path)]) == 2
    assert "карт" in capsys.readouterr().out


def test_helpers_map_names_stable():
    assert helpers.build_maps()["K1_default"]["finish"] == (6, 5)


def test_run_experiment_dqn_deterministic():
    """seed фіксується ДО створення агента: ваги мережі й навчання DQN відтворювані."""
    pytest.importorskip("torch")

    def run():
        r = run_experiment("dqn", load_map("default"), seed=4, episodes=5)
        return r["metrics"], r["summary"]["final_path_return"]
    assert run() == run()
