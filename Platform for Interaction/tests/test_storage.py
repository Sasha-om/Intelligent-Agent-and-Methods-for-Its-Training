"""Юніт-тести збереження запусків у JSON (схема v1): qlearn.recording.save_run / load_run / list_runs (етап 4)."""
import json
import random
import re

import pytest

from qlearn.environment import GridEnv
from qlearn.recording import SCHEMA_VERSION, RunRecorder, list_runs, load_run, save_run
from qlearn.tabular import TabularAgent

REQUIRED = {"schema_version", "run_id", "created", "agent", "seed", "gamma", "params", "env", "settings",
            "metrics", "greedy_eval", "summary", "greedy_path", "versions", "note"}


def make_result(maps, episodes=30, seed=3, note=""):
    env = GridEnv(maps["K1_default"])
    random.seed(seed)
    return RunRecorder(env, TabularAgent(env), seed=seed, note=note).run(episodes)


def test_schema_v1_fields(maps):
    r = make_result(maps)
    assert SCHEMA_VERSION == 1 and r["schema_version"] == 1
    assert REQUIRED <= set(r)
    assert set(r["env"]) >= {"map", "fingerprint", "optimal_cost", "optimal_return"}
    assert set(r["metrics"]) == {"reward_per_episode", "steps_per_episode", "reached_goal"}
    assert len(r["metrics"]["reward_per_episode"]) == 30
    assert r["env"]["map"]["cols"] == 7 and r["env"]["optimal_cost"] == 11


def test_run_id_format(maps):
    r = make_result(maps, seed=3)
    assert re.fullmatch(r"\d{8}-\d{6}_tabular_7x6_seed3", r["run_id"])


def test_save_load_roundtrip(maps, tmp_path):
    r = make_result(maps)
    path = save_run(r, tmp_path)
    assert path.parent == tmp_path and path.name == r["run_id"] + ".json"
    assert load_run(path) == json.loads(json.dumps(r))


def test_filename_unique(maps, tmp_path):
    r = make_result(maps)
    first, second = save_run(r, tmp_path), save_run(r, tmp_path)
    assert first != second and second.name == r["run_id"] + "-2.json"


def test_utf8_note_and_size(maps, tmp_path):
    r = make_result(maps, episodes=3000, note="Тест «кирилиця»: γ=0.9")
    path = save_run(r, tmp_path)
    raw = path.read_bytes()
    assert "Тест «кирилиця»".encode("utf-8") in raw          # не \u-послідовності
    assert len(raw) <= 300_000, f"{len(raw)} байтів"
    assert load_run(path)["note"] == "Тест «кирилиця»: γ=0.9"


def test_unknown_version_rejected(maps, tmp_path):
    r = make_result(maps)
    r["schema_version"] = 99
    path = tmp_path / "x.json"
    path.write_text(json.dumps(r), encoding="utf-8")
    with pytest.raises(ValueError, match="верс"):
        load_run(path)


def test_corrupt_and_incomplete_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ не json", encoding="utf-8")
    with pytest.raises(ValueError, match="пошкоджен"):
        load_run(bad)
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"schema_version": 1, "agent": "tabular"}), encoding="utf-8")
    with pytest.raises(ValueError, match="бракує"):
        load_run(partial)


def test_list_runs(maps, tmp_path):
    a = save_run(make_result(maps, seed=1), tmp_path)
    b = save_run(make_result(maps, seed=2), tmp_path)
    (tmp_path / "readme.txt").write_text("не запуск", encoding="utf-8")
    assert list_runs(tmp_path) == sorted([a, b])
    assert list_runs(tmp_path / "немає") == []


def test_numbers_rounded(maps):
    r = make_result(maps)
    for v in r["metrics"]["reward_per_episode"]:
        assert round(v, 4) == v
