"""Регресія табличного режиму: поточний код має давати ТОЧНО той самий результат, що й golden-знімок.

Якщо тест упав — поведінка табличного Q-навчання змінилась (NFR-1 у docs/SPEC.md).
Не перегенеровуйте golden, щоб «полагодити» тест, без погодження користувача.
"""
import json

import pytest

import helpers
import make_golden

GOLDEN = json.loads((helpers.TESTS_DIR / "golden_tabular.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(make_golden.SCENARIOS))
def test_golden_snapshot(trainer, maps, name):
    expected = GOLDEN["scenarios"][name]
    m, map_name, seed, episodes = make_golden.run_scenario(trainer, maps, name)
    assert (map_name, seed, episodes) == (expected["map"], expected["seed"], expected["episodes"])
    assert {"alpha": m.alpha, "gamma": m.gamma, "epsilon": m.epsilon} == expected["params"]
    actual = helpers.snapshot(m)
    exp = expected["snapshot"]
    assert actual["path_len"] == exp["path_len"]
    assert actual["path_cost"] == exp["path_cost"]
    assert actual["reaches_finish"] == exp["reaches_finish"]
    assert actual["sum_q_along_path"] == pytest.approx(exp["sum_q_along_path"], abs=1e-9)
    assert actual["q_sha256"] == exp["q_sha256"], "Q-таблиця відрізняється від golden-знімка"


@pytest.mark.parametrize("name", sorted(make_golden.SCENARIOS))
def test_golden_snapshot_direct_agent(maps, name):
    """Той самий golden — але через qlearn.TabularAgent напряму, без фасаду тренажера (етап 2)."""
    import random

    from qlearn.environment import GridEnv
    from qlearn.tabular import TabularAgent

    expected = GOLDEN["scenarios"][name]
    env = GridEnv(maps[expected["map"]])
    agent = TabularAgent(env)
    random.seed(expected["seed"])
    for _ in range(expected["episodes"]):
        agent.run_episode()
    actual = helpers.snapshot(helpers.AgentView(env, agent))
    assert actual["q_sha256"] == expected["snapshot"]["q_sha256"]
    assert actual["path_cost"] == expected["snapshot"]["path_cost"]


def test_known_baseline_numbers():
    """Числа, виміряні до початку робіт (2026-11-11): 12 клітинок, вартість 11, сума Q 294.8517."""
    snap = GOLDEN["scenarios"]["K1_default_seed0_600"]["snapshot"]
    assert snap["path_len"] == 12 and snap["path_cost"] == 11
    assert snap["sum_q_along_path"] == pytest.approx(294.8517, abs=1e-4)


def test_export_format_unchanged(trainer, maps):
    """CSV і JSON експорт (кнопки «Завантажити CSV/JSON») — той самий формат і вміст."""
    m, _map, _seed, episodes = make_golden.run_scenario(trainer, maps, make_golden.EXPORT_SCENARIO)
    actual = make_golden.export_fingerprint(trainer, m, episodes)
    expected = GOLDEN["export"]
    for key in ("csv_header", "csv_lines", "json_keys", "json_params_keys",
                "json_environment_keys", "json_table_row_keys"):
        assert actual[key] == expected[key], f"змінився формат експорту: {key}"
    assert actual["csv_sha256"] == expected["csv_sha256"], "змінився вміст CSV"
    assert actual["json_sha256"] == expected["json_sha256"], "змінився вміст JSON"
