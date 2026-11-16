"""Юніт-тести фасаду QModel із режимами «tabular» / «dqn» (етап 6, FR-N1…N3, FR-N5, FR-N7)."""
import math
import random
import types

import pytest

import helpers


def test_default_mode_is_tabular(trainer):
    m = trainer.QModel()
    assert m.mode == "tabular" and m.agent.name == "tabular"


def test_unknown_mode(trainer):
    with pytest.raises(ValueError):
        trainer.QModel().set_mode("sarsa")


def test_switch_keeps_env_and_shared_params(trainer, maps):
    pytest.importorskip("torch")
    m = trainer.QModel()
    m.load_env(maps["K3_rough"])
    m.alpha, m.gamma, m.epsilon = 0.5, 0.95, 0.2
    random.seed(0)
    m.set_mode("dqn")
    assert m.mode == "dqn" and m.agent.name == "dqn"
    assert m.env.fingerprint() == trainer.GridEnv(maps["K3_rough"]).fingerprint()
    assert (m.gamma, m.epsilon) == (0.95, 0.2) and (m.agent.gamma, m.agent.epsilon) == (0.95, 0.2)
    assert m.alpha == 0.5                       # α таблиці зберігається, хоч DQN його не використовує
    m.set_mode("tabular")
    assert m.agent.alpha == 0.5 and all(v == 0.0 for v in m.q)   # повернення — нова чиста таблиця


def test_dqn_mode_works_through_facade(trainer, maps):
    pytest.importorskip("torch")
    m = trainer.QModel()
    random.seed(1)
    m.set_mode("dqn")
    res = m.apply_step(m.start_i)
    assert {"target", "loss", "old_q", "new_q"} <= set(res)
    m.run_episode_silent()
    assert len(m.q) == m.cols * m.rows * 4 and all(math.isfinite(v) for v in m.q)
    assert m.q_table_data() and m.greedy_path()[0] == m.start_i
    m.cols, m.rows = 9, 5                       # редактор змінює карту в режимі DQN
    m.finish = (8, 4)
    m.rebuild()
    assert len(m.q) == 9 * 5 * 4


def test_slider_gamma_reaches_dqn(trainer):
    pytest.importorskip("torch")
    m = trainer.QModel()
    m.set_mode("dqn")
    m.gamma = 0.8
    assert m.agent.gamma == 0.8


def test_json_export_tabular_has_no_mode_key(trainer):
    """Табличний експорт не змінився (golden етапу 1); ключ mode з'являється лише в режимі DQN."""
    m = trainer.QModel()
    data = trainer.App.json_data(types.SimpleNamespace(m=m, episode=0, max_episodes=600))
    assert "mode" not in data


def test_json_export_dqn_has_mode_and_params(trainer):
    pytest.importorskip("torch")
    m = trainer.QModel()
    random.seed(2)
    m.set_mode("dqn")
    data = trainer.App.json_data(types.SimpleNamespace(m=m, episode=0, max_episodes=600))
    assert data["mode"] == "dqn" and data["agentParams"]["features"] == "compact"
    assert set(data) >= {"params", "environment", "table", "greedyPath"}


def test_golden_still_via_facade_after_mode_roundtrip(trainer, maps):
    """Таблиця → DQN → таблиця: навчання таблиці знову дає golden (перемикач нічого не «залишає»)."""
    import json
    golden = json.loads((helpers.TESTS_DIR / "golden_tabular.json").read_text(encoding="utf-8"))
    m = trainer.QModel()
    if helpers.load_trainer().dqn_available():
        m.set_mode("dqn")
        m.set_mode("tabular")
    m.load_env(maps["K1_default"])
    helpers.train(m, 600, seed=0)
    assert helpers.q_hash(m.q) == golden["scenarios"]["K1_default_seed0_600"]["snapshot"]["q_sha256"]
