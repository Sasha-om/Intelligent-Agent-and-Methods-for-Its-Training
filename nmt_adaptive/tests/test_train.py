"""Тести навчання (симуляція): коротке навчання, детермінізм, типи учнів, файли результатів, абляції, міст до тренажера."""
import numpy as np
import pytest

import config
from core.agent import QAgent
from sim import legacy, train

SHORT = 300


@pytest.fixture(scope="module")
def short_run() -> tuple:
    """Одне коротке навчання на весь модуль (300 сесій) — щоб тести були швидкі."""
    return train.train(episodes=SHORT, seed=0)


def test_short_training_runs_and_learns_something(short_run: tuple) -> None:
    """300 сесій: історія з 300 рядків, ε спадає, таблиця Q змінилась, агент заморожений після навчання."""
    agent, history = short_run
    assert len(history) == SHORT
    assert history[0]["epsilon"] == config.EPS_START and history[-1]["epsilon"] == pytest.approx(config.EPS_END)
    assert np.count_nonzero(agent.q) > 100
    assert agent.frozen and agent.episodes == SHORT


def test_student_types_alternate(short_run: tuple) -> None:
    """Типи учнів ідуть строго по черзі — кожного порівну."""
    _agent, history = short_run
    types = list(config.STUDENT_TYPES)
    assert [r["student_type"] for r in history[:8]] == types * 2
    assert all(sum(r["student_type"] == t for r in history) == SHORT // 4 for t in types)


def test_history_metrics_are_sane(short_run: tuple) -> None:
    """У кожній сесії: 1–20 завдань, ≤ 3 перерв, частка в зоні від 0 до 1; відмова — це сесія коротша за 20."""
    _agent, history = short_run
    for row in history:
        assert 1 <= row["items"] <= config.SESSION_ITEMS and 0 <= row["breaks"] <= config.MAX_BREAKS
        assert 0 <= row["zpd_share"] <= 1
        assert row["dropped"] or row["items"] == config.SESSION_ITEMS


def test_same_seed_same_result() -> None:
    """Два навчання з тим самим seed → однакові таблиці Q (відтворюваність)."""
    a, _ = train.train(episodes=60, seed=5)
    b, _ = train.train(episodes=60, seed=5)
    c, _ = train.train(episodes=60, seed=6)
    assert np.array_equal(a.q, b.q) and not np.array_equal(a.q, c.q)


def test_no_break_ablation_never_takes_a_break() -> None:
    """Абляція «без перерви»: жодної перерви за все навчання."""
    _agent, history = train.train(episodes=80, seed=1, ablation="no_break")
    assert sum(r["breaks"] for r in history) == 0
    with pytest.raises(ValueError, match="Невідома абляція"):
        train.train(episodes=1, ablation="magic")


def test_results_files(short_run: tuple, tmp_path) -> None:
    """CSV і PNG кривої та політики створюються, назви з «симуляція», CSV має 1 + N рядків."""
    agent, history = short_run
    files = [*train.save_curve(history, tmp_path), *train.save_policy(agent, tmp_path)]
    assert all(f.exists() and f.stat().st_size > 0 and "симуляція" in f.name for f in files)
    curve_lines = files[0].read_bytes().decode("utf-8-sig").split("\r\n")
    policy_lines = files[2].read_bytes().decode("utf-8-sig").split("\r\n")
    assert len(curve_lines) == SHORT // config.CURVE_CSV_EVERY + 1 and curve_lines[0].startswith("сесія;")
    assert len(policy_lines) == config.N_STATES + 1


def test_policy_file_roundtrip(short_run: tuple, tmp_path) -> None:
    """Збережена політика читається QAgent.load і дає ту саму таблицю."""
    agent, _history = short_run
    loaded = QAgent.load(agent.save(tmp_path / "policy.json"), np.random.default_rng(0))
    assert np.array_equal(loaded.q, agent.q)
    assert train.policy_path("no_break").name == "policy_no_break.json"
    assert train.policy_path() == config.POLICY_PATH


def test_main_writes_everything(tmp_path, capsys) -> None:
    """Командний рядок: коротке навчання пише політику й 4 файли результатів, друкує підсумок."""
    policy = tmp_path / "p.json"
    train.main(["--episodes", "40", "--seed", "3", "--out", str(tmp_path), "--policy", str(policy)])
    out = capsys.readouterr().out
    assert policy.exists() and "СИМУЛЯЦІЯ" in out and "нагорода" in out
    assert len(list(tmp_path.glob("симуляція_*"))) == 4


def test_legacy_bridge_formats_like_trainer(tmp_path) -> None:
    """Міст до тренажера: CSV з «;», десятковою комою і BOM; ковзне середнє «назад»."""
    path = legacy.write_csv(tmp_path / "t.csv", [{"a": 1.5, "b": True}], ["a", "b"])
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf") and raw.decode("utf-8-sig") == "a;b\r\n1,5;так"
    assert legacy.moving_average([1, 2, 3, 4], 2) == [1.0, 1.5, 2.5, 3.5]


def test_legacy_fallbacks_match_trainer(tmp_path) -> None:
    """Запасні копії (якщо тренажера поруч немає) дають той самий результат, що й функції тренажера."""
    rows, fields = [{"x": 0.25, "y": "a;b", "z": None}], ["x", "y", "z"]
    a = legacy._fallback_csv(tmp_path / "f.csv", rows, fields).read_bytes()
    b = legacy.write_csv(tmp_path / "t.csv", rows, fields).read_bytes()
    assert a == b
    assert legacy._fallback_ma([3, 1, 2], 5) == legacy.moving_average([3, 1, 2], 5)


class _ToyEnv:
    """Іграшкове середовище на 2 кроки: дія 1 дає +1 і кінець, інші — 0 і ще один крок."""

    def __init__(self) -> None:
        """Лічильник кроків."""
        self.steps = 0

    def reset(self) -> int:
        """Старт у стані 0."""
        self.steps = 0
        return 0

    def valid_actions(self) -> list:
        """Усі 5 дій дозволені."""
        return list(range(config.N_ACTIONS))

    def step(self, action: int) -> tuple:
        """Крок іграшки; info — мінімальний, як у справжнього середовища."""
        self.steps += 1
        done = action == 1 or self.steps >= 2
        parts = {"two_wrong": 0}
        info = {"item": {"subtopic": "domain"}, "in_zpd": action == 1, "reward_parts": parts,
                "breaks": 0, "dropped": False}
        return 1, (1.0 if action == 1 else 0.0), done, info

    def session_gain(self) -> float:
        """Приріст для метрик."""
        return 0.0


def test_run_episode_on_toy_env() -> None:
    """run_episode: правильно передає (s, a, r, s′, done) агентові і рахує метрики."""
    agent = QAgent(np.random.default_rng(0), alpha_mode="const")
    agent.epsilon = lambda: 0.0                                      # без випадковості: жадібна дія 0 → потім 0
    row = train.run_episode(agent, _ToyEnv())
    assert row["items"] == 2 and row["reward"] == 0.0 and row["zpd_share"] == 0.0
    assert agent.visits[0, 0] == 1 and agent.visits[1, 0] == 1


def test_policy_rows_have_no_break_column_and_low_visits(short_run: tuple) -> None:
    """У CSV політики є «найкраща без перерви» (ніколи не перерва) і позначка «мало даних»."""
    agent, _history = short_run
    rows = train.policy_rows(agent)
    assert all(r["найкраща_без_перерви"] != "перерва" for r in rows)
    assert any(r["мало_даних"] for r in rows)
    assert "без перерви" in train.action_counts(agent)


def test_policy_meta_saved(tmp_path) -> None:
    """main записує в policy.json seed, абляцію і кількість сесій."""
    import json
    policy = tmp_path / "p.json"
    train.main(["--episodes", "8", "--seed", "4", "--out", str(tmp_path), "--policy", str(policy), "--no-plot"])
    meta = json.loads(policy.read_text(encoding="utf-8"))["meta"]
    assert meta == {"seed": 4, "ablation": "none", "episodes": 8, "gain_source": "true"}
