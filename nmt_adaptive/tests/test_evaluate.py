"""Тести порівняння стратегій (симуляція): сесія, парність учнів, bootstrap, стійкість, файли."""
import random

import numpy as np
import pytest

import config
from core.baselines import RandomPolicy, RuleBreakPolicy, make_policies
from sim import evaluate
from sim.student_sim import VirtualStudent, clone


def test_run_session_metrics_are_in_range() -> None:
    """Метрики однієї сесії: частки в [0; 1], відмова 0/1, похибка оцінки ≥ 0."""
    student = VirtualStudent("weak_tiring", random.Random(1))
    m = evaluate.run_session(RuleBreakPolicy(6), student, 7)
    assert set(m) == set(evaluate.METRICS)
    assert 0 <= m["zpd_true"] <= 1 and 0 <= m["frustration"] <= 1 and m["dropped"] in (0.0, 1.0)
    assert m["est_error"] >= 0


def test_same_student_same_result_for_same_policy() -> None:
    """Той самий учень (clone) і той самий seed середовища → та сама сесія (основа парного порівняння)."""
    student = VirtualStudent("strong_tiring", random.Random(2))
    a = evaluate.run_session(RuleBreakPolicy(5), clone(student, 9), 9)
    b = evaluate.run_session(RuleBreakPolicy(5), clone(student, 9), 9)
    assert a == b


def test_evaluate_rows_are_paired() -> None:
    """У кожної стратегії — ті самі (seed, тип, учень) в тому самому порядку."""
    policies = [RandomPolicy(np.random.default_rng(0)), RuleBreakPolicy(6)]
    rows = evaluate.evaluate(policies, students=3, seeds=[0, 1])
    assert len(rows) == 2 * 4 * 3 * 2
    keys = {p.name: [(r["seed"], r["type"], r["student"]) for r in rows if r["policy"] == p.name] for p in policies}
    assert keys["random"] == keys["rule_6"]


def test_bootstrap_ci_on_known_data() -> None:
    """ДІ середнього [1..5] містить 3; вужчий для більшої вибірки; парна різниця однакових = 0."""
    rng = np.random.default_rng(0)
    mean, low, high = evaluate.bootstrap_ci(np.array([1, 2, 3, 4, 5]), rng)
    assert mean == 3 and low < 3 < high
    big = evaluate.bootstrap_ci(np.tile([1, 2, 3, 4, 5], 100), rng)
    assert (big[2] - big[1]) < (high - low)
    rows = [{"policy": p, "type": "weak_tiring", "seed": 0, "student": i, "gain": float(i)} for p in ("a", "b")
            for i in range(10)]
    test = evaluate.paired_test(rows, "a", "b", "gain", evaluate.ALL, rng)
    assert test["різниця"] == 0 and not test["значуща"]


def test_simulator_override_restores_config() -> None:
    """Зміна припущень симулятора тимчасова: після блоку (і навіть після помилки) config як був."""
    before_rate = config.STUDENT_TYPES["weak_tiring"]["fatigue_rate"]
    before_noise = config.SIM_SELF_REPORT_NOISE
    with pytest.raises(RuntimeError):
        with evaluate.simulator_override({"fatigue_scale": 2.0, "SIM_SELF_REPORT_NOISE": 9.0}):
            assert config.STUDENT_TYPES["weak_tiring"]["fatigue_rate"] == pytest.approx(2 * before_rate)
            assert config.SIM_SELF_REPORT_NOISE == 9.0
            raise RuntimeError("перевірка")
    assert config.STUDENT_TYPES["weak_tiring"]["fatigue_rate"] == before_rate
    assert config.SIM_SELF_REPORT_NOISE == before_noise


def test_quick_run_writes_all_files(tmp_path, capsys) -> None:
    """--quick: усі CSV і PNG у results-папці, назви з «симуляція», таблиця на екрані."""
    evaluate.main(["--quick", "--students", "6", "--out", str(tmp_path)])
    evaluate.main(["--students", "6", "--seeds", "0", "--out", str(tmp_path / "x"), "--no-ablations", "--no-robustness"])
    out = capsys.readouterr().out
    files = list(tmp_path.glob("симуляція_*"))
    assert len(files) == 12 and all(f.stat().st_size > 0 for f in files)
    header = lambda name: (tmp_path / name).read_bytes().decode("utf-8-sig").split("\r\n")[0]  # noqa: E731
    assert "значуща" in header("симуляція_стійкість.csv")
    assert "значуща_з_поправкою" in header("симуляція_bootstrap_тест.csv")
    assert "значуща_з_поправкою" in header("симуляція_абляції_тест.csv")
    assert "СИМУЛЯЦІЯ" in out and "Парний bootstrap" in out
    assert not (tmp_path / "x" / "симуляція_абляції.png").exists()


def test_ablation_agents_are_evaluated_in_their_own_conditions() -> None:
    """Агент «без перерви» оцінюється в середовищі без перерви (як навчався)."""
    q_no_break = [p for p in make_policies(np.random.default_rng(0)) if p.name == "q_no_break"]
    if not q_no_break:
        pytest.skip("немає data/policy_no_break.json")
    assert q_no_break[0].env_kwargs == {"allow_break": False}


def test_bonferroni_is_stricter() -> None:
    """Поправка Бонферроні: тест, значущий без поправки (z ≈ 2,5), при 75 тестах стає незначущим."""
    tests = [{"різниця": 0.5, "ДІ_низ": 0.108, "ДІ_верх": 0.892, "значуща": True} for _ in range(75)]
    out = evaluate.add_bonferroni(tests)
    assert out[0]["z"] == pytest.approx(2.5, abs=0.01) and not out[0]["значуща_з_поправкою"]
    assert evaluate.add_bonferroni([dict(tests[0])])[0]["значуща_з_поправкою"]


def test_training_variability_small(tmp_path, monkeypatch) -> None:
    """Мінливість навчання: для кожного seed — рядки «Q − правило/6» і «Q − правило/5»."""
    monkeypatch.setattr(config, "TRAIN_EPISODES", 40)
    rows = evaluate.training_variability_rows([0, 1], students=3, eval_seed=0, rng=np.random.default_rng(0))
    assert [r["seed_навчання"] for r in rows] == [0, 0, 1, 1]
    assert {"різниця_приросту", "значуща", "приріст_Q"} <= set(rows[0])
