"""Регресія етапу 9: результати експериментів у results/exp-* (AC-7, AC-9).

- кожен файл запуску читається (схема v1), summary не порожній, довжини кривих = кількості епізодів;
- на стандартній карті K1 обидва агенти дають оптимальний шлях (return_ratio = 1.0) у ≥ 4 з 5 seed (AC-9).

Якщо експериментів ще немає (свіжий клон без results/exp-*) — тест пропускається.
"""
from collections import defaultdict

import pytest

from qlearn.environment import GridEnv
from qlearn.maps import load_map
from qlearn.recording import RESULTS_DIR, list_runs, load_run

EXP_DIRS = sorted(p for p in RESULTS_DIR.glob("exp-*") if p.is_dir())
RUN_FILES = [p for d in EXP_DIRS for p in list_runs(d, recursive=True)]

pytestmark = pytest.mark.skipif(not RUN_FILES, reason="немає results/exp-* (експерименти етапу 9 не запускались)")


@pytest.mark.parametrize("path", RUN_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_run_file_valid(path):
    run = load_run(path)
    s = run["summary"]
    n = s["episodes"]
    assert n > 0 and s["total_env_steps"] > 0
    for key in ("reward_per_episode", "steps_per_episode", "reached_goal"):
        assert len(run["metrics"][key]) == n
    assert run["greedy_eval"], "немає жодної оцінки жадібного шляху"
    assert run["seed"] is not None, "експерименти мають бути з фіксованим seed"


def test_k1_both_agents_optimal_in_4_of_5_seeds():
    """AC-9: стандартна карта, γ = 0.9 — оптимальний шлях у ≥ 4 з 5 seed для кожного агента."""
    k1 = GridEnv(load_map("default")).fingerprint()
    ratios = defaultdict(list)
    for path in RUN_FILES:
        run = load_run(path)
        if run["env"]["fingerprint"] == k1 and run["gamma"] == 0.9:
            ratios[run["agent"]].append(run["summary"]["return_ratio"])
    assert set(ratios) == {"tabular", "dqn"}, "у results/exp-* немає запусків обох агентів на K1"
    for agent, values in ratios.items():
        good = sum(1 for v in values if v is not None and v >= 1.0 - 1e-6)   # 1.0000000032 — округлення V* у файлі
        assert len(values) >= 5 and good >= 4, f"{agent}: оптимальний шлях у {good} з {len(values)} seed"
