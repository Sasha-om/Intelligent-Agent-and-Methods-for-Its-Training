"""Smoke-тест пакетного запуску: обидва агенти × 2 seed × 30 епізодів → 4 файли запусків і CSV порівняння."""
import pytest

pytest.importorskip("torch")

from qlearn.comparison import aggregate          # noqa: E402
from qlearn.recording import list_runs, load_run  # noqa: E402
from qlearn.runner import main                    # noqa: E402

pytestmark = pytest.mark.torch


def test_runner_both_agents(tmp_path):
    assert main(["--agent", "both", "--episodes", "30", "--seeds", "0", "1", "--out", str(tmp_path)]) == 0
    runs = [load_run(p) for p in list_runs(tmp_path)]
    assert sorted((r["agent"], r["seed"]) for r in runs) == [("dqn", 0), ("dqn", 1), ("tabular", 0), ("tabular", 1)]
    assert len(aggregate(runs)) == 2
    comparison = next(tmp_path.glob("comparison-*.csv")).read_text(encoding="utf-8-sig").splitlines()
    assert len(comparison) == 1 + 4
