"""Тест симуляції пілоту (малий запуск): файли, режими, потужність у межах."""
import numpy as np

import config
from core.agent import QAgent
from sim import pilot_sim


def test_quick_run_writes_files(tmp_path) -> None:
    """--quick створює CSV і PNG з трьома режимами і таблицею потужності."""
    pilot_sim.main(["--quick", "--out", str(tmp_path)])
    names = {p.name for p in tmp_path.iterdir()}
    assert {"симуляція_пілоту_режими.csv", "симуляція_пілоту_потужність.csv",
            "симуляція_пілоту_потужність.png"} <= names


def test_lesson_shape_and_same_students() -> None:
    """Урок: вхід 0–6; вихід 0–6 або None (кинув); ті самі учні в усіх режимах (однаковий вхідний бал)."""
    agent = QAgent.load(config.POLICY_PATH, np.random.default_rng(0))
    rows = pilot_sim.simulate(8, seed=3, agent=agent)
    assert len(rows) == 8 * len(config.PILOT_ARMS)
    for r in rows:
        assert 0 <= r["вхід"] <= 6 and (r["вихід"] is None or 0 <= r["вихід"] <= 6)
    by_student = {}
    for r in rows:
        by_student.setdefault(r["учень"], set()).add(r["вхід"])
    assert all(len(v) == 1 for v in by_student.values())


def test_power_between_zero_and_one() -> None:
    """Потужність — частка від 0 до 1 для кожного n і порівняння."""
    agent = QAgent.load(config.POLICY_PATH, np.random.default_rng(0))
    rows = pilot_sim.simulate(40, seed=0, agent=agent)
    power = pilot_sim.power_rows(rows, 20, np.random.default_rng(0))
    assert len(power) == 2 * len(config.PILOT_SIM_GROUP_SIZES)
    assert all(0 <= r["потужність"] <= 1 for r in power)
