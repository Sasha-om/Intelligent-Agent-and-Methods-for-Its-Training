"""Вимірює швидкість табличного навчання і записує базову лінію в tests/golden_perf.json.

Запускати на ПК користувача (швидкість залежить від комп'ютера):
    cd "C:\\Q-lerning and AI\\Platform for Interaction"
    & "..\\Pytorch_test\\.venv\\Scripts\\python.exe" tests\\bench_tabular.py

На етапі 2 тест regression_perf.py вимагатиме ≥ 85 % цієї швидкості після рефакторингу.
"""
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import helpers

EPISODES, REPEATS = 600, 7


def measure():
    trainer = helpers.load_trainer()
    maps = helpers.build_maps()
    results = {}
    for name in ("K1_default", "R2_15x15"):
        rates = []
        for _ in range(REPEATS):
            m = trainer.QModel()
            m.load_env(maps[name])
            t0 = time.perf_counter()
            helpers.train(m, EPISODES, seed=0)
            rates.append(EPISODES / (time.perf_counter() - t0))
        results[name] = round(statistics.median(rates), 1)
    return results


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rates = measure()
    data = {
        "measured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "machine": platform.node(),
        "episodes": EPISODES,
        "repeats": REPEATS,
        "episodes_per_sec_median": rates,
    }
    out = helpers.TESTS_DIR / "golden_perf.json"
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Записано: {out}")
    for name, rate in rates.items():
        print(f"  {name}: {rate} епізодів/с (медіана з {REPEATS})")


if __name__ == "__main__":
    main()
