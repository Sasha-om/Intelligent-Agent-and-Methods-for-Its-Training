"""Генератор фікстур для тестів вікна порівняння (етап 8): два невеликі запуски на карті K1.

Запуск (з папки "Platform for Interaction"):
    & "..\\Pytorch_test\\.venv\\Scripts\\python.exe" tests\\make_fixture_runs.py
Фікстури йдуть у git; перегенеровувати — лише якщо змінилась схема файлу запуску.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from qlearn.maps import BUILTIN_MAPS  # noqa: E402
from qlearn.runner import run_experiment  # noqa: E402

FIXTURES = {
    "fixture_run_tabular.json": ("tabular", 120),
    "fixture_run_dqn.json": ("dqn", 40),
}


def main():
    for name, (mode, episodes) in FIXTURES.items():
        result = run_experiment(mode, BUILTIN_MAPS["default"], seed=1, episodes=episodes, note="фікстура етапу 8")
        result["run_id"] = f"fixture_{mode}_7x6_seed1"        # стале ім'я замість часу створення
        result["created"] = "2026-11-19T00:00:00"
        result["summary"]["duration_sec"] = 1.0               # щоб файл не змінювався від прогону до прогону
        (HERE / name).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print("записано", name, result["summary"]["episodes"], "епізодів")


if __name__ == "__main__":
    main()
