"""Генерує tests/golden_tabular.json — еталонний знімок поведінки табличного режиму.

Запускати ЛИШЕ коли поведінку свідомо змінено й це погоджено (інакше golden втрачає сенс):
    cd "C:\\Q-lerning and AI\\Platform for Interaction"
    & "..\\Pytorch_test\\.venv\\Scripts\\python.exe" tests\\make_golden.py

Тест tests/regression_tabular_golden.py порівнює поточний код із цим файлом.
"""
import hashlib
import json
import platform
import sys
import types
from datetime import datetime, timezone

import helpers

# (назва карти з helpers.MAPS, seed, кількість епізодів)
SCENARIOS = {
    "K1_default_seed0_600": ("K1_default", 0, 600),
    "K3_rough_seed1_800": ("K3_rough", 1, 800),
    "R1_10x8_seed2_800": ("R1_10x8", 2, 800),
}
EXPORT_SCENARIO = "K1_default_seed0_600"


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_scenario(trainer, maps, name):
    map_name, seed, episodes = SCENARIOS[name]
    m = trainer.QModel()
    m.load_env(maps[map_name])
    helpers.train(m, episodes, seed)
    return m, map_name, seed, episodes


def export_fingerprint(trainer, model, episodes):
    """Формат і вміст CSV/JSON-експорту (як кнопки «Завантажити CSV/JSON»), без вікна."""
    fake_app = types.SimpleNamespace(m=model, episode=episodes, max_episodes=episodes)
    csv_text = trainer.App.csv_text(fake_app)
    data = trainer.App.json_data(fake_app)
    data.pop("created")  # час створення щоразу інший
    return {
        "csv_header": csv_text.split("\r\n")[0],
        "csv_lines": csv_text.count("\r\n") + 1,
        "csv_sha256": sha256(csv_text),
        "json_keys": sorted(data),
        "json_params_keys": sorted(data["params"]),
        "json_environment_keys": sorted(data["environment"]),
        "json_table_row_keys": sorted(data["table"][0]),
        "json_sha256": sha256(json.dumps(data, ensure_ascii=False, sort_keys=True)),
    }


def build_golden():
    trainer = helpers.load_trainer()
    maps = helpers.build_maps()
    golden = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "trainer_sha256": hashlib.sha256(helpers.TRAINER_PATH.read_bytes()).hexdigest(),
        "scenarios": {},
    }
    for name in SCENARIOS:
        m, map_name, seed, episodes = run_scenario(trainer, maps, name)
        golden["scenarios"][name] = {
            "map": map_name, "seed": seed, "episodes": episodes,
            "params": {"alpha": m.alpha, "gamma": m.gamma, "epsilon": m.epsilon},
            "snapshot": helpers.snapshot(m),
        }
        if name == EXPORT_SCENARIO:
            golden["export"] = export_fingerprint(trainer, m, episodes)
    return golden


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    golden = build_golden()
    out = helpers.TESTS_DIR / "golden_tabular.json"
    out.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Записано: {out}")
    for name, sc in golden["scenarios"].items():
        print(f"  {name}: {sc['snapshot']}")


if __name__ == "__main__":
    main()
