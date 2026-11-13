"""Регресія швидкості: табличне навчання після рефакторингу ≥ 85 % від швидкості ПОЧАТКОВОГО коду.

Як міряємо (уточнено після етапу 2, INS-007): порівнювати з числом, записаним раніше (golden_perf.json),
ненадійно — кожен замір триває кілька мілісекунд, і стан комп'ютера між днями інший (шум до ±20 %).
Тому тест бере початкову версію тренажера з історії git (коміт LEGACY_COMMIT) і міряє її
ПОЧЕРГОВО з поточною в тому самому процесі: шум діє на обидві однаково.

Маркер slow: запускається з --runslow. Без git або без історії — пропускається.
"""
import importlib.util
import random
import shutil
import statistics
import subprocess
import time

import pytest

import helpers

LEGACY_COMMIT = "0c36b8a"   # «chore: початковий стан тренажера» — до будь-яких змін
LEGACY_PATH = "Platform for Interaction/q_learning_trainer.py"
MIN_RATIO = 0.85
ROUNDS, EPISODES = 25, 600


def _load_legacy(tmp_path):
    if not shutil.which("git"):
        pytest.skip("git не знайдено")
    out = subprocess.run(["git", "show", f"{LEGACY_COMMIT}:{LEGACY_PATH}"], cwd=helpers.APP_DIR,
                         capture_output=True)
    if out.returncode != 0:
        pytest.skip(f"у git немає коміту {LEGACY_COMMIT} (не репозиторій проєкту?)")
    path = tmp_path / "legacy_trainer.py"
    path.write_bytes(out.stdout)
    helpers.load_trainer()          # ставить заглушку tkinter, якщо його немає
    spec = importlib.util.spec_from_file_location("legacy_trainer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rate(model_cls, env):
    m = model_cls()
    m.load_env(env)
    random.seed(0)
    t0 = time.perf_counter()
    for _ in range(EPISODES):
        m.run_episode_silent()
    return EPISODES / (time.perf_counter() - t0)


@pytest.mark.slow
@pytest.mark.parametrize("map_name", ["K1_default", "R2_15x15"])
def test_speed_vs_legacy(tmp_path, maps, map_name):
    legacy = _load_legacy(tmp_path)
    current = helpers.load_trainer()
    env = maps[map_name]
    old_rates, new_rates = [], []
    for _ in range(ROUNDS):
        old_rates.append(_rate(legacy.QModel, env))
        new_rates.append(_rate(current.QModel, env))
    ratio = statistics.median(new_rates) / statistics.median(old_rates)
    print(f"{map_name}: поточний / початковий = {ratio:.1%}")
    assert ratio >= MIN_RATIO, f"{map_name}: швидкість {ratio:.0%} від початкового коду (межа {MIN_RATIO:.0%})"
