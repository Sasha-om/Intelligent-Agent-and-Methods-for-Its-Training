"""Юніт-тести qlearn.factory: створення агента за режимом і лінивий імпорт torch (етап 3)."""
import os
import subprocess
import sys

import pytest

import helpers
from qlearn.environment import GridEnv
from qlearn.factory import MODES, make_agent
from qlearn.tabular import TabularAgent

# Дочірній Python на Windows пише в кодуванні консолі (cp1251), а ми читаємо UTF-8 — задаємо явно (INS-010).
UTF8_ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def test_modes():
    assert MODES == ("tabular", "dqn")


def test_make_tabular(maps):
    agent = make_agent("tabular", GridEnv(maps["K1_default"]), epsilon=0.2)
    assert isinstance(agent, TabularAgent) and agent.epsilon == 0.2


def test_unknown_mode(maps):
    with pytest.raises(ValueError):
        make_agent("sarsa", GridEnv(maps["K1_default"]))


def test_lazy_import_without_torch():
    """Без torch: пакет імпортується, таблиця працює, DQN дає зрозумілу помилку українською (FR-N8)."""
    code = r"""
import sys
sys.modules["torch"] = None          # імітуємо відсутній PyTorch
import qlearn, qlearn.dqn, qlearn.factory as f
from qlearn.environment import GridEnv
print("available:", f.dqn_available())
f.make_agent("tabular", GridEnv()).run_episode()
try:
    f.make_agent("dqn", GridEnv())
except f.DQNUnavailable as e:
    print("error:", e)
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=helpers.APP_DIR, capture_output=True,
                         text=True, encoding="utf-8", check=True, env=UTF8_ENV)
    assert "available: False" in out.stdout
    assert "error:" in out.stdout and "PyTorch" in out.stdout and "Pytorch_test" in out.stdout
