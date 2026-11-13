"""Налаштування pytest для тренажера.

- Маркер рівня ставиться автоматично за префіксом файлу: test_* → unit, smoke_* → smoke, regression_* → regression.
- Тести з маркером slow пропускаються, якщо не передано --runslow.
- Тести з маркером gui пропускаються, якщо немає tkinter або дисплея.
"""
import random

import pytest

import helpers


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False,
                     help="запустити й повільні тести (маркер slow)")


def _gui_available():
    try:
        import tkinter
        root = tkinter.Tk()
        root.destroy()
        return True
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    gui_ok = None
    run_slow = config.getoption("--runslow")
    for item in items:
        name = item.path.name
        if name.startswith("test_"):
            item.add_marker(pytest.mark.unit)
        elif name.startswith("smoke_"):
            item.add_marker(pytest.mark.smoke)
        elif name.startswith("regression_"):
            item.add_marker(pytest.mark.regression)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(pytest.mark.skip(reason="повільний тест: запустіть з --runslow"))
        if "gui" in item.keywords:
            if gui_ok is None:
                gui_ok = _gui_available()
            if not gui_ok:
                item.add_marker(pytest.mark.skip(reason="немає tkinter або дисплея"))


@pytest.fixture(scope="session")
def trainer():
    """Модуль q_learning_trainer.py (без вікна)."""
    return helpers.load_trainer()


@pytest.fixture(scope="session")
def maps():
    """Словник тестових карт (див. helpers.build_maps)."""
    return helpers.build_maps()


@pytest.fixture
def make_model(trainer):
    """Фабрика моделей: make_model(env=None, epsilon=None) → новий QModel."""
    def _make(env=None, epsilon=None):
        m = trainer.QModel()
        if env is not None:
            m.load_env(env)
        if epsilon is not None:
            m.epsilon = epsilon
        return m
    return _make


@pytest.fixture(autouse=True)
def _restore_random_state():
    """Кожен тест не впливає на генератор випадкових чисел інших тестів."""
    state = random.getstate()
    yield
    random.setstate(state)
