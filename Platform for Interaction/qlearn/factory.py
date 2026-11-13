"""Єдина точка створення агента за назвою режиму. torch імпортується лише тоді, коли потрібен DQN."""
from .dqn import DQNUnavailable, torch_available
from .tabular import TabularAgent

MODES = ("tabular", "dqn")


def dqn_available():
    """Чи можна створити DQN-агента (чи встановлено PyTorch)."""
    return torch_available()


def make_agent(mode, env, **params):
    """mode: "tabular" або "dqn"; params — параметри агента (alpha/gamma/epsilon, для DQN — ще hidden, lr, …)."""
    if mode == "tabular":
        return TabularAgent(env, **params)
    if mode == "dqn":
        from .dqn import DQNAgent      # тут уже потрібен torch
        return DQNAgent(env, **params)
    raise ValueError(f"невідомий режим: {mode!r} (можливі: {', '.join(MODES)})")


__all__ = ["MODES", "DQNUnavailable", "dqn_available", "make_agent"]
