"""Юніт-тести qlearn.ui_params — розбір полів інтерфейсу (етап 7, FR-R4, FR-N3).

Поля tkinter повертають рядки. Тут вони перетворюються на числа з перевіркою меж.
Помилки — зрозумілими реченнями українською, щоб їх можна було показати під полями.
"""
import pytest

from qlearn import ui_params
from qlearn.dqn import DEFAULT_PARAMS

GOOD = {"lr": "0.001", "hidden": "64", "batch": "32", "buffer": "5000", "target_sync": "100",
        "features": "compact"}


def test_parse_dqn_params_good():
    params, errors = ui_params.parse_dqn_params(GOOD, cells=42)
    assert errors == []
    assert params == {"lr": 0.001, "hidden": 64, "batch": 32, "buffer": 5000, "target_sync": 100,
                      "features": "compact"}


def test_parse_accepts_decimal_comma_and_spaces():
    params, errors = ui_params.parse_dqn_params(dict(GOOD, lr=" 0,0005 ", buffer="10 000"), cells=42)
    assert errors == [] and params["lr"] == 0.0005 and params["buffer"] == 10000


def test_defaults_round_trip():
    """Типові значення DQN → рядки полів → назад ті самі числа."""
    fields = ui_params.dqn_fields(DEFAULT_PARAMS)
    params, errors = ui_params.parse_dqn_params(fields, cells=42)
    assert errors == []
    assert params == {k: DEFAULT_PARAMS[k] for k in ui_params.DQN_KEYS}


@pytest.mark.parametrize("key, value, words", [
    ("lr", "0", "lr"),
    ("lr", "0.5", "lr"),
    ("lr", "abc", "lr"),
    ("hidden", "4", "нейронів"),
    ("hidden", "1000", "нейронів"),
    ("hidden", "64.5", "нейронів"),
    ("batch", "4", "батч"),
    ("batch", "300", "батч"),
    ("buffer", "16", "буфер"),          # менший за батч 32
    ("target_sync", "0", "синхронізац"),
    ("features", "pixels", "вхід"),
])
def test_parse_dqn_params_bad(key, value, words):
    params, errors = ui_params.parse_dqn_params(dict(GOOD, **{key: value}), cells=42)
    assert params is None and len(errors) == 1
    assert words in errors[0].lower()


def test_onehot_only_for_small_grids():
    _, ok = ui_params.parse_dqn_params(dict(GOOD, features="onehot"), cells=900)
    params, errors = ui_params.parse_dqn_params(dict(GOOD, features="onehot"), cells=901)
    assert ok == [] and params is None and "compact" in errors[0]


def test_several_errors_reported_together():
    _, errors = ui_params.parse_dqn_params(dict(GOOD, lr="x", hidden="1"), cells=42)
    assert len(errors) == 2


def test_parse_seed():
    assert ui_params.parse_seed("") is None              # порожнє поле — як досі, без фіксації
    assert ui_params.parse_seed("   ") is None
    assert ui_params.parse_seed("42") == 42
    assert ui_params.parse_seed(" 7 ") == 7
    for bad in ("-1", "1.5", "abc", str(2 ** 32)):
        with pytest.raises(ValueError, match="Seed"):
            ui_params.parse_seed(bad)


def test_autosave_decision():
    assert ui_params.autosave_decision("done", True)
    assert not ui_params.autosave_decision("done", False)
    for status in ("paused", "running"):
        assert not ui_params.autosave_decision(status, True)
