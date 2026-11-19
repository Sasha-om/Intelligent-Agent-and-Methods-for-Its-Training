"""Розбір полів інтерфейсу тренажера (етап 7): параметри нейромережі, seed, автозбереження.

Поля tkinter дають рядки. Тут вони стають числами з перевіркою меж, а помилки — реченнями
українською, які можна показати користувачу. Без tkinter — тому легко тестується
(tests/test_ui_params.py).
"""
from .features import ONEHOT_MAX_CELLS

# Порядок полів у панелі «Параметри нейромережі»
DQN_KEYS = ("lr", "hidden", "batch", "buffer", "target_sync", "features")

# Підписи полів і короткі підказки з допустимими значеннями
DQN_LABELS = {
    "lr": ("Швидкість навчання мережі, lr", "0.00001 … 0.1, типово 0.001"),
    "hidden": ("Нейронів у шарі", "8 … 512, типово 64"),
    "batch": ("Батч", "8 … 256, типово 32"),
    "buffer": ("Буфер досвіду", "≥ батча, типово 5000"),
    "target_sync": ("Синхронізація цільової мережі, кроків", "≥ 1, типово 100"),
    "features": ("Вхід мережі", "compact — 8 чисел; onehot — до 900 клітинок"),
}
FEATURE_KINDS = ("compact", "onehot")
SEED_MAX = 2 ** 32 - 1


def _number(text):
    """«0,001» / « 10 000 » → рядок, зрозумілий float()/int()."""
    return text.strip().replace(" ", "").replace(" ", "").replace(",", ".")


def _int(text, name, lo, hi):
    try:
        value = float(_number(text))
    except ValueError:
        return None, f"{name}: потрібне ціле число"
    if not value.is_integer():
        return None, f"{name}: потрібне ціле число"
    value = int(value)
    if not lo <= value <= hi:
        return None, f"{name}: від {lo} до {hi}"
    return value, None


def parse_dqn_params(fields, cells):
    """fields — словник рядків з полів панелі; cells — кількість клітинок поточної сітки.

    Повертає (params, errors): params — словник чисел для DQNAgent (або None, якщо є помилки),
    errors — список повідомлень українською (порожній, якщо все гаразд).
    """
    errors, params = [], {}

    try:
        lr = float(_number(fields["lr"]))
        if not 1e-5 <= lr <= 0.1:
            raise ValueError
        params["lr"] = lr
    except ValueError:
        errors.append("lr (швидкість навчання мережі): число від 0.00001 до 0.1, наприклад 0.001")

    for key, name, lo, hi in (("hidden", "Нейронів у шарі", 8, 512),
                              ("batch", "Батч", 8, 256),
                              ("target_sync", "Синхронізація цільової мережі", 1, 100_000)):
        value, err = _int(fields[key], name, lo, hi)
        if err:
            errors.append(err)
        else:
            params[key] = value

    batch = params.get("batch", 32)
    value, err = _int(fields["buffer"], "Буфер досвіду", batch, 1_000_000)
    if err:
        errors.append(err + (" (не менше за батч)" if "від" in err else ""))
    else:
        params["buffer"] = value

    kind = fields["features"].strip()
    if kind not in FEATURE_KINDS:
        errors.append(f"Вхід мережі: {' або '.join(FEATURE_KINDS)}")
    elif kind == "onehot" and cells > ONEHOT_MAX_CELLS:
        errors.append(f"Вхід мережі onehot — лише для сіток до {ONEHOT_MAX_CELLS} клітинок; "
                      f"тут {cells}, оберіть compact")
    else:
        params["features"] = kind

    return (None, errors) if errors else (params, [])


def dqn_fields(params):
    """Числові параметри DQN → рядки для полів панелі."""
    return {key: f"{params[key]:g}" if key == "lr" else str(params[key]) for key in DQN_KEYS}


def parse_seed(text):
    """Порожнє поле → None (seed не фіксується, як до етапу 7). Ціле 0…2³²−1 → число. Інакше ValueError."""
    text = text.strip()
    if not text:
        return None
    if not text.isdigit() or int(text) > SEED_MAX:
        raise ValueError(f"Seed — ціле число від 0 до {SEED_MAX} або порожнє поле")
    return int(text)


def autosave_decision(status, enabled):
    """Автозбереження — лише коли навчання завершилось («Завершено») і прапорець увімкнений."""
    return enabled and status == "done"
