"""Тексти інтерфейсу, що залежать від режиму («Таблиця» / «Нейромережа») — чисті функції без tkinter.

Винесені з q_learning_trainer.App, щоб їх можна було перевірити тестами (tests/test_ui_text.py).
Для табличного режиму панель «Обчислення на поточному кроці» — байт-у-байт як до етапу 6 (NFR-1).
"""
from .environment import GOAL_BONUS

MODE_LABELS = {"tabular": "Таблиця", "dqn": "Нейромережа"}
BIG_GRID_CELLS = 900   # понад 30×30 — DQN повільний і (з γ=0.9) майже «не бачить» далекого фінішу (INS-011)

_TEXTS = {
    "tabular": {
        "subtitle": "Табличне Q-навчання методом проб і помилок · шлях не заданий заздалегідь — "
                    "агент прокладає його сам",
        "alpha_label": "Швидкість навчання, α",
        "q_title": "ТАБЛИЦЯ Q-ЗНАЧЕНЬ Q(S, A)",
    },
    "dqn": {
        "subtitle": "Глибоке Q-навчання (DQN): нейромережа оцінює Q замість таблиці · шлях не заданий "
                    "заздалегідь — агент прокладає його сам",
        "alpha_label": "Швидкість навчання мережі, lr",
        "q_title": "Q-ЗНАЧЕННЯ Q(S, A) — ПРОГНОЗ НЕЙРОМЕРЕЖІ",
    },
}


def ui_texts(mode):
    if mode not in _TEXTS:
        raise ValueError(f"невідомий режим: {mode!r}")
    return dict(_TEXTS[mode])


def signed(v, digits):
    return ("+" if v >= 0 else "") + f"{v:.{digits}f}"


def calc_text(t, cell_of, mode):
    """(основний текст, рядок «було → стало») для панелі «Обчислення на поточному кроці»."""
    (fx, fy), (tx, ty) = cell_of(t["frm"]), cell_of(t["to"])
    bonus = f", бонус за фініш +{GOAL_BONUS}" if t["goal"] else ""
    head = (f"крок {t['step_no']} · клітинка ({fx},{fy}) → дія: {t['dir']} → ({tx},{ty})\n"
            f"r = {signed(t['reward'], 2)}  (вартість переходу {t['cost']}{bonus})\n")
    if mode == "tabular":
        main = head + "Q(s,a) ← Q + α·[ r + γ·maxQ(s′,·) − Q ]"
    else:
        loss = t.get("loss")
        tail = (f"втрата (Huber) = {loss:.4f}" if loss is not None else
                "навчання мережі почнеться, коли в буфері набереться досить переходів")
        main = head + f"y = r + γ·max Q_target(s′,·) = {t['target']:.2f}\n" + tail
    return main, f"{t['old_q']:.2f} → {t['new_q']:.2f}"


def mode_hint(mode, cols, rows, dqn_ok):
    """Підказка під перемикачем режиму ("" — підказка не потрібна)."""
    if not dqn_ok:
        return ("Режим «Нейромережа» недоступний: потрібен PyTorch — запустіть тренажер через "
                "Pytorch_test\\.venv\\Scripts\\python.exe")
    if mode == "dqn" and cols * rows > BIG_GRID_CELLS:
        return ("Велика сітка: нейромережа навчатиметься повільно, а з γ = 0.9 агент майже не «бачить» "
                "далекого фінішу — спробуйте γ = 0.99")
    return ""
