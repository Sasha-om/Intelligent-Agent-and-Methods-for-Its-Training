"""Юніт-тести qlearn.ui_text — тексти інтерфейсу, що залежать від режиму (етап 6, FR-N6).

Головна регресія: для табличного режиму панель «Обчислення на поточному кроці» має бути
БАЙТ-У-БАЙТ такою самою, як до етапу 6 (NFR-1).
"""
import pytest

from qlearn import ui_text
from qlearn.environment import GridEnv

ENV = GridEnv()
STEP = {"frm": 0, "to": 7, "step_no": 3, "dir": "ВНИЗ", "reward": -1, "cost": 1, "goal": False,
        "old_q": 0.0, "new_q": -0.35}


def legacy_calc_text(t, cell_of):
    """Точна копія форматування з q_learning_trainer.App.render до етапу 6 (еталон)."""
    def signed(v, digits):
        return ("+" if v >= 0 else "") + f"{v:.{digits}f}"
    (fx, fy), (tx, ty) = cell_of(t["frm"]), cell_of(t["to"])
    bonus = f", бонус за фініш +{50}" if t["goal"] else ""
    return (f"крок {t['step_no']} · клітинка ({fx},{fy}) → дія: {t['dir']} → ({tx},{ty})\n"
            f"r = {signed(t['reward'], 2)}  (вартість переходу {t['cost']}{bonus})\n"
            f"Q(s,a) ← Q + α·[ r + γ·maxQ(s′,·) − Q ]"), f"{t['old_q']:.2f} → {t['new_q']:.2f}"


@pytest.mark.parametrize("goal", [False, True])
def test_calc_text_tabular_unchanged(goal):
    t = dict(STEP, goal=goal, reward=49 if goal else -1)
    assert ui_text.calc_text(t, ENV.cell_of, "tabular") == legacy_calc_text(t, ENV.cell_of)


def test_calc_text_dqn_with_loss():
    t = dict(STEP, target=1.7, loss=0.01234)
    main, delta = ui_text.calc_text(t, ENV.cell_of, "dqn")
    assert main.splitlines()[0] == "крок 3 · клітинка (0,0) → дія: ВНИЗ → (0,1)"
    assert "r = -1.00" in main
    assert "y = r + γ·max Q_target(s′,·) = 1.70" in main
    assert "втрата (Huber) = 0.0123" in main
    assert delta == "0.00 → -0.35"


def test_calc_text_dqn_before_training():
    """Поки в буфері < warmup переходів, мережа ще не навчається — пояснюємо це."""
    t = dict(STEP, target=-0.1, loss=None)
    main, _ = ui_text.calc_text(t, ENV.cell_of, "dqn")
    assert "навчання мережі почнеться" in main


def test_ui_texts_by_mode():
    tab, dqn = ui_text.ui_texts("tabular"), ui_text.ui_texts("dqn")
    assert tab["subtitle"] == ("Табличне Q-навчання методом проб і помилок · шлях не заданий "
                               "заздалегідь — агент прокладає його сам")
    assert tab["alpha_label"] == "Швидкість навчання, α"
    assert tab["q_title"] == "ТАБЛИЦЯ Q-ЗНАЧЕНЬ Q(S, A)"
    assert "нейромереж" in dqn["subtitle"].lower() and "lr" in dqn["alpha_label"]
    assert "НЕЙРОМЕРЕЖ" in dqn["q_title"]
    assert ui_text.MODE_LABELS == {"tabular": "Таблиця", "dqn": "Нейромережа"}


def test_mode_hint(monkeypatch):
    assert ui_text.mode_hint("tabular", 7, 6, dqn_ok=True) == ""
    assert "PyTorch" in ui_text.mode_hint("tabular", 7, 6, dqn_ok=False)
    assert ui_text.mode_hint("dqn", 7, 6, dqn_ok=True) == ""
    big = ui_text.mode_hint("dqn", 40, 30, dqn_ok=True)
    assert "повільно" in big and "γ" in big      # INS-011: на великих сітках з γ=0.9 агент «не бачить» фінішу


def test_unknown_mode():
    with pytest.raises(ValueError):
        ui_text.ui_texts("sarsa")
