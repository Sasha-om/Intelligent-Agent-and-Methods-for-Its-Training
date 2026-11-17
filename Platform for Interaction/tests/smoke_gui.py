"""GUI smoke-тест тренажера (етап 6): справжнє вікно tkinter (приховане), обидва режими.

Маркер gui: пропускається без tkinter/дисплея. На ПК користувача — запускається;
у sandbox — через віртуальний дисплей Xvfb. Помилки всередині обробників tkinter не «вилітають» самі,
тому їх збираємо через report_callback_exception і перевіряємо, що їх немає.
"""
import random
import time

import pytest

import helpers

pytestmark = pytest.mark.gui


@pytest.fixture
def app(trainer):
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    errors = []
    root.report_callback_exception = lambda exc, val, tb: errors.append(val)
    random.seed(0)
    application = trainer.App(root)
    application.errors = errors
    yield application
    application._cancel()
    root.destroy()


def pump(app, seconds):
    """Дати tkinter попрацювати (обробити after-виклики) заданий час."""
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        app.root.update()
        time.sleep(0.005)


def test_tabular_basic_flow(app):
    for _ in range(30):
        app.step()
    assert app.last_transition is not None
    app.start_fast()
    pump(app, 0.5)
    app.pause_run()
    assert app.episode > 0 and app.m.mode == "tabular"
    app.toggle_q()
    app.render_q_table(force=True)
    assert app.errors == []


def test_mode_switch_roundtrip(app):
    pytest.importorskip("torch")
    for _ in range(10):
        app.step()
    app.set_mode("dqn")
    assert app.m.mode == "dqn" and app.episode == 0 and app.last_transition is None   # FR-N2: скидання
    assert app.mode_btns["dqn"].active and not app.mode_btns["tabular"].active
    for _ in range(20):
        app.step()
    assert "y = r" in app.calc_label.cget("text")
    app.start_fast()
    pump(app, 1.0)
    app.pause_run()
    assert app.episode >= 0 and app.errors == []
    data = app.json_data()
    assert data["mode"] == "dqn"
    app.set_mode("tabular")
    assert app.m.mode == "tabular" and app.episode == 0
    assert "Q(s,a) ← Q + α" not in app.calc_label.cget("text")      # панель скинута
    assert app.errors == []


def idle_gaps(app, seconds):
    """Як часто вікно встигає перемалюватися: проміжки між «вільними» моментами tkinter.

    tkinter перемальовує вікно в idle-обробниках, тобто коли немає інших подій. Проба кожні 10 мс
    ставить свій idle-обробник і записує час, коли він спрацював. Великий проміжок між записами
    означає, що стільки часу вікно не могло перемалюватися («зависло»).
    Тривалість одного root.update() для цього НЕ годиться: він може виконати поспіль багато
    порцій навчання, а між ними вікно все одно перемальовується (INS-013).
    Повертає (95-й перцентиль, найбільший проміжок) у секундах.
    """
    stamps = []
    alive = [True]

    def probe():
        if alive[0]:
            app.root.after_idle(lambda: stamps.append(time.perf_counter()))
            app.root.after(10, probe)

    probe()
    pump(app, seconds)
    alive[0] = False
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]))
    return gaps[int(len(gaps) * 0.95)], gaps[-1]


def test_fast_mode_stays_responsive_with_dqn(app):
    """«Швидке навчання» в DQN не «заморожує» вікно (NFR-2): навчання йде порціями ~25 мс,
    і між порціями вікно перемальовується. 95 % проміжків < 100 мс, найдовший < 0,5 с."""
    pytest.importorskip("torch")
    app.set_mode("dqn")
    app.start_fast()
    p95, worst = idle_gaps(app, 1.5)
    app.pause_run()
    assert app.episode > 0, "навчання не йшло"
    assert p95 < 0.1, f"95-й перцентиль проміжку між перемальовуваннями {p95 * 1000:.0f} мс"
    assert worst < 0.5, f"вікно не перемальовувалося {worst:.2f} с"
    assert app.errors == []


def test_editor_changes_in_dqn_mode(app):
    pytest.importorskip("torch")
    app.set_mode("dqn")
    app.random_env()
    app.default_env()
    app.on_dims() if hasattr(app, "dims_vars") else None
    for _ in range(5):
        app.step()
    assert len(app.m.q) == app.m.cols * app.m.rows * 4 and app.errors == []


def test_dqn_button_disabled_without_torch(trainer, monkeypatch):
    import tkinter as tk
    monkeypatch.setattr(trainer, "dqn_available", lambda: False)
    root = tk.Tk()
    root.withdraw()
    try:
        app = trainer.App(root)
        assert not app.mode_btns["dqn"].enabled
        assert "PyTorch" in app.mode_hint.cget("text")
        app.set_mode("dqn")                      # натискання вимкненої кнопки нічого не робить
        assert app.m.mode == "tabular"
    finally:
        root.destroy()


def test_helpers_unaffected():
    assert helpers.TRAINER_PATH.exists()


def test_alpha_slider_disabled_in_dqn(app):
    pytest.importorskip("torch")
    title, val, scale, _fmt = app.slider_parts["Швидкість навчання, α"]
    assert not scale.instate(["disabled"]) and title.cget("text") == "Швидкість навчання, α"
    app.set_mode("dqn")
    assert scale.instate(["disabled"]) and "lr" in title.cget("text") and val.cget("text") == "0.001"
    app.set_mode("tabular")
    assert not scale.instate(["disabled"]) and val.cget("text") == "0.35"


def test_fast_mode_pause_mid_episode_keeps_position(app):
    """DQN: пауза «Швидкого навчання» посеред епізоду — вікно показує те місце й ту кількість кроків,
    де зупинився stepper (епізод продовжиться з цього місця)."""
    pytest.importorskip("torch")
    app.set_mode("dqn")
    for _ in range(20):                       # шукаємо паузу посеред епізоду (у DQN епізоди довгі)
        app.start_fast()
        pump(app, 0.1)
        app.pause_run()
        if app.stepper.steps > 0:
            break
    assert app.stepper.steps > 0, "не вдалося зупинитись посеред епізоду"
    assert (app.state, app.steps, app.ep_reward) == (app.stepper.state, app.stepper.steps, app.stepper.ep_reward)
    assert app.errors == []


def test_moving_items_have_whole_pixel_geometry(app):
    """Синя рамка під курсором, ореол і агент мають цілі координати й товщину ліній.
    З дробовими (масштаб s ≈ 1.63) Windows лишає на полотні «сині смужки» після руху рамки (INS-015)."""
    cv = app.cv
    assert app.s != int(app.s)                       # масштаб справді дробовий — перевірка має сенс
    for k in range(12):
        cv.event_generate("<Motion>", x=17 + k * 37, y=23 + k * 29)
        app.root.update()
        app.step()
        for item in (app.hover_item, app.halo, app.agent_item):
            assert all(v == int(v) for v in cv.coords(item)), cv.coords(item)
            assert float(cv.itemcget(item, "width")).is_integer()
    assert app.errors == []


def test_window_redraws_between_fast_portions(app):
    """Між двома порціями «Швидкого навчання» tkinter обов'язково встигає обробити idle-події
    (перемальовування вікна). Інакше вікно «смикається» (INS-015)."""
    pytest.importorskip("torch")
    app.set_mode("dqn")
    st = app.stepper
    orig = st.train_for
    flag = {"idle": True}
    missed = []

    def train_for(*a, **kw):
        missed.append(not flag["idle"])
        res = orig(*a, **kw)
        flag["idle"] = False
        app.root.after_idle(lambda: flag.update(idle=True))
        return res

    st.train_for = train_for
    app.start_fast()
    pump(app, 1.0)
    app.pause_run()
    assert len(missed) > 5, "порцій навчання замало"
    assert not any(missed), f"без перемальовування між порціями: {sum(missed)} з {len(missed)}"
    assert app.errors == []
