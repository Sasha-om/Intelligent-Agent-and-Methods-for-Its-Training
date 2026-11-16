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


def test_fast_mode_stays_responsive_with_dqn(app):
    """«Швидке навчання» в DQN не «заморожує» вікно (NFR-2): навчання йде порціями ~25 мс.

    Поодинокі паузи 0.1–0.5 с можливі (потоки PyTorch конкурують за процесор, INS-013),
    тому перевіряємо: 95 % оновлень коротші за 100 мс і жодне не довше за 1 с.
    """
    pytest.importorskip("torch")
    app.set_mode("dqn")
    app.start_fast()
    times = []
    end = time.perf_counter() + 1.5
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        app.root.update()
        times.append(time.perf_counter() - t0)
    app.pause_run()
    times.sort()
    p95, worst = times[int(len(times) * 0.95)], times[-1]
    assert p95 < 0.1, f"95-й перцентиль оновлення вікна {p95 * 1000:.0f} мс"
    assert worst < 1.0, f"вікно не відповідало {worst:.2f} с"
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
