"""Сторінки в режимі порівняння (етап 10): вхідний тест → вибір режиму → тренування → вихідний тест."""
from contextlib import closing

import pytest
from streamlit.testing.v1 import AppTest

import config
from app import storage

STUDENT = str(config.BASE_DIR / "app" / "app_student.py")
TEACHER = str(config.BASE_DIR / "app" / "app_teacher.py")
TOTAL = len(config.PILOT_TEST_SLOTS)


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Тимчасова база; режим порівняння увімкнено, учень обирає сам."""
    path = tmp_path / "students" / "pilot.sqlite"
    monkeypatch.setattr(config, "DB_PATH", path)
    monkeypatch.setattr(config, "PILOT_COMPARE", True)
    monkeypatch.setattr(config, "PILOT_ASSIGNMENT", "choice")
    return path


def _run(target: object) -> AppTest:
    """Запуск сторінки або дії; помилка Python на сторінці — провал тесту."""
    at = target.run(timeout=30)
    assert not at.exception, at.exception
    return at


def _query(db_path, sql: str) -> list:
    """Запит до тимчасової бази."""
    with closing(storage.connect(db_path)) as conn:
        return [tuple(r) for r in conn.execute(sql).fetchall()]


def _texts(at: AppTest) -> str:
    """Видимий текст сторінки."""
    parts = [m.value for m in at.markdown] + [e.value for kind in ("success", "error", "info", "warning")
                                             for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts) + str([m.value for m in at.metric]) + str([c.value for c in at.caption])


def _start(db_path) -> AppTest:
    """Згода → код → пропустити чек-ін → екран вхідного тесту."""
    at = _run(AppTest.from_file(STUDENT))
    _run(at.checkbox(key="consent_box").check())
    _run(at.button(key="consent_next").click())
    _run(at.button(key="get_code").click())
    return _run(at.button(key="checkin_skip").click())


def _exam(at: AppTest, phase: str, right: int) -> AppTest:
    """Розв'язує тест: перші right завдань правильно, решту — неправильно."""
    for k in range(TOTAL):
        item = at.session_state.test.current
        chosen = item["answer"] if k < right else (item["answer"] + 1) % 4
        _run(at.radio(key=f"exam_{phase}_{k}").set_value(chosen))
        at = _run(at.button(key="exam_btn").click())
    return at


def _train(at: AppTest) -> AppTest:
    """Усі навчальні завдання (перерви — «Продовжити»)."""
    for n in range(config.PILOT_TRAIN_ITEMS):
        while at.session_state.step == "break":
            at = _run(at.button(key="break_continue").click())
        _run(at.radio(key=f"answer_{n}").set_value(0))
        at = _run(at.button(key="answer_btn").click())
        if n < config.PILOT_TRAIN_ITEMS - 1:
            at = _run(at.button(key="feedback_next").click())
    return at


def test_pre_test_has_no_review(db_path) -> None:
    """Вхідний тест: «Розбору тут не буде», після відповіді — наступне завдання тесту, не розбір."""
    at = _start(db_path)
    assert at.session_state.step == "exam" and "Розбору тут не буде" in _texts(at)
    item = at.session_state.test.current
    _run(at.radio(key="exam_pre_0").set_value((item["answer"] + 1) % 4))
    at = _run(at.button(key="exam_btn").click())
    assert at.session_state.step == "exam" and "Типова помилка" not in _texts(at)
    assert _query(db_path, "SELECT phase, slot, correct FROM test_answers") == [("pre", 1, 0)]


@pytest.mark.parametrize("arm", config.PILOT_ARMS)
def test_full_lesson_each_arm(db_path, arm) -> None:
    """Повний урок у кожному режимі: бали в базі, режим і позиція картки, сесія завершена, підсумок «вхід → вихід»."""
    at = _exam(_start(db_path), "pre", right=2)
    assert at.session_state.step == "choose" and len(at.session_state.card_order) == 3
    at = _run(at.button(key=f"choose_{arm}").click())
    assert at.session_state.step in ("task", "break")
    at = _train(at)
    at = _run(at.button(key="to_post").click())
    at = _exam(at, "post", right=5)
    assert at.session_state.step == "summary" and "2 → 5 з 6" in _texts(at)
    row = _query(db_path, "SELECT arm, card_position, pre_score, post_score, phase, dropped, ended_at IS NOT NULL "
                          "FROM sessions")[0]
    position = at.session_state.card_order.index(arm) + 1
    assert row == (arm, position, 2, 5, "done", 0, 1)
    assert _query(db_path, "SELECT COUNT(*) FROM answers") == [(config.PILOT_TRAIN_ITEMS,)]
    assert _query(db_path, "SELECT phase, COUNT(*) FROM test_answers GROUP BY phase") == [("post", 6), ("pre", 6)]


def test_forms_follow_order(db_path) -> None:
    """Вхідний тест — перша форма з порядку учня, вихідний — друга; завдання тестів не повторюються в тренуванні."""
    at = _exam(_start(db_path), "pre", right=0)
    order = at.session_state.pilot_order
    at = _train(_run(at.button(key="choose_fixed").click()))
    at = _exam(_run(at.button(key="to_post").click()), "post", right=0)
    forms = _query(db_path, "SELECT phase, form FROM test_answers GROUP BY phase, form ORDER BY phase DESC")
    assert forms == [("pre", order[0]), ("post", order[1])]
    test_ids = {r[0] for r in _query(db_path, "SELECT item_id FROM test_answers")}
    train_ids = {r[0] for r in _query(db_path, "SELECT item_id FROM answers")}
    assert not test_ids & train_ids


def test_only_q_shows_agent_explanation(db_path) -> None:
    """CAT і «звичайний тест»: без «Як агент це вирішив?» і без перерв."""
    at = _run(_exam(_start(db_path), "pre", right=3).button(key="choose_cat").click())
    assert at.session_state.live.decision["state_view"] is None and len(at.expander) == 0
    assert "Наступне завдання: тема" in _texts(at)


def test_stop_during_post_test(db_path) -> None:
    """«Завершити зараз» на вихідному тесті → відмова, вихідного балу немає."""
    at = _train(_run(_exam(_start(db_path), "pre", right=3).button(key="choose_q_agent").click()))
    at = _run(at.button(key="to_post").click())
    at = _run(at.button(key="stop").click())
    assert at.session_state.step == "summary"
    assert _query(db_path, "SELECT pre_score, post_score, dropped FROM sessions") == [(3, None, 1)]


def test_stop_during_pre_test(db_path) -> None:
    """«Завершити зараз» ще на вхідному тесті — без помилок; сесія достроково завершена."""
    at = _run(_start(db_path).button(key="stop").click())
    assert at.session_state.step == "summary"
    assert _query(db_path, "SELECT dropped, ended_at IS NOT NULL FROM sessions") == [(1, 1)]


def test_random_assignment_skips_choice(db_path, monkeypatch) -> None:
    """PILOT_ASSIGNMENT = "random": екрана вибору немає, режим записано, позиції картки немає."""
    monkeypatch.setattr(config, "PILOT_ASSIGNMENT", "random")
    at = _exam(_start(db_path), "pre", right=1)
    assert at.session_state.step in ("task", "break")
    arm, position = _query(db_path, "SELECT arm, card_position FROM sessions")[0]
    assert arm in config.PILOT_ARMS and position is None


def _fill_pilot(db_path, plan: dict) -> None:
    """База з завершеними уроками: plan = {режим: [(вхід, вихід), ...]} («вчорашні» сесії)."""
    import random
    from app import pilot_storage as ps
    with closing(storage.connect(db_path)) as conn:
        rng = random.Random(0)
        for arm, lessons in plan.items():
            for pre, post in lessons:
                code = storage.new_student_code(conn, rng)
                storage.register_consent(conn, code, True)
                sid = storage.start_session(conn, code, None)
                ps.save_score(conn, sid, "pre", pre)
                ps.choose_arm(conn, sid, arm, 1)
                ps.save_score(conn, sid, "post", post)
                storage.end_session(conn, sid, False)
        conn.execute("UPDATE sessions SET started_at = '2026-01-01T10:00:00+02:00', "
                     "ended_at = '2026-01-01T10:30:00+02:00'")
        conn.commit()


def test_teacher_pilot_table(db_path) -> None:
    """Вчитель: таблиця режимів; режим з 2 учнями — «замало даних»; є рядок «Q-агент − CAT»; кодів немає."""
    _fill_pilot(db_path, {"q_agent": [(2, 4), (3, 5), (4, 5)], "cat": [(3, 3), (2, 3), (4, 4)],
                          "fixed": [(3, 4), (2, 2)]})
    at = _run(AppTest.from_file(TEACHER))
    table = at.dataframe[0].value
    assert list(table.columns) == [config.PILOT_ARM_NAMES_UA[a] for a in config.PILOT_ARMS]
    assert table.loc["Вхідний тест, середнє з 6", "Q-агент"] == "3,00"
    assert table.loc["Хвилин на сесію", "Q-агент"] == "30"
    assert table.loc["Учнів", "Звичайний тест"] == "замало даних"
    text = _texts(at)
    assert "Q-агент − Адаптивний тест (CAT)" in text and "Q-агент − Звичайний тест" not in text
    assert "Сьогодні обрали" in text
    codes = [r[0] for r in _query(db_path, "SELECT code FROM students")]
    assert not any(code in text + table.to_string() for code in codes)


def test_logout_during_pre_test_closes_session(db_path) -> None:
    """«Вийти» посеред вхідного тесту: сесія закрита як дострокова (не висить у «зараз проходять»)."""
    at = _start(db_path)
    at = _run(at.sidebar.button(key="logout").click())
    assert at.session_state.step == "consent"
    assert _query(db_path, "SELECT dropped, ended_at IS NOT NULL FROM sessions") == [(1, 1)]
