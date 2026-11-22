"""Сторінки Streamlit без браузера (streamlit.testing.v1.AppTest): учень і вчитель."""
import random
from contextlib import closing

import pytest
from streamlit.testing.v1 import AppTest

import config
from app import storage

STUDENT = str(config.BASE_DIR / "app" / "app_student.py")
TEACHER = str(config.BASE_DIR / "app" / "app_teacher.py")


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Сторінки пишуть у тимчасову базу, а не в data/students/."""
    path = tmp_path / "students" / "app.sqlite"
    monkeypatch.setattr(config, "DB_PATH", path)
    return path


def _query(db_path, sql: str) -> list:
    """Запит до тимчасової бази з закриттям з'єднання."""
    with closing(storage.connect(db_path)) as conn:
        return [tuple(r) for r in conn.execute(sql).fetchall()]


def _run(target: object) -> AppTest:
    """Запуск сторінки (або дії з віджетом); будь-яка помилка Python у сторінці — провал тесту."""
    at = target.run(timeout=30)
    assert not at.exception, at.exception
    return at


def _texts(at: AppTest) -> str:
    """Увесь видимий текст сторінки (markdown, повідомлення) одним рядком."""
    parts = [m.value for m in at.markdown] + [e.value for kind in ("success", "error", "info", "warning")
                                             for e in getattr(at, kind)]
    return "\n".join(str(p) for p in parts)


def _student_until_task(db_path) -> AppTest:
    """Згода → новий код → пропустити чек-ін → перше рішення агента (якщо перерва — «Продовжити»)."""
    at = _run(AppTest.from_file(STUDENT))
    _run(at.checkbox(key="consent_box").check())
    _run(at.button(key="consent_next").click())
    _run(at.button(key="get_code").click())
    _run(at.button(key="checkin_skip").click())
    while at.session_state.step == "break":
        _run(at.button(key="break_continue").click())
    return at


def test_consent_required(db_path) -> None:
    """Без галочки кнопка «Далі» неактивна, і в базі нічого немає."""
    at = _run(AppTest.from_file(STUDENT))
    assert at.button(key="consent_next").disabled
    assert not db_path.exists()            # до згоди не створюється навіть файл бази


def test_new_code_after_consent(db_path) -> None:
    """Після згоди «Отримати код» видає код, який є в базі, і показує його учню."""
    at = _run(AppTest.from_file(STUDENT))
    _run(at.checkbox(key="consent_box").check())
    _run(at.button(key="consent_next").click())
    _run(at.button(key="get_code").click())
    code = at.session_state.code
    assert _query(db_path, "SELECT code FROM students") == [(code,)] and code in _texts(at)


def test_unknown_code_rejected(db_path) -> None:
    """Неіснуючий код — повідомлення про помилку, далі не пускає."""
    at = _run(AppTest.from_file(STUDENT))
    _run(at.checkbox(key="consent_box").check())
    _run(at.button(key="consent_next").click())
    at.text_input(key="code_input").input("ZZZZZZ")
    _run(at.button(key="login").click())
    assert at.session_state.step == "code" and "Такого коду немає" in _texts(at)


def test_answer_shows_review_and_saves(db_path) -> None:
    """Відповідь → розбір на екрані, запис у базі; «Далі» → наступне рішення агента."""
    at = _student_until_task(db_path)
    assert at.session_state.step == "task"
    assert at.session_state.live.decision["reason"] in _texts(at)      # рішення агента видно учню
    item = at.session_state.live.item
    wrong = (item["answer"] + 1) % len(item["options"])
    _run(at.radio(key="answer_0").set_value(wrong))
    _run(at.button(key="answer_btn").click())
    assert at.session_state.step == "feedback" and "Типова помилка" in _texts(at)
    assert _query(db_path, "SELECT correct FROM answers") == [(0,)]
    _run(at.button(key="feedback_next").click())
    assert at.session_state.step in ("task", "break")


def test_stop_and_summary(db_path) -> None:
    """«Завершити зараз» → підсумок; сесія в базі позначена як достроково завершена."""
    at = _student_until_task(db_path)
    item = at.session_state.live.item
    _run(at.radio(key="answer_0").set_value(item["answer"]))
    _run(at.button(key="answer_btn").click())
    _run(at.button(key="feedback_next").click())
    _run(at.button(key="stop").click())
    assert at.session_state.step == "summary" and "1 з 1" in str([m.value for m in at.metric])
    row = _query(db_path, "SELECT dropped, items_done, ended_at FROM sessions")[0]
    assert row[0] == 1 and row[1] == 1 and row[2]


def test_delete_my_data(db_path) -> None:
    """«Видалити мої дані» (з підтвердженням) прибирає учня з бази і повертає на екран згоди."""
    at = _student_until_task(db_path)
    code = at.session_state.code
    assert at.sidebar.button(key="delete").disabled
    _run(at.sidebar.checkbox(key="delete_sure").check())
    _run(at.sidebar.button(key="delete").click())
    assert _query(db_path, "SELECT * FROM students") == []
    assert at.session_state.step == "consent" and "видалено" in _texts(at)


def _fill(db_path, students: int) -> None:
    """База з кількома учнями, у кожного — одна відповідь."""
    conn = storage.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    rng = random.Random(0)
    for _ in range(students):
        code = storage.new_student_code(conn, rng)
        storage.register_consent(conn, code, True)
        sid = storage.start_session(conn, code, None)
        storage.save_answer(conn, sid, {"n": 1, "item_id": "it001", "subtopic": "domain", "difficulty": 0.0,
                                        "params": {}, "chosen": 0, "correct": True, "time_sec": 30.0,
                                        "action": "same", "p_hat": 0.7, "theta_hat": {}, "state": 0,
                                        "fallback": False})
        storage.end_session(conn, sid, False, {"theta_gain": 0.1})
    conn.execute("UPDATE sessions SET started_at = '2026-01-01T10:00:00+02:00'")   # «учорашні» сесії
    conn.commit()
    conn.close()


def test_teacher_small_group(db_path) -> None:
    """2 учні — лише повідомлення «замало даних», жодних чисел."""
    _fill(db_path, 2)
    at = _run(AppTest.from_file(TEACHER))
    assert "Замало даних" in _texts(at) and len(at.metric) == 0
    assert "Це не оцінки учнів" in str([c.value for c in at.caption])


def test_teacher_aggregates_without_codes(db_path) -> None:
    """3 учні — агрегати є; кодів учнів на сторінці немає."""
    _fill(db_path, 3)
    at = _run(AppTest.from_file(TEACHER))
    assert [m.value for m in at.metric][:2] == ["3", "3"]
    codes = [r[0] for r in _query(db_path, "SELECT code FROM students")]
    page = _texts(at) + str([m.value for m in at.metric])
    assert not any(code in page for code in codes) and "Область визначення" in page


def test_full_session_closes_itself(db_path) -> None:
    """Повна сесія (20 завдань, перерви — «Продовжити»): після 20-го завдання сесія закрита в базі ще до «Підсумку»."""
    at = _student_until_task(db_path)
    for n in range(config.SESSION_ITEMS):
        while at.session_state.step == "break":
            _run(at.button(key="break_continue").click())
        _run(at.radio(key=f"answer_{n}").set_value(0))
        _run(at.button(key="answer_btn").click())
        if n < config.SESSION_ITEMS - 1:
            _run(at.button(key="feedback_next").click())
    row = _query(db_path, "SELECT dropped, items_done, ended_at FROM sessions")[0]
    assert row[0] == 0 and row[1] == config.SESSION_ITEMS and row[2]
    _run(at.button(key="to_summary").click())
    assert at.session_state.step == "summary"
    assert _query(db_path, "SELECT COUNT(*) FROM sessions") == [(1,)]


def test_teacher_sees_finished_today(db_path) -> None:
    """Сьогоднішні ЗАВЕРШЕНІ сесії вчитель бачить одразу; є адреса для учнів і кнопка «Оновити»."""
    _fill(db_path, 3)
    with closing(storage.connect(db_path)) as conn:
        conn.execute("UPDATE sessions SET started_at = ?", (storage.today() + "T10:00:00+02:00",))
        conn.commit()
    at = _run(AppTest.from_file(TEACHER))
    assert [m.value for m in at.metric][:2] == ["3", "3"]
    assert "http://" in _texts(at) and at.button(key="refresh")


def test_teacher_hides_unfinished_today(db_path) -> None:
    """Учень ще розв'язує (сесія не завершена) — його відповідей вчитель не бачить, лише «зараз проходять: 1»."""
    _fill(db_path, 3)
    with closing(storage.connect(db_path)) as conn:
        conn.execute("UPDATE sessions SET started_at = ?, ended_at = NULL", (storage.today() + "T10:00:00+02:00",))
        conn.commit()
    at = _run(AppTest.from_file(TEACHER))
    assert "Замало даних" in _texts(at) and len(at.metric) == 0 and "проходять тест: **3**" in _texts(at)


def test_login_with_existing_code(db_path) -> None:
    """Вхід за наявним кодом (маленькими літерами) → чек-ін того самого учня."""
    with closing(storage.connect(db_path)) as conn:
        storage.register_consent(conn, "ABCDEF", True)
    at = _run(AppTest.from_file(STUDENT))
    _run(at.checkbox(key="consent_box").check())
    _run(at.button(key="consent_next").click())
    _run(at.text_input(key="code_input").input("abcdef"))
    _run(at.button(key="login").click())
    assert at.session_state.step == "checkin" and at.session_state.code == "ABCDEF"


def test_logout_clears_everything(db_path) -> None:
    """«Вийти» посеред сесії: сесія закрита як дострокова, стан порожній, наступний учень бачить екран згоди."""
    at = _student_until_task(db_path)
    _run(at.sidebar.button(key="logout").click())
    assert at.session_state.step == "consent" and "code" not in at.session_state
    assert len(at.sidebar.button) == 0
    assert _query(db_path, "SELECT dropped, ended_at IS NOT NULL FROM sessions") == [(1, 1)]


def test_code_hidden_in_sidebar(db_path) -> None:
    """У бічній панелі код схований, доки учень не ввімкне «Показати мій код»."""
    at = _student_until_task(db_path)
    code = at.session_state.code
    sidebar = str([m.value for m in at.sidebar.markdown])
    assert code not in sidebar
    _run(at.sidebar.toggle(key="show_code").set_value(True))
    assert code in str([m.value for m in at.sidebar.markdown])
