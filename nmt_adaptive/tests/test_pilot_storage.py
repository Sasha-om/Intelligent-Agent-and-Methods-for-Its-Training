"""Тести бази пілоту-порівняння (app/pilot_storage.py) і оновлення старих баз."""
import random
import sqlite3

import pytest

import config
from app import pilot_storage as ps
from app import storage

FUTURE = "9999-12-31"


@pytest.fixture
def conn(tmp_path):
    """Тимчасова база."""
    connection = storage.connect(tmp_path / "pilot.sqlite")
    yield connection
    connection.close()


def _student(conn, rng) -> str:
    """Учень зі згодою."""
    code = storage.new_student_code(conn, rng)
    storage.register_consent(conn, code, True)
    return code


def _lesson(conn, code, arm, pre, post, finish=True) -> int:
    """Сесія пілоту: вхідний бал, режим, (вихідний бал і завершення)."""
    sid = storage.start_session(conn, code, None)
    ps.save_score(conn, sid, "pre", pre)
    ps.choose_arm(conn, sid, arm, 1)
    if post is not None:
        ps.save_score(conn, sid, "post", post)
    if finish:
        storage.end_session(conn, sid, dropped=post is None)
    return sid


def test_old_database_is_upgraded(tmp_path) -> None:
    """База з етапу 9 (без нових полів) відкривається, отримує поля, старі дані цілі."""
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript("CREATE TABLE students (code TEXT PRIMARY KEY, created_at TEXT NOT NULL, "
                      "consent_at TEXT NOT NULL, guardian_consent INTEGER NOT NULL);"
                      "INSERT INTO students VALUES ('ABCDEF', 'x', 'x', 1);")
    old.close()
    conn = storage.connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(students)")}
    assert "form_order" in columns and storage.student_exists(conn, "ABCDEF")
    assert "arm" in {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    conn.close()


def test_form_order_balanced_and_fixed(conn) -> None:
    """Порядок форм: порівну AB/BA; для того самого учня не змінюється."""
    rng = random.Random(0)
    codes = [_student(conn, rng) for _ in range(40)]
    orders = [ps.form_order(conn, code, rng) for code in codes]
    assert orders.count("AB") == orders.count("BA") == 20
    assert ps.form_order(conn, codes[0], rng) == orders[0]
    with pytest.raises(KeyError):
        ps.form_order(conn, "ZZZZZZ", rng)


def test_assign_arm_balanced(conn) -> None:
    """Режим «навмання, порівну»: 30 сесій → по 10 у кожному режимі."""
    rng = random.Random(0)
    code = _student(conn, rng)
    for _ in range(30):
        sid = storage.start_session(conn, code, None)
        ps.choose_arm(conn, sid, ps.assign_arm(conn, rng), None)
    counts = dict(conn.execute("SELECT arm, COUNT(*) FROM sessions GROUP BY arm").fetchall())
    assert counts == {arm: 10 for arm in config.PILOT_ARMS}


def test_choose_arm_and_phase_validation(conn) -> None:
    """Невідомий режим чи етап — помилка; обраний режим і позиція картки записуються."""
    sid = storage.start_session(conn, _student(conn, random.Random(0)), None)
    with pytest.raises(ValueError):
        ps.choose_arm(conn, sid, "magic", 1)
    with pytest.raises(ValueError):
        ps.set_phase(conn, sid, "lunch")
    ps.choose_arm(conn, sid, "cat", 3)
    row = conn.execute("SELECT arm, card_position, phase FROM sessions WHERE id = ?", (sid,)).fetchone()
    assert tuple(row) == ("cat", 3, "train")


def test_test_answers_exported_and_deleted(conn) -> None:
    """Відповіді тестів — в експорті учня і зникають при видаленні."""
    code = _student(conn, random.Random(0))
    sid = storage.start_session(conn, code, None)
    record = {"phase": "pre", "form": "A", "slot": 1, "item_id": "testA1", "subtopic": "domain",
              "difficulty": -1.0, "chosen": 2, "correct": True, "time_sec": 30.0}
    ps.save_test_answer(conn, sid, record)
    exported = storage.student_data(conn, code)["sessions"][0]["test_answers"][0]
    assert exported["item_id"] == "testA1" and exported["correct"] is True
    assert storage.delete_student(conn, code) == 3      # відповідь тесту + сесія + учень


def test_arm_aggregates_hide_small_groups(conn) -> None:
    """Режим з 2 учнями — лише «обрали: 2»; режим з 3 — числа; кодів у результаті немає."""
    rng = random.Random(0)
    codes = [_student(conn, rng) for _ in range(5)]
    for code, pre, post in zip(codes[:3], (2, 3, 4), (4, 4, 5)):
        _lesson(conn, code, "q_agent", pre, post)
    for code in codes[3:]:
        _lesson(conn, code, "cat", 3, 3)
    agg = ps.arm_aggregates(conn, before=FUTURE)
    q = agg["arms"]["q_agent"]
    assert q["students"] == 3 and q["pre"] == pytest.approx(3) and q["post"] == pytest.approx(13 / 3)
    assert q["pre_done"] == pytest.approx(3)
    assert q["gain"] == pytest.approx(4 / 3) and q["gain_low"] <= q["gain"] <= q["gain_high"]
    assert agg["arms"]["cat"] == {"chosen": 0, "too_few": True}       # 2 < 3: навіть у пачку не потрапили
    assert agg["arms"]["fixed"] == {"chosen": 0, "too_few": True}
    assert agg["compare"] == [] and not any(code in str(agg) for code in codes)


def test_results_hidden_when_few_finished(conn) -> None:
    """3 учні почали, лише 1 завершив вихідний тест → результати сховані (інакше це бали однієї людини)."""
    rng = random.Random(0)
    _lesson(conn, _student(conn, rng), "q_agent", 2, 6)
    for _ in range(2):
        _lesson(conn, _student(conn, rng), "q_agent", 3, None)
    q = ps.arm_aggregates(conn, before=FUTURE)["arms"]["q_agent"]
    assert q["results_hidden"] is True and q["students"] == 3 and q["dropped"] == pytest.approx(2 / 3)
    assert all(q[field] is None for field in ps.RESULT_FIELDS)


def test_arm_aggregates_compare_and_dropout(conn) -> None:
    """Два режими з ≥ 3 завершеними → порівняння Q − CAT (α з поправкою); відмови рахуються."""
    rng = random.Random(0)
    for pre, post in ((2, 5), (3, 5), (4, 6), (3, 5), (1, None), (2, None)):
        _lesson(conn, _student(conn, rng), "q_agent", pre, post)
    for pre, post in ((2, 3), (3, 3), (4, 4)):
        _lesson(conn, _student(conn, rng), "cat", pre, post)
    agg = ps.arm_aggregates(conn, before=FUTURE)
    q = agg["arms"]["q_agent"]
    assert q["dropped"] == pytest.approx(2 / 6) and q["finished"] == 4
    assert q["pre"] == pytest.approx(2.5) and q["pre_done"] == pytest.approx(3)
    comparison = agg["compare"][0]
    assert comparison["other"] == "cat" and comparison["diff"] == pytest.approx(2.25 - 1 / 3)
    assert comparison["adjusted"]["effect"] > 0 and "significant" in comparison["adjusted"]


def test_table_changes_only_in_batches(conn) -> None:
    """Таблиця вчителя оновлюється пачками по 3: 4 завершені → видно 3; 6 → видно 6."""
    rng = random.Random(0)
    for _ in range(4):
        _lesson(conn, _student(conn, rng), "fixed", 3, 4)
    assert ps.arm_aggregates(conn, before=FUTURE)["arms"]["fixed"]["chosen"] == 3
    for _ in range(2):
        _lesson(conn, _student(conn, rng), "fixed", 3, 4)
    assert ps.arm_aggregates(conn, before=FUTURE)["arms"]["fixed"]["chosen"] == 6


def test_only_first_session_of_student(conn) -> None:
    """У таблиці вчителя — лише перша сесія кожного учня (повторні мають ті самі тести)."""
    rng = random.Random(0)
    codes = [_student(conn, rng) for _ in range(3)]
    for code in codes:
        _lesson(conn, code, "cat", 2, 3)
    for code in codes:
        _lesson(conn, code, "q_agent", 4, 6)          # другі сесії тих самих учнів
    agg = ps.arm_aggregates(conn, before=FUTURE)
    assert agg["arms"]["cat"]["chosen"] == 3 and agg["arms"]["q_agent"] == {"chosen": 0, "too_few": True}


def test_choice_counts_live(conn) -> None:
    """Розподіл вибору наживо: усі сьогоднішні сесії з режимом, навіть незавершені (лише кількості)."""
    rng = random.Random(0)
    _lesson(conn, _student(conn, rng), "cat", 2, None, finish=False)
    _lesson(conn, _student(conn, rng), "cat", 2, 3)
    assert ps.choice_counts(conn) == {"q_agent": 0, "cat": 2, "fixed": 0}


def test_unfinished_today_hidden(conn) -> None:
    """Сьогоднішня незавершена сесія пілоту не потрапляє в порівняння."""
    rng = random.Random(0)
    for _ in range(3):
        _lesson(conn, _student(conn, rng), "fixed", 3, None, finish=False)
    assert ps.arm_aggregates(conn)["arms"]["fixed"] == {"chosen": 0, "too_few": True}
