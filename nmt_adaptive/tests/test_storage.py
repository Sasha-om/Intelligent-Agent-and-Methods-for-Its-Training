"""Тести сховища SQLite: анонімні коди, згода, сесії, експорт, видалення, агрегати для вчителя."""
import json
import random
import sqlite3

import pytest

import config
from app import storage

FUTURE = "9999-12-31"   # «рахувати всі сесії, і сьогоднішні теж» (у застосунку сьогоднішні сховані)


@pytest.fixture
def conn(tmp_path):
    """Тимчасова база для кожного тесту (data/students/ не чіпаємо)."""
    connection = storage.connect(tmp_path / "students" / "test.sqlite")
    yield connection
    connection.close()


def _student(conn, rng, guardian=True) -> str:
    """Новий учень зі згодою."""
    code = storage.new_student_code(conn, rng)
    storage.register_consent(conn, code, guardian_consent=guardian)
    return code


def _answer(n: int, subtopic: str = "domain", correct: bool = True) -> dict:
    """Запис однієї відповіді в тому вигляді, як його готує застосунок."""
    return {"n": n, "item_id": f"it{n:03d}", "subtopic": subtopic, "difficulty": 0.0, "params": {"a": n},
            "chosen": 1, "correct": correct, "time_sec": 42.0, "action": "same", "p_hat": 0.7,
            "theta_hat": {"domain": 0.1}, "state": 67, "fallback": False}


def _session(conn, code, answers=3, dropped=False, gain=0.2) -> int:
    """Сесія з кількома відповідями, завершена."""
    sid = storage.start_session(conn, code, {"mood": 4, "energy": 3, "sleep": None})
    for n in range(1, answers + 1):
        storage.save_answer(conn, sid, _answer(n, correct=n % 2 == 1))
    storage.end_session(conn, sid, dropped=dropped, stats={"theta_gain": gain, "breaks": 1, "break_fallbacks": 0})
    return sid


def test_connect_creates_folder_and_tables(tmp_path) -> None:
    """Файл бази й три таблиці з'являються самі."""
    path = tmp_path / "new" / "db.sqlite"
    connection = storage.connect(path)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    connection.close()
    assert path.exists() and {"students", "sessions", "answers"} <= tables


def test_tables_have_no_personal_fields(conn) -> None:
    """У схемі немає полів для імені, пошти, класу, дати народження, IP."""
    columns = set()
    for table in ("students", "sessions", "answers"):
        columns |= {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    forbidden = {"name", "first_name", "last_name", "email", "class", "birth", "birthday", "ip", "phone"}
    assert columns.isdisjoint(forbidden)


def test_student_code_format_and_unique(conn) -> None:
    """Код: потрібна довжина, лише «несхожі» символи; повторно не видається."""
    rng = random.Random(0)
    codes = {_student(conn, rng) for _ in range(50)}
    assert len(codes) == 50
    for code in codes:
        assert len(code) == config.STUDENT_CODE_LENGTH and set(code) <= set(config.STUDENT_CODE_ALPHABET)


def test_new_code_skips_existing(conn) -> None:
    """Якщо згенерований код уже є в базі — генерується інший."""
    first = _student(conn, random.Random(1))
    second = storage.new_student_code(conn, random.Random(1))   # той самий seed → спершу той самий код
    assert second != first


def test_new_code_is_not_stored_without_consent(conn) -> None:
    """new_student_code нічого не записує: без згоди учня в базі немає."""
    code = storage.new_student_code(conn, random.Random(0))
    assert not storage.student_exists(conn, code)


def test_session_needs_consent(conn) -> None:
    """Без згоди (або без згоди батьків/школи) сесію не почати."""
    with pytest.raises(PermissionError):
        storage.start_session(conn, "ABCDEF", None)
    code = _student(conn, random.Random(0), guardian=False)
    with pytest.raises(PermissionError):
        storage.start_session(conn, code, None)


def test_bad_checkin_rejected(conn) -> None:
    """Чек-ін — лише 1–5 або None."""
    code = _student(conn, random.Random(0))
    with pytest.raises(ValueError):
        storage.start_session(conn, code, {"mood": 7})


def test_answer_roundtrip(conn) -> None:
    """Записана відповідь читається назад з тими самими значеннями (JSON-поля — як dict)."""
    code = _student(conn, random.Random(0))
    sid = _session(conn, code, answers=2)
    data = storage.student_data(conn, code)
    first = data["sessions"][0]["answers"][0]
    assert sid == data["sessions"][0]["id"]
    assert first["params"] == {"a": 1} and first["theta_hat"] == {"domain": 0.1}
    assert first["correct"] is True and first["fallback"] is False and first["time_sec"] == 42.0


def test_end_session_counts_items(conn) -> None:
    """end_session ставить час кінця, кількість завдань і відмову."""
    code = _student(conn, random.Random(0))
    _session(conn, code, answers=3, dropped=True)
    session = storage.student_data(conn, code)["sessions"][0]
    assert session["items_done"] == 3 and session["dropped"] is True and session["ended_at"]
    assert session["theta_gain"] == pytest.approx(0.2) and session["breaks"] == 1


def test_export_has_everything_of_one_student(conn, tmp_path) -> None:
    """Експорт: JSON з усіма сесіями й відповідями цього учня і жодними чужими."""
    rng = random.Random(0)
    me, other = _student(conn, rng), _student(conn, rng)
    _session(conn, me, answers=2)
    _session(conn, me, answers=3)
    _session(conn, other, answers=4)
    path = storage.export_student(conn, me, folder=tmp_path / "exports")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["code"] == me and len(data["sessions"]) == 2
    assert sum(len(s["answers"]) for s in data["sessions"]) == 5
    assert other not in path.read_text(encoding="utf-8")


def test_delete_removes_everything(conn) -> None:
    """Видалення: зникають учень, його сесії й відповіді; чужі дані лишаються."""
    rng = random.Random(0)
    me, other = _student(conn, rng), _student(conn, rng)
    _session(conn, me, answers=2)
    _session(conn, other, answers=1)
    removed = storage.delete_student(conn, me)
    assert removed == 1 + 1 + 2
    assert not storage.student_exists(conn, me) and storage.student_exists(conn, other)
    assert conn.execute("SELECT COUNT(*) FROM answers").fetchone()[0] == 1


def test_delete_unknown_code_is_zero(conn) -> None:
    """Видалення неіснуючого коду нічого не ламає."""
    assert storage.delete_student(conn, "ZZZZZZ") == 0


def test_history_by_day(conn) -> None:
    """Історія: по днях — сесії, завдання, частка правильних."""
    code = _student(conn, random.Random(0))
    _session(conn, code, answers=3)
    _session(conn, code, answers=1)
    history = storage.student_history(conn, code)
    assert len(history) == 1
    day = history[0]
    assert day["sessions"] == 2 and day["items"] == 4 and day["accuracy"] == pytest.approx(3 / 4)


def test_aggregates_hidden_for_small_group(conn) -> None:
    """2 учні — агрегатів немає (порожній словник)."""
    rng = random.Random(0)
    for _ in range(2):
        _session(conn, _student(conn, rng))
    assert storage.class_aggregates(conn, before=FUTURE) == {}


def test_aggregates_for_group_without_codes(conn) -> None:
    """3 учні — агрегати є, і в них немає жодного коду учня."""
    rng = random.Random(0)
    codes = [_student(conn, rng) for _ in range(3)]
    for i, code in enumerate(codes):
        _session(conn, code, answers=2, dropped=(i == 0))
    agg = storage.class_aggregates(conn, before=FUTURE)
    assert agg["students"] == 3 and agg["sessions"] == 3
    assert agg["dropped_share"] == pytest.approx(1 / 3)
    assert agg["accuracy_by_subtopic"]["domain"] == pytest.approx(0.5)
    assert agg["mean_theta_gain"] == pytest.approx(0.2)
    text = json.dumps(agg, ensure_ascii=False)
    assert not any(code in text for code in codes)


def test_aggregates_hide_rare_subtopics(conn) -> None:
    """Підтему, яку розв'язували менше ніж 3 учні, вчитель не бачить (щоб не впізнати учня)."""
    rng = random.Random(0)
    codes = [_student(conn, rng) for _ in range(3)]
    for code in codes:
        _session(conn, code, answers=1)
    sid = storage.start_session(conn, codes[0], None)
    storage.save_answer(conn, sid, _answer(1, subtopic="range"))
    agg = storage.class_aggregates(conn, before=FUTURE)
    assert "domain" in agg["accuracy_by_subtopic"] and "range" not in agg["accuracy_by_subtopic"]


def test_unfinished_session_counts_as_dropped(conn) -> None:
    """Сесія без end_session (учень закрив вкладку) рахується як достроково завершена."""
    rng = random.Random(0)
    codes = [_student(conn, rng) for _ in range(3)]
    for code in codes[:2]:
        _session(conn, code)
    sid = storage.start_session(conn, codes[2], None)
    storage.save_answer(conn, sid, _answer(1))
    assert storage.class_aggregates(conn, before=FUTURE)["dropped_share"] == pytest.approx(1 / 3)


def test_foreign_keys_on(conn) -> None:
    """Відповідь до неіснуючої сесії не записується."""
    with pytest.raises(sqlite3.IntegrityError):
        storage.save_answer(conn, 999, _answer(1))


def test_delete_wipes_file_bytes(tmp_path) -> None:
    """Після видалення код і відповіді учня не лишаються в байтах файлу бази (secure_delete)."""
    path = tmp_path / "wipe.sqlite"
    conn = storage.connect(path)
    code = _student(conn, random.Random(0))
    _session(conn, code, answers=3)
    storage.delete_student(conn, code, folder=tmp_path)
    conn.close()
    raw = path.read_bytes()
    assert code.encode() not in raw and b"it003" not in raw


def test_delete_removes_export_file(conn, tmp_path) -> None:
    """Видалення прибирає і файл експорту учня."""
    code = _student(conn, random.Random(0))
    path = storage.export_student(conn, code, folder=tmp_path)
    storage.delete_student(conn, code, folder=tmp_path)
    assert not path.exists()


def test_breaks_logged_and_exported(conn) -> None:
    """Рішення про перерву записується (і чи повернувся учень); видно в експорті; видаляється з учнем."""
    code = _student(conn, random.Random(0))
    sid = storage.start_session(conn, code, None)
    bid = storage.save_break(conn, sid, {"after_items": 6, "state": 70, "fallback": False})
    storage.break_taken(conn, bid, 4)
    pause = storage.student_data(conn, code)["sessions"][0]["break_log"][0]
    assert pause["after_items"] == 6 and pause["taken"] is True and pause["energy"] == 4
    assert storage.delete_student(conn, code) == 3      # перерва + сесія + учень


def test_aggregates_show_finished_today(conn) -> None:
    """Завершені сьогоднішні сесії вчитель бачить одразу; незавершені — ні (лише з наступного дня)."""
    rng = random.Random(0)
    codes = [_student(conn, rng) for _ in range(3)]
    for code in codes[:2]:
        _session(conn, code)
    sid = storage.start_session(conn, codes[2], None)
    storage.save_answer(conn, sid, _answer(1))
    assert storage.class_aggregates(conn) == {}                 # третій ще розв'язує → видно лише 2
    assert storage.in_progress_today(conn) == 1
    storage.end_session(conn, sid, dropped=False)
    agg = storage.class_aggregates(conn)
    assert agg["students"] == 3 and agg["sessions"] == 3 and storage.in_progress_today(conn) == 0


def test_accuracy_is_mean_of_students(conn) -> None:
    """Частка правильних — середнє по учнях: учень з 10 відповідями важить стільки ж, скільки з 1."""
    rng = random.Random(0)
    codes = [_student(conn, rng) for _ in range(3)]
    for code, (answers, correct) in zip(codes, [(10, True), (1, False), (1, False)]):
        sid = storage.start_session(conn, code, None)
        for n in range(answers):
            storage.save_answer(conn, sid, _answer(n + 1, correct=correct))
    assert storage.class_aggregates(conn, before=FUTURE)["accuracy_by_subtopic"]["domain"] == pytest.approx(1 / 3)


def test_gain_hidden_when_few_finished(conn) -> None:
    """Приріст θ̂ не показується, якщо завершені сесії є менш ніж у 3 учнів."""
    rng = random.Random(0)
    codes = [_student(conn, rng) for _ in range(3)]
    for code in codes[:2]:
        _session(conn, code)
    sid = storage.start_session(conn, codes[2], None)
    storage.save_answer(conn, sid, _answer(1))
    assert storage.class_aggregates(conn, before=FUTURE)["mean_theta_gain"] is None
