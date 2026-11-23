"""Сховище пілоту: SQLite з анонімними сесіями (без streamlit — логіку можна тестувати окремо).

Що зберігаємо: випадковий код учня, час згоди, добровільний чек-ін (1–5), відповіді на завдання і рішення агента.
Чого НЕ зберігаємо: імені, пошти, класу, дати народження, IP, медичних даних.
Файл бази — data/students/ (у .gitignore: дані учнів ніколи не йдуть у git).
"""
import json
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    code TEXT PRIMARY KEY, created_at TEXT NOT NULL, consent_at TEXT NOT NULL,
    guardian_consent INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, code TEXT NOT NULL REFERENCES students(code),
    started_at TEXT NOT NULL, ended_at TEXT, mood INTEGER, energy INTEGER, sleep INTEGER,
    dropped INTEGER NOT NULL DEFAULT 0, items_done INTEGER NOT NULL DEFAULT 0,
    breaks INTEGER NOT NULL DEFAULT 0, break_fallbacks INTEGER NOT NULL DEFAULT 0, theta_gain REAL);
CREATE TABLE IF NOT EXISTS answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL REFERENCES sessions(id),
    n INTEGER NOT NULL, item_id TEXT NOT NULL, subtopic TEXT NOT NULL, difficulty REAL NOT NULL,
    params TEXT NOT NULL, chosen INTEGER NOT NULL, correct INTEGER NOT NULL, time_sec REAL NOT NULL,
    action TEXT NOT NULL, p_hat REAL, theta_hat TEXT NOT NULL, state INTEGER, fallback INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS breaks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL REFERENCES sessions(id),
    after_items INTEGER NOT NULL, state INTEGER, fallback INTEGER NOT NULL, taken INTEGER NOT NULL DEFAULT 0,
    energy INTEGER);
CREATE TABLE IF NOT EXISTS test_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL REFERENCES sessions(id),
    phase TEXT NOT NULL, form TEXT NOT NULL, slot INTEGER NOT NULL, item_id TEXT NOT NULL, subtopic TEXT NOT NULL,
    difficulty REAL NOT NULL, chosen INTEGER NOT NULL, correct INTEGER NOT NULL, time_sec REAL NOT NULL);
"""
MIGRATIONS = {   # поля, додані на етапі 10: старі бази (етап 9) отримують їх автоматично
    "students": {"form_order": "TEXT"},
    "sessions": {"arm": "TEXT", "card_position": "INTEGER", "pre_score": "INTEGER", "post_score": "INTEGER",
                 "phase": "TEXT"},
}
ANSWER_FIELDS = ("n", "item_id", "subtopic", "difficulty", "params", "chosen", "correct", "time_sec",
                 "action", "p_hat", "theta_hat", "state", "fallback")
JSON_FIELDS = ("params", "theta_hat")
BOOL_FIELDS = ("correct", "fallback", "dropped", "guardian_consent", "taken")


def _now() -> str:
    """Місцевий час до секунди, напр. 2026-11-22T10:15:00+02:00 (перші 10 символів — дата)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Відкриває базу (за замовчуванням config.DB_PATH); створює папку й таблиці, якщо їх немає."""
    path = Path(path if path is not None else config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=config.DB_TIMEOUT_SEC)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA secure_delete = ON")   # видалене затирається нулями, а не лишається у файлі
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Додає поля з MIGRATIONS, яких ще немає (база, створена до етапу 10, працює далі без втрати даних)."""
    for table, columns in MIGRATIONS.items():
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns.items():
            if name not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    conn.commit()


# ----------------------------------------------------------------------------- учень і згода
def student_exists(conn: sqlite3.Connection, code: str) -> bool:
    """Чи є учень з таким кодом."""
    return conn.execute("SELECT 1 FROM students WHERE code = ?", (code,)).fetchone() is not None


def new_student_code(conn: sqlite3.Connection, rng: random.Random) -> str:
    """Новий унікальний код (STUDENT_CODE_LENGTH символів без схожих 0/O, 1/I/L). Нічого не записує.

    У застосунку rng = random.SystemRandom(): код не можна вгадати за seed.
    """
    for _ in range(config.CODE_ATTEMPTS):
        code = "".join(rng.choice(config.STUDENT_CODE_ALPHABET) for _ in range(config.STUDENT_CODE_LENGTH))
        if not student_exists(conn, code):
            return code
    raise RuntimeError("Не вдалося знайти вільний код — збільште STUDENT_CODE_LENGTH.")


def register_consent(conn: sqlite3.Connection, code: str, guardian_consent: bool) -> None:
    """Записує (або оновлює) згоду учня. Це ПЕРШИЙ запис про учня: до згоди в базі нічого немає."""
    now = _now()
    conn.execute("INSERT INTO students (code, created_at, consent_at, guardian_consent) VALUES (?, ?, ?, ?) "
                 "ON CONFLICT(code) DO UPDATE SET consent_at = excluded.consent_at, "
                 "guardian_consent = excluded.guardian_consent", (code, now, now, int(guardian_consent)))
    conn.commit()


def has_consent(conn: sqlite3.Connection, code: str) -> bool:
    """Чи можна записувати дані учня: є згода (і згода батьків/школи, якщо вона вимагається)."""
    row = conn.execute("SELECT guardian_consent FROM students WHERE code = ?", (code,)).fetchone()
    return row is not None and (bool(row[0]) or not config.REQUIRE_GUARDIAN_CONSENT)


# ----------------------------------------------------------------------------- сесії й відповіді
def _checked_checkin(checkin: dict | None) -> dict:
    """Чек-ін: лише відомі питання, значення 1–5 або None (не вказано)."""
    checkin = checkin or {}
    unknown = set(checkin) - set(config.CHECKIN_QUESTIONS)
    bad = [k for k, v in checkin.items() if v is not None and v not in config.CHECKIN_SCALE]
    if unknown or bad:
        raise ValueError(f"Чек-ін: невідомі питання {sorted(unknown)} або значення не 1–5 у {bad}.")
    return {key: checkin.get(key) for key in config.CHECKIN_QUESTIONS}


def start_session(conn: sqlite3.Connection, code: str, checkin: dict | None) -> int:
    """Нова сесія учня; повертає її номер. Без згоди — PermissionError."""
    if not has_consent(conn, code):
        raise PermissionError("Немає згоди учня (або батьків/школи) — дані не записуємо.")
    c = _checked_checkin(checkin)
    cur = conn.execute("INSERT INTO sessions (code, started_at, mood, energy, sleep) VALUES (?, ?, ?, ?, ?)",
                       (code, _now(), c["mood"], c["energy"], c["sleep"]))
    conn.commit()
    return int(cur.lastrowid)


def save_answer(conn: sqlite3.Connection, session_id: int, record: dict) -> None:
    """Записує одну відповідь (поля ANSWER_FIELDS; params і theta_hat — як JSON)."""
    values = [json.dumps(record[f], ensure_ascii=False) if f in JSON_FIELDS else record[f] for f in ANSWER_FIELDS]
    values = [int(v) if isinstance(v, bool) else v for v in values]
    marks = ", ".join("?" * (len(ANSWER_FIELDS) + 1))
    conn.execute(f"INSERT INTO answers (session_id, {', '.join(ANSWER_FIELDS)}) VALUES ({marks})",
                 [session_id, *values])
    conn.commit()


def save_break(conn: sqlite3.Connection, session_id: int, record: dict) -> int:
    """Записує рішення агента про перерву (after_items, state, fallback); повертає номер запису."""
    cur = conn.execute("INSERT INTO breaks (session_id, after_items, state, fallback) VALUES (?, ?, ?, ?)",
                       (session_id, record["after_items"], record["state"], int(record["fallback"])))
    conn.commit()
    return int(cur.lastrowid)


def break_taken(conn: sqlite3.Connection, break_id: int, energy: int | None) -> None:
    """Учень повернувся з перерви (energy — самооцінка 1–5 або None)."""
    conn.execute("UPDATE breaks SET taken = 1, energy = ? WHERE id = ?", (energy, break_id))
    conn.commit()


def end_session(conn: sqlite3.Connection, session_id: int, dropped: bool, stats: dict | None = None) -> None:
    """Завершує сесію: час кінця, кількість завдань, чи кинув учень; stats — theta_gain, breaks, break_fallbacks."""
    stats = stats or {}
    items = conn.execute("SELECT COUNT(*) FROM answers WHERE session_id = ?", (session_id,)).fetchone()[0]
    conn.execute("UPDATE sessions SET ended_at = ?, dropped = ?, items_done = ?, theta_gain = ?, breaks = ?, "
                 "break_fallbacks = ? WHERE id = ?",
                 (_now(), int(dropped), items, stats.get("theta_gain"), stats.get("breaks", 0),
                  stats.get("break_fallbacks", 0), session_id))
    conn.commit()


# ----------------------------------------------------------------------------- дані учня: історія, експорт, видалення
def _row_dict(row: sqlite3.Row) -> dict:
    """Рядок бази → dict: JSON-поля розгорнуті, 0/1 → False/True."""
    data = dict(row)
    for key in JSON_FIELDS:
        if key in data:
            data[key] = json.loads(data[key])
    for key in BOOL_FIELDS:
        if key in data and data[key] is not None:
            data[key] = bool(data[key])
    return data


def student_data(conn: sqlite3.Connection, code: str) -> dict:
    """Усі дані одного учня: згода, сесії, відповіді (для експорту й кнопки «Завантажити»)."""
    student = conn.execute("SELECT * FROM students WHERE code = ?", (code,)).fetchone()
    if student is None:
        return {}
    sessions = []
    for s in conn.execute("SELECT * FROM sessions WHERE code = ? ORDER BY id", (code,)):
        rows = conn.execute("SELECT * FROM answers WHERE session_id = ? ORDER BY n, id", (s["id"],))
        breaks = conn.execute("SELECT * FROM breaks WHERE session_id = ? ORDER BY id", (s["id"],))
        tests = conn.execute("SELECT * FROM test_answers WHERE session_id = ? ORDER BY id", (s["id"],))
        sessions.append({**_row_dict(s), "answers": [_row_dict(a) for a in rows],
                         "break_log": [_row_dict(b) for b in breaks], "test_answers": [_row_dict(t) for t in tests]})
    return {**_row_dict(student), "exported_at": _now(), "sessions": sessions}


def export_student(conn: sqlite3.Connection, code: str, folder: Path | None = None) -> Path:
    """Записує всі дані учня в JSON-файл у folder (за замовчуванням config.EXPORTS_DIR, не в git)."""
    folder = Path(folder if folder is not None else config.EXPORTS_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"nmt_{code}.json"
    path.write_text(json.dumps(student_data(conn, code), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def delete_student(conn: sqlite3.Connection, code: str, folder: Path | None = None) -> int:
    """Видаляє ВСЕ про учня: відповіді, перерви, тести, сесії, згоду і файл експорту. Повертає кількість рядків бази.

    PRAGMA secure_delete (у connect) затирає видалене у файлі бази, тож його не відновити.
    """
    export = Path(folder if folder is not None else config.EXPORTS_DIR) / f"nmt_{code}.json"
    export.unlink(missing_ok=True)
    ids = "SELECT id FROM sessions WHERE code = ?"
    removed = conn.execute(f"DELETE FROM answers WHERE session_id IN ({ids})", (code,)).rowcount
    removed += conn.execute(f"DELETE FROM breaks WHERE session_id IN ({ids})", (code,)).rowcount
    removed += conn.execute(f"DELETE FROM test_answers WHERE session_id IN ({ids})", (code,)).rowcount
    removed += conn.execute("DELETE FROM sessions WHERE code = ?", (code,)).rowcount
    removed += conn.execute("DELETE FROM students WHERE code = ?", (code,)).rowcount
    conn.commit()
    return removed


def student_history(conn: sqlite3.Connection, code: str) -> list:
    """Прогрес учня по днях: [{"day", "sessions", "items", "accuracy"}] — для підсумку."""
    rows = conn.execute(
        "SELECT substr(s.started_at, 1, 10) AS day, COUNT(DISTINCT s.id) AS sessions, COUNT(a.id) AS items, "
        "AVG(a.correct) AS accuracy FROM sessions s JOIN answers a ON a.session_id = s.id "
        "WHERE s.code = ? GROUP BY day ORDER BY day", (code,))
    return [dict(row) for row in rows]


# ----------------------------------------------------------------------------- вчитель: лише агрегати
def today() -> str:
    """Сьогоднішня дата (місцева) у форматі РРРР-ММ-ДД."""
    return _now()[:10]


VISIBLE = "(s.ended_at IS NOT NULL OR substr(s.started_at, 1, 10) < ?)"   # завершені або з минулих днів


def _answers_before(conn: sqlite3.Connection, before: str) -> list:
    """Відповіді з видимих вчителю сесій (завершені — одразу, незавершені — з дати before), з кодом учня."""
    return conn.execute("SELECT s.code, a.subtopic, a.correct, a.time_sec, a.fallback FROM answers a "
                        f"JOIN sessions s ON s.id = a.session_id WHERE {VISIBLE}", (before,)).fetchall()


def _mean_of_students(rows: list, field: str) -> float:
    """Середнє середніх по учнях: кожен учень важить однаково, хоч скільки відповідей він дав."""
    per_student = {}
    for r in rows:
        per_student.setdefault(r["code"], []).append(r[field])
    return sum(sum(v) / len(v) for v in per_student.values()) / len(per_student)


def _accuracy_by_subtopic(rows: list, min_group: int) -> dict:
    """Частка правильних за підтемами (середнє по учнях) — лише для підтем, які розв'язували ≥ min_group учнів."""
    by_topic = {}
    for r in rows:
        by_topic.setdefault(r["subtopic"], []).append(r)
    return {topic: _mean_of_students(items, "correct") for topic, items in by_topic.items()
            if len({r["code"] for r in items}) >= min_group}


def _session_stats(conn: sqlite3.Connection, min_group: int, before: str) -> dict:
    """Видимі сесії з відповідями: кількість, частка недокінчених, середній приріст θ̂ (від ≥ min_group учнів)."""
    rows = conn.execute("SELECT s.code, s.dropped, s.ended_at, s.theta_gain, COUNT(a.id) AS items "
                        "FROM sessions s JOIN answers a ON a.session_id = s.id "
                        f"WHERE {VISIBLE} GROUP BY s.id", (before,)).fetchall()
    dropped = [bool(r["dropped"]) or r["ended_at"] is None for r in rows]
    gains = [r for r in rows if r["ended_at"] is not None and r["theta_gain"] is not None]
    enough = len({r["code"] for r in gains}) >= min_group
    return {"sessions": len(rows), "dropped_share": sum(dropped) / len(rows),
            "mean_items": sum(r["items"] for r in rows) / len(rows),
            "mean_theta_gain": sum(r["theta_gain"] for r in gains) / len(gains) if enough else None}


def class_aggregates(conn: sqlite3.Connection, min_group: int = config.MIN_GROUP_FOR_STATS,
                     before: str | None = None) -> dict:
    """Лише агреговані числа класу, без кодів учнів. Учнів < min_group → {} (щоб не впізнати людину).

    Завершену сесію вчитель бачить одразу. Незавершену (учень ще розв'язує) — лише з дати before
    (за замовчуванням — із завтра): інакше, оновлюючи сторінку до і після відповіді, можна було б
    побачити результат конкретного учня за зміною чисел.
    """
    before = before or (today() if config.TEACHER_UNFINISHED_HIDDEN else "9999-12-31")
    rows = _answers_before(conn, before)
    students = len({r["code"] for r in rows})
    if students < min_group:
        return {}
    accuracy = _accuracy_by_subtopic(rows, min_group)
    weak = sorted(accuracy, key=accuracy.get)[:config.TEACHER_WEAK_TOPICS]
    return {"students": students, **_session_stats(conn, min_group, before), "accuracy_by_subtopic": accuracy,
            "weak_topics": weak, "mean_time_sec": _mean_of_students(rows, "time_sec"),
            "fallback_share": sum(r["fallback"] for r in rows) / len(rows)}


def in_progress_today(conn: sqlite3.Connection) -> int:
    """Скільки сесій сьогодні почато і ще не завершено (лише кількість, без результатів)."""
    return conn.execute("SELECT COUNT(*) FROM sessions WHERE ended_at IS NULL AND substr(started_at, 1, 10) = ?",
                        (today(),)).fetchone()[0]
