"""База пілоту-порівняння (етап 10): порядок форм, режим сесії, відповіді тестів, бали, порівняння режимів.

Таблиці й поля створює app/storage.connect (SCHEMA + MIGRATIONS). Без streamlit.
Вчитель бачить лише агрегати: режим із < MIN_GROUP_FOR_STATS учнів — «замало даних»; лише видимі сесії
(завершені або з минулих днів — як у storage.class_aggregates).
"""
import random
import sqlite3
from datetime import datetime

import numpy as np

import config
from app import storage
from app.pilot import normalized_gain
from core.stats import ancova, diff_ci, hedges_g, mean_ci

TEST_FIELDS = ("phase", "form", "slot", "item_id", "subtopic", "difficulty", "chosen", "correct", "time_sec")
PHASES = ("pre", "choose", "train", "post", "done")
TEST_ITEMS = len(config.PILOT_TEST_SLOTS)


def _least_used(counts: dict, rng: random.Random) -> str:
    """Варіант, який трапляється найрідше (нічия — випадково): так групи лишаються рівними."""
    low = min(counts.values())
    return rng.choice(sorted(k for k, v in counts.items() if v == low))


def form_order(conn: sqlite3.Connection, code: str, rng: random.Random) -> str:
    """Порядок форм учня («AB»: вхід А, вихід Б; або «BA»). Призначається один раз, порівну між учнями."""
    row = conn.execute("SELECT form_order FROM students WHERE code = ?", (code,)).fetchone()
    if row is None:
        raise KeyError(f"Учня {code!r} немає.")
    if row[0]:
        return row[0]
    counts = {order: 0 for order in config.PILOT_FORM_ORDERS}
    for (order,) in conn.execute("SELECT form_order FROM students WHERE form_order IS NOT NULL"):
        counts[order] = counts.get(order, 0) + 1
    order = _least_used(counts, rng)
    conn.execute("UPDATE students SET form_order = ? WHERE code = ?", (order, code))
    conn.commit()
    return order


def assign_arm(conn: sqlite3.Connection, rng: random.Random) -> str:
    """Режим «навмання, порівну» (PILOT_ASSIGNMENT = "random"): той, що досі обирався найрідше."""
    counts = {arm: 0 for arm in config.PILOT_ARMS}
    for (arm,) in conn.execute("SELECT arm FROM sessions WHERE arm IS NOT NULL"):
        counts[arm] = counts.get(arm, 0) + 1
    return _least_used(counts, rng)


def choose_arm(conn: sqlite3.Connection, session_id: int, arm: str, card_position: int | None) -> None:
    """Записує режим сесії (обраний учнем; card_position — на якому місці 1–3 стояла картка)."""
    if arm not in config.PILOT_ARMS:
        raise ValueError(f"Невідомий режим {arm!r}")
    conn.execute("UPDATE sessions SET arm = ?, card_position = ?, phase = 'train' WHERE id = ?",
                 (arm, card_position, session_id))
    conn.commit()


def set_phase(conn: sqlite3.Connection, session_id: int, phase: str) -> None:
    """Етап сесії: pre → choose → train → post → done."""
    if phase not in PHASES:
        raise ValueError(f"Невідомий етап {phase!r}")
    conn.execute("UPDATE sessions SET phase = ? WHERE id = ?", (phase, session_id))
    conn.commit()


def save_test_answer(conn: sqlite3.Connection, session_id: int, record: dict) -> None:
    """Одна відповідь вхідного/вихідного тесту (поля TEST_FIELDS; record — з PilotTest.answer)."""
    values = [int(record[f]) if isinstance(record[f], bool) else record[f] for f in TEST_FIELDS]
    marks = ", ".join("?" * (len(TEST_FIELDS) + 1))
    conn.execute(f"INSERT INTO test_answers (session_id, {', '.join(TEST_FIELDS)}) VALUES ({marks})",
                 [session_id, *values])
    conn.commit()


def save_score(conn: sqlite3.Connection, session_id: int, phase: str, score: int) -> None:
    """Бал тесту: phase "pre" → pre_score, "post" → post_score."""
    column = {"pre": "pre_score", "post": "post_score"}[phase]
    conn.execute(f"UPDATE sessions SET {column} = ? WHERE id = ?", (int(score), session_id))
    conn.commit()


# ----------------------------------------------------------------------------- порівняння режимів (вчитель)
def first_sessions(rows: list) -> list:
    """Лише ПЕРША сесія пілоту кожного учня: повторні — з тими самими тестами і не незалежні (H1, H2)."""
    first = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        first.setdefault(row["code"], row)
    return list(first.values())


def _batched(rows: list, min_group: int) -> list:
    """Сесії режиму «пачками» по min_group (за часом завершення): таблиця змінюється, лише коли завершать ще
    min_group учнів — тож за різницею двох оновлень сторінки не видно балів одного учня."""
    ordered = sorted(rows, key=lambda r: r["ended_at"] or r["started_at"])
    return ordered[:len(ordered) // min_group * min_group]


def pilot_sessions(conn: sqlite3.Connection, before: str | None = None,
                   min_group: int = config.MIN_GROUP_FOR_STATS) -> list:
    """Сесії для таблиці вчителя: перша сесія кожного учня, видима (завершена або з минулих днів), пачками."""
    before = before or (storage.today() if config.TEACHER_UNFINISHED_HIDDEN else "9999-12-31")
    rows = [dict(r) for r in conn.execute(
        "SELECT id, code, arm, pre_score, post_score, dropped, started_at, ended_at, card_position FROM sessions "
        "WHERE arm IS NOT NULL AND pre_score IS NOT NULL")]
    visible = [r for r in first_sessions(rows) if r["ended_at"] or r["started_at"][:10] < before]
    return [r for arm in config.PILOT_ARMS for r in _batched([v for v in visible if v["arm"] == arm], min_group)]


def choice_counts(conn: sqlite3.Connection) -> dict:
    """Скільки сесій сьогодні у кожному режимі (лише кількості — щоб автор бачив розподіл вибору наживо)."""
    rows = conn.execute("SELECT arm, COUNT(*) FROM sessions WHERE arm IS NOT NULL AND substr(started_at, 1, 10) = ? "
                        "GROUP BY arm", (storage.today(),)).fetchall()
    return {arm: dict(rows).get(arm, 0) for arm in config.PILOT_ARMS}


def _minutes(row: dict) -> float | None:
    """Тривалість сесії у хвилинах (лише для завершених)."""
    if not row["ended_at"]:
        return None
    span = datetime.fromisoformat(row["ended_at"]) - datetime.fromisoformat(row["started_at"])
    return span.total_seconds() / 60


RESULT_FIELDS = ("pre_done", "post", "gain", "gain_low", "gain_high", "norm_gain")


def arm_summary(rows: list, rng: np.random.Generator, min_group: int = 1) -> dict:
    """Числа одного режиму. Результати (вихід, приріст…) — лише якщо вихідний тест завершили ≥ min_group учнів."""
    done = [r for r in rows if r["post_score"] is not None]
    gains = [r["post_score"] - r["pre_score"] for r in done]
    norm = [g for g in (normalized_gain(r["pre_score"], r["post_score"]) for r in done) if g is not None]
    minutes = [m for m in map(_minutes, rows) if m is not None]
    gain = mean_ci(gains, rng)
    summary = {"chosen": len(rows), "students": len({r["code"] for r in rows}), "finished": len(done),
               "pre": float(np.mean([r["pre_score"] for r in rows])),
               "pre_done": float(np.mean([r["pre_score"] for r in done])) if done else None,
               "post": float(np.mean([r["post_score"] for r in done])) if done else None,
               "gain": gain[0], "gain_low": gain[1], "gain_high": gain[2],
               "norm_gain": float(np.mean(norm)) if norm else None, "dropped": 1 - len(done) / len(rows),
               "minutes": float(np.mean(minutes)) if minutes else None, "results_hidden": False}
    if len({r["code"] for r in done}) < min_group:
        summary.update(dict.fromkeys(RESULT_FIELDS), results_hidden=True)
    return summary


def compare(rows: list, other: str, rng: np.random.Generator) -> dict | None:
    """Q-агент мінус інший режим (лише завершені), α = PILOT_ALPHA (Бонферроні): сирий приріст [ДІ], Hedges g,
    ANCOVA (різниця при однаковому вхідному балі) з t, df і рішенням «значущо»."""
    q = [r for r in rows if r["arm"] == "q_agent" and r["post_score"] is not None]
    o = [r for r in rows if r["arm"] == other and r["post_score"] is not None]
    if len(q) < 2 or len(o) < 2:
        return None
    gq, go = ([r["post_score"] - r["pre_score"] for r in rs] for rs in (q, o))
    raw = diff_ci(gq, go, rng, alpha=config.PILOT_ALPHA)
    both = q + o
    adj = ancova([r["pre_score"] for r in both], [r["post_score"] for r in both], [r in q for r in both],
                 alpha=config.PILOT_ALPHA)
    return {"other": other, "diff": raw[0], "low": raw[1], "high": raw[2], "hedges_g": hedges_g(gq, go),
            "adjusted": adj, "n_q": len(q), "n_other": len(o)}


def arm_aggregates(conn: sqlite3.Connection, min_group: int = config.MIN_GROUP_FOR_STATS,
                   before: str | None = None, seed: int = 0) -> dict:
    """Порівняння режимів для вчителя: {"arms": {режим: числа або {"chosen", "too_few"}}, "compare": [...]}.

    Режим з < min_group учнів — лише «обрали N»; результати — лише від ≥ min_group учнів, що завершили;
    порівняння — лише між режимами з показаними результатами. Кодів учнів у результаті немає.
    """
    rows, rng = pilot_sessions(conn, before, min_group), np.random.default_rng(seed)
    arms, shown = {}, []
    for arm in config.PILOT_ARMS:
        mine = [r for r in rows if r["arm"] == arm]
        if len({r["code"] for r in mine}) < min_group:
            arms[arm] = {"chosen": len(mine), "too_few": True}
            continue
        arms[arm] = arm_summary(mine, rng, min_group)
        if not arms[arm]["results_hidden"]:
            shown.append(arm)
    rows = [r for r in rows if r["arm"] in shown]
    comparisons = [c for c in (compare(rows, other, rng) for other in config.PILOT_ARMS[1:]
                               if "q_agent" in shown and other in shown) if c]
    return {"arms": arms, "compare": comparisons}
