"""Тест звіту після уроку (app/pilot_report.py) на синтетичній базі з відомим ефектом."""
import random
from contextlib import closing

import pytest

import config
from app import pilot_report, pilot_storage, storage


def _fill(path, plan: dict, seed: int = 0) -> None:
    """База: plan = {режим: [(вхід, вихід або None), ...]}; кожен урок — новий учень, тест + 2 навчальні відповіді."""
    rng = random.Random(seed)
    with closing(storage.connect(path)) as conn:
        for arm, lessons in plan.items():
            for pre, post in lessons:
                code = storage.new_student_code(conn, rng)
                storage.register_consent(conn, code, True)
                pilot_storage.form_order(conn, code, rng)
                sid = storage.start_session(conn, code, None)
                for slot in range(1, len(config.PILOT_TEST_SLOTS) + 1):
                    pilot_storage.save_test_answer(conn, sid, {
                        "phase": "pre", "form": "A", "slot": slot, "item_id": f"testA{slot}", "subtopic": "domain",
                        "difficulty": float(slot % 3 - 1), "chosen": 0, "correct": slot <= pre, "time_sec": 20.0})
                pilot_storage.save_score(conn, sid, "pre", pre)
                pilot_storage.choose_arm(conn, sid, arm, 1 + len(arm) % 3)
                for n, level in ((1, -1.0), (2, 1.0)):
                    storage.save_answer(conn, sid, {"n": n, "item_id": "x", "subtopic": "domain", "difficulty": level,
                                                    "params": {}, "chosen": 0, "correct": level < 0, "time_sec": 30.0,
                                                    "action": "same", "p_hat": 0.7, "theta_hat": {}, "state": 0,
                                                    "fallback": False})
                if post is not None:
                    pilot_storage.save_score(conn, sid, "post", post)
                storage.end_session(conn, sid, dropped=post is None)


def test_report_recovers_effect(tmp_path) -> None:
    """Q дає +2 бали більше за CAT при тих самих вхідних балах → ANCOVA ≈ 2; файли лише в out."""
    db = tmp_path / "a.sqlite"
    pres = [1, 2, 3, 4, 2, 3]
    _fill(db, {"q_agent": [(p, min(6, p + 2)) for p in pres], "cat": [(p, p) for p in pres],
               "fixed": [(2, 3), (3, None)]})
    out = tmp_path / "out"
    tables = pilot_report.build([db], out)
    comparison = tables["пілот_порівняння"][0]
    assert comparison["ANCOVA_ефект"] == pytest.approx(2, abs=0.35)
    fixed = next(r for r in tables["пілот_режими"] if r["режим"] == config.PILOT_ARM_NAMES_UA["fixed"])
    assert fixed["dropped"] == pytest.approx(0.5)
    names = {p.name for p in out.iterdir()}
    assert {"пілот_режими.csv", "пілот_порівняння.csv", "пілот_вибір.csv", "пілот_форми.csv",
            "пілот_вхід_вихід.png", "пілот_складність_банку.png", "пілот_крива_навчання.csv",
            "пілот_швидкість_навчання.csv", "пілот_крива_навчання.png"} <= names
    curve = [r for r in tables["пілот_крива_навчання"] if r["режим"] == "Q-агент"]
    assert [(r["спроба"], r["відповідей"], r["правильно"]) for r in curve] == [(1, 6, 1.0), (2, 6, 0.0)]
    rates = {r["режим"]: r for r in tables["пілот_швидкість_навчання"]}
    assert rates["Q-агент"]["сесій"] == 6 and rates["Q-агент"]["швидкість_пп_за_спробу"] < 0


def test_report_merges_several_databases(tmp_path) -> None:
    """Дві бази (два ПК) — сесії складаються; однакові номери сесій у різних базах не змішуються."""
    a, b = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    _fill(a, {"q_agent": [(2, 3)]}, seed=1)
    _fill(b, {"q_agent": [(4, 5)]}, seed=2)
    tables = pilot_report.build([a, b], tmp_path / "out")
    q = tables["пілот_режими"][0]
    assert q["chosen"] == 2 and q["pre"] == pytest.approx(3)
    assert sum(r["сесій"] for r in tables["пілот_форми"]) == 2


def test_difficulty_and_no_codes(tmp_path) -> None:
    """Складність: рівень −1 розв'язують частіше за +1; у файлах звіту немає кодів учнів."""
    db = tmp_path / "a.sqlite"
    _fill(db, {"cat": [(3, 3), (2, 4), (4, 4)]})
    out = tmp_path / "out"
    tables = pilot_report.build([db], out)
    train = {r["рівень"]: r["частка_правильних"] for r in tables["пілот_складність_банку"] if r["джерело"] == "тренування"}
    assert train[-1] > train[1]
    with closing(storage.connect(db)) as conn:
        codes = [r[0] for r in conn.execute("SELECT code FROM students")]
    text = "".join(p.read_text(encoding="utf-8-sig") for p in out.glob("*.csv"))
    assert not any(code in text for code in codes)


def test_cli_missing_db(tmp_path) -> None:
    """Неіснуюча база — зрозуміле повідомлення, а не traceback."""
    with pytest.raises(SystemExit):
        pilot_report.main(["--db", str(tmp_path / "немає.sqlite"), "--out", str(tmp_path / "o")])


def test_report_reads_only_and_uses_first_sessions(tmp_path) -> None:
    """Звіт не змінює файл бази; повторна сесія того самого учня не потрапляє в порівняння режимів."""
    db = tmp_path / "a.sqlite"
    _fill(db, {"cat": [(2, 3), (3, 3), (4, 4)]})
    with closing(storage.connect(db)) as conn:          # друга сесія першого учня — в іншому режимі
        code = conn.execute("SELECT code FROM students ORDER BY rowid LIMIT 1").fetchone()[0]
        sid = storage.start_session(conn, code, None)
        pilot_storage.save_score(conn, sid, "pre", 5)
        pilot_storage.choose_arm(conn, sid, "q_agent", 1)
        pilot_storage.save_score(conn, sid, "post", 6)
        storage.end_session(conn, sid, False)
    before = db.read_bytes()
    tables = pilot_report.build([db], tmp_path / "out")
    assert db.read_bytes() == before
    q = next(r for r in tables["пілот_режими"] if r["режим"] == config.PILOT_ARM_NAMES_UA["q_agent"])
    assert q["chosen"] == 0
    choice = next(r for r in tables["пілот_вибір"] if r["режим"] == config.PILOT_ARM_NAMES_UA["cat"])
    assert choice["порядок_AB"] + choice["порядок_BA"] == 3
