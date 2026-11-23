"""Звіт після уроку (етап 10): таблиці й графіки з бази пілоту → results_pilot/ (НЕ в git).

Запуск (з папки nmt_adaptive):
    python -m app.pilot_report                                   # база data/students/nmt_sessions.sqlite
    python -m app.pilot_report --db пк1.sqlite пк2.sqlite        # кілька баз (кожен ПК окремо) — разом

Файли: пілот_режими, пілот_порівняння, пілот_сам_із_собою, пілот_вибір, пілот_форми,
пілот_складність_банку, пілот_проти_симуляції (CSV) + пілот_вхід_вихід.png, пілот_складність_банку.png.
У звіт потрапляють лише агреговані числа — без кодів учнів.
"""
import argparse
import sqlite3
from contextlib import closing
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

import config
from app import pilot_storage
from sim.legacy import write_csv

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ARMS = {arm: config.PILOT_ARM_NAMES_UA[arm] for arm in config.PILOT_ARMS}
SIM_FILE = config.RESULTS_DIR / "симуляція_пілоту_режими.csv"


def load(paths: list) -> dict:
    """Читає одну чи кілька баз: сесії пілоту, відповіді тестів, навчальні відповіді (з номером бази)."""
    frames = {"sessions": [], "tests": [], "answers": []}
    queries = {"sessions": "SELECT s.*, st.form_order FROM sessions s JOIN students st ON st.code = s.code "
                           "WHERE s.arm IS NOT NULL AND s.pre_score IS NOT NULL",
               "tests": "SELECT * FROM test_answers", "answers": "SELECT * FROM answers"}
    for index, path in enumerate(paths):
        uri = Path(path).resolve().as_uri() + "?mode=ro"     # лише читання: бази з інших ПК не змінюються
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            for name, sql in queries.items():
                frame = pd.read_sql_query(sql, conn)
                frame["db"] = index
                frames[name].append(frame)
    return {name: pd.concat(parts, ignore_index=True) for name, parts in frames.items()}


def session_rows(data: dict) -> list:
    """Сесії пілоту як список dict для pilot_storage (код учня — разом з номером бази)."""
    s = data["sessions"]
    rows = s[["id", "code", "arm", "pre_score", "post_score", "dropped", "started_at", "ended_at", "card_position",
              "form_order"]]
    rows = rows.astype(object).where(rows.notna(), None).to_dict("records")
    for row, db in zip(rows, s["db"]):
        row["code"], row["id"] = f"{db}:{row['code']}", (db, int(row["id"]))
        row["post_score"] = None if row["post_score"] is None else int(row["post_score"])
        row["pre_score"] = int(row["pre_score"])
    return rows


def arm_table(rows: list, rng: np.random.Generator) -> list:
    """Числа кожного режиму (перші сесії учнів; навіть якщо учнів < 3 — це звіт автора, не сторінка вчителя)."""
    out = []
    for arm, name in ARMS.items():
        mine = [r for r in rows if r["arm"] == arm]
        out.append({"режим": name, **(pilot_storage.arm_summary(mine, rng) if mine else {"chosen": 0})})
    return out


def comparison_table(rows: list, rng: np.random.Generator) -> list:
    """Q-агент мінус CAT і мінус «звичайний тест» (перші сесії, лише ті, хто завершив; α з поправкою Бонферроні)."""
    out = []
    for other in config.PILOT_ARMS[1:]:
        c = pilot_storage.compare(rows, other, rng)
        if c is None:
            out.append({"порівняння": f"Q-агент − {ARMS[other]}", "примітка": "замало даних (< 2 у групі)"})
            continue
        adj = c["adjusted"] or {}
        out.append({"порівняння": f"Q-агент − {ARMS[other]}", "завершили_Q": c["n_q"], "завершили_інший": c["n_other"],
                    "різниця_приросту": c["diff"], "ДІ_низ": c["low"], "ДІ_верх": c["high"], "hedges_g": c["hedges_g"],
                    "ANCOVA_ефект": adj.get("effect"), "ANCOVA_ДІ_низ": adj.get("low"),
                    "ANCOVA_ДІ_верх": adj.get("high"), "t": adj.get("t"), "df": adj.get("df"),
                    "альфа": config.PILOT_ALPHA, "значущо": adj.get("significant"),
                    "примітка": "серед тих, хто завершив; ДІ 97,5 %"})
    return out


def within_table(rows: list) -> list:
    """«Сам із собою»: учні, що завершили уроки в Q-агента і в іншому режимі — середня різниця їхніх приростів."""
    done = [r for r in rows if r["post_score"] is not None]
    gains = {}
    for r in done:
        gains.setdefault(r["code"], {}).setdefault(r["arm"], []).append(r["post_score"] - r["pre_score"])
    out = []
    for other in config.PILOT_ARMS[1:]:
        diffs = [np.mean(g["q_agent"]) - np.mean(g[other]) for g in gains.values() if "q_agent" in g and other in g]
        out.append({"порівняння": f"Q-агент − {ARMS[other]} (той самий учень)", "учнів": len(diffs),
                    "середня_різниця": float(np.mean(diffs)) if diffs else None})
    return out


def choice_table(rows: list) -> list:
    """Хто що обирав: кількість, частка, середній вхідний бал, на якій позиції стояла обрана картка."""
    out = []
    for arm, name in ARMS.items():
        mine = [r for r in rows if r["arm"] == arm]
        positions = [r["card_position"] for r in mine if r["card_position"] is not None]
        orders = [r["form_order"] for r in mine]
        out.append({"режим": name, "обрали": len(mine), "частка": len(mine) / max(len(rows), 1),
                    "вхідний_бал": float(np.mean([r["pre_score"] for r in mine])) if mine else None,
                    **{f"картка_{p}": positions.count(p) for p in (1, 2, 3)},
                    **{f"порядок_{o}": orders.count(o) for o in config.PILOT_FORM_ORDERS}})
    return out


def forms_table(data: dict) -> list:
    """Рівноцінність форм: середній бал форми А і Б на вхідному (і вихідному) тесті."""
    tests = data["tests"]
    if tests.empty:
        return []
    grouped = tests.groupby(["db", "session_id", "phase", "form"])["correct"]
    scores = grouped.agg(["sum", "count"]).reset_index().rename(columns={"sum": "correct"})
    scores = scores[scores["count"] == len(config.PILOT_TEST_SLOTS)]      # лише повні тести
    out = []
    for (phase, form), part in scores.groupby(["phase", "form"]):
        out.append({"тест": {"pre": "вхідний", "post": "вихідний"}[phase], "форма": form, "сесій": len(part),
                    "середній_бал": float(part["correct"].mean())})
    return out


def difficulty_table(data: dict) -> list:
    """Чи справді «складніші» завдання розв'язують рідше: задуманий рівень → частка правильних (тренування + тести)."""
    parts = [data["answers"][["difficulty", "correct"]].assign(джерело="тренування"),
             data["tests"][["difficulty", "correct"]].assign(джерело="тести")]
    both = pd.concat(parts, ignore_index=True)
    out = []
    for (source, level), part in both.groupby(["джерело", "difficulty"]):
        out.append({"джерело": source, "рівень": int(level), "відповідей": len(part),
                    "частка_правильних": float(part["correct"].mean())})
    if len(both) > 2 and both["difficulty"].nunique() > 1:
        r = float(np.corrcoef(both["difficulty"], both["correct"])[0, 1])
        out.append({"джерело": "усі", "рівень": None, "відповідей": len(both), "частка_правильних": None,
                    "кореляція_рівень_правильність": r})
    return out


def vs_simulation(arms: list) -> list:
    """Реальні учні проти віртуальних (results/симуляція_пілоту_режими.csv): вхід, вихід, приріст, відмови."""
    if not SIM_FILE.exists():
        return []
    sim = pd.read_csv(SIM_FILE, sep=";", decimal=",", encoding="utf-8-sig").set_index("режим")
    out = []
    for real in arms:
        name = real["режим"]
        if name not in sim.index or not real.get("chosen"):
            continue
        s = sim.loc[name]
        out.append({"режим": name, "вхід_реальні": real["pre"], "вхід_симуляція": s["вхід"],
                    "вихід_реальні": real["post"], "вихід_симуляція": s["вихід"],
                    "приріст_реальні": real["gain"], "приріст_симуляція": s["приріст_тест"],
                    "кинули_реальні": real["dropped"], "кинули_симуляція": s["кинули"]})
    return out


def chart_pre_post(arms: list, path: Path) -> None:
    """Графік «вхідний → вихідний тест» по режимах (числа біля точок, десяткова кома)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for row in arms:
        if not row.get("chosen") or row.get("post") is None:
            continue
        ax.plot([0, 1], [row["pre_done"], row["post"]], marker="o", label=f"{row['режим']} (n = {row['finished']})")
        ax.annotate(f"{row['post']:.2f}".replace(".", ","), (1, row["post"]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8)
    ax.set_xticks([0, 1], ["вхідний тест", "вихідний тест"])
    ax.set_ylim(0, len(config.PILOT_TEST_SLOTS))
    ax.set_ylabel("правильних із 6 (середнє)")
    ax.set_title("Пілот: вхідний → вихідний тест (учні, що завершили)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def chart_difficulty(rows: list, path: Path) -> None:
    """Стовпчики: частка правильних за задуманим рівнем (тренування)."""
    train = [r for r in rows if r["джерело"] == "тренування"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar([str(r["рівень"]) for r in train], [100 * r["частка_правильних"] for r in train], color="#4C78A8")
    ax.set_xlabel("задуманий рівень завдання (−2 … 2)")
    ax.set_ylabel("правильних, %")
    ax.set_title("Пілот: чи складніші завдання розв'язують рідше")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save(out: Path, name: str, rows: list) -> None:
    """CSV (роздільник «;», десяткова кома); порожня таблиця — не пишеться."""
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        write_csv(out / f"{name}.csv", rows, fields)


def build(paths: list, out: Path, seed: int = 0) -> dict:
    """Будує всі таблиці й графіки; повертає їх (для тестів і друку)."""
    data, rng = load(paths), np.random.default_rng(seed)
    rows = session_rows(data)
    first = pilot_storage.first_sessions(rows)       # H1, H2: перша сесія кожного учня (незалежні спостереження)
    tables = {"пілот_режими": arm_table(first, rng), "пілот_порівняння": comparison_table(first, rng),
              "пілот_сам_із_собою": within_table(rows), "пілот_вибір": choice_table(rows),
              "пілот_форми": forms_table(data), "пілот_складність_банку": difficulty_table(data)}
    tables["пілот_проти_симуляції"] = vs_simulation(tables["пілот_режими"])
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        save(out, name, table)
    chart_pre_post(tables["пілот_режими"], out / "пілот_вхід_вихід.png")
    if tables["пілот_складність_банку"]:
        chart_difficulty(tables["пілот_складність_банку"], out / "пілот_складність_банку.png")
    return tables


def main(argv: list | None = None) -> None:
    """CLI: python -m app.pilot_report [--db файли…] [--out папка]."""
    parser = argparse.ArgumentParser(description="Звіт пілоту: порівняння режимів → results_pilot/ (не в git).")
    parser.add_argument("--db", nargs="+", default=[str(config.DB_PATH)], help="одна чи кілька баз SQLite")
    parser.add_argument("--out", default=str(config.PILOT_RESULTS_DIR))
    args = parser.parse_args(argv)
    missing = [p for p in args.db if not Path(p).exists()]
    if missing:
        raise SystemExit(f"Бази не знайдено: {', '.join(missing)}")
    try:
        tables = build(args.db, Path(args.out))
    except sqlite3.Error as error:
        raise SystemExit(f"Не вдалося прочитати базу: {error}")
    for row in tables["пілот_режими"]:
        print(f"{row['режим']:<24} обрали {row.get('chosen', 0):>3}  вхід {row.get('pre') or 0:.2f}  "
              f"вихід {row.get('post') or 0:.2f}  приріст {row.get('gain') or 0:+.2f}")
    print(f"Готово → {args.out} (ця папка не йде в git)")


if __name__ == "__main__":
    main()
