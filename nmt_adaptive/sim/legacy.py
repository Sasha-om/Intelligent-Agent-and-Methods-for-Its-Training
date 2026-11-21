"""Міст до тренажера МАН: беремо готові функції з Platform for Interaction/qlearn/comparison.py.

Що беремо і навіщо:
    write_csv(path, rows, fields)   — CSV для Excel: роздільник «;», десяткова кома, BOM (кирилиця відкривається одразу);
    moving_average(values, window)  — ковзне середнє для кривої навчання (як у тренажері).
Файли тренажера НЕ змінюються — лише імпортуються. Якщо тренажера поруч немає (наприклад, папку скопіювали окремо),
спрацьовують запасні копії з тим самим форматом, і TRAINER_AVAILABLE = False.
"""
import sys
from pathlib import Path

import config


def _import_trainer() -> tuple:
    """Пробує імпортувати функції тренажера; повертає (write_csv, moving_average) або (None, None)."""
    folder = str(config.TRAINER_DIR)
    if not Path(folder, "qlearn", "comparison.py").exists():
        return None, None
    if folder not in sys.path:
        sys.path.append(folder)
    try:
        from qlearn.comparison import moving_average as trainer_ma
        from qlearn.comparison import write_csv as trainer_csv
    except ImportError:
        return None, None
    return trainer_csv, trainer_ma


def _fmt(value: object) -> str:
    """Запасний формат клітинки — такий самий, як у тренажері (десяткова кома, «так/ні»)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "так" if value else "ні"
    if isinstance(value, float):
        return ("%.6f" % value).rstrip("0").rstrip(".").replace(".", ",")
    return str(value).replace(";", ",").replace("\r", " ").replace("\n", " ")


def _fallback_csv(path: Path, rows: list, fields: list) -> Path:
    """Запасний CSV для Excel: «;», десяткова кома, CRLF, utf-8-sig."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [";".join(fields)] + [";".join(_fmt(r.get(f)) for f in fields) for r in rows]
    with open(path, "w", encoding="utf-8-sig", newline="") as file:
        file.write("\r\n".join(lines))
    return path


def _fallback_ma(values: list, window: int) -> list:
    """Запасне ковзне середнє «назад»: точка i — середнє останніх window значень до i включно."""
    out, total = [], 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


_trainer_csv, _trainer_ma = _import_trainer()
TRAINER_AVAILABLE = _trainer_csv is not None
write_csv = _trainer_csv or _fallback_csv
moving_average = _trainer_ma or _fallback_ma
