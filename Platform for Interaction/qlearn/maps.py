"""Вбудовані карти для експериментів і читання карти з файлу.

Вбудовані (ті самі, що K1–K3 у тестах):
    default  — стандартна 7×6 без перешкод (K1);
    wall_gap — стіна посередині з одним проходом (K2);
    rough    — поле дорогих ділянок 3× між стартом і фінішем (K3): тут найдешевший шлях ≠ оптимальний з γ=0.9 (INS-008).

load_map приймає назву вбудованої карти або шлях до JSON:
    • експорт тренажера (кнопка «Завантажити JSON», розділ "environment", клітинки як "c,r");
    • файл запуску з results/ (розділ "env" → "map");
    • просто словник карти (cols, rows, obstacles, rough, start, finish).
"""
import json
from pathlib import Path

from .environment import DEFAULT_ENV

BUILTIN_MAPS = {
    "default": DEFAULT_ENV,
    "wall_gap": {"cols": 7, "rows": 6, "obstacles": [(3, r) for r in range(6) if r != 4], "rough": [],
                 "start": (0, 0), "finish": (6, 0)},
    "rough": {"cols": 7, "rows": 6, "obstacles": [],
              "rough": [(c, r) for c in range(1, 6) for r in range(1, 5)],
              "start": (0, 2), "finish": (6, 2)},
}


def _cell(value):
    """Клітинка з "c,r", [c, r] або (c, r) → (c, r)."""
    if isinstance(value, str):
        c, r = value.split(",")
        return int(c), int(r)
    return int(value[0]), int(value[1])


def _normalize(env):
    return {"cols": int(env["cols"]), "rows": int(env["rows"]),
            "obstacles": [_cell(p) for p in env.get("obstacles", [])],
            "rough": [_cell(p) for p in env.get("rough", [])],
            "start": _cell(env["start"]), "finish": _cell(env["finish"])}


def load_map(name_or_path):
    """Карта як словник для GridEnv. ValueError з поясненням, якщо карту не знайдено чи не розпізнано."""
    if name_or_path in BUILTIN_MAPS:
        return _normalize(BUILTIN_MAPS[name_or_path])
    path = Path(name_or_path)
    if not path.is_file():
        raise ValueError(f"невідома карта {name_or_path!r}: вбудовані карти — {', '.join(BUILTIN_MAPS)}, "
                         f"або вкажіть шлях до JSON-файлу")
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"файл карти {path.name} пошкоджений: {exc}") from exc
    if "environment" in data:                       # експорт тренажера
        return _normalize(data["environment"])
    if "env" in data and "map" in data["env"]:      # файл запуску (схема v1)
        return _normalize(data["env"]["map"])
    if "cols" in data:
        return _normalize(data)
    raise ValueError(f"у файлі {path.name} не знайдено карти (очікувались розділи environment, env.map або cols/rows)")
