"""Дані для вікна «Порівняння результатів» (етап 8): таблиця метрик, криві, кольори, попередження.

Без tkinter, тому все легко тестується (tests/test_compare_view.py). Вікно `CompareWindow`
у q_learning_trainer.py бере звідси готові рядки й точки та лише малює їх.

    rows = rows_for_view(runs)                       # рядки таблиці (тексти)
    xs, ys = run_series(run, "reward", window=20)    # крива одного запуску (≤ 500 точок)
    groups = group_series(runs, "reward", 20)        # «Групувати за seed»: середнє ± σ
    colors = assign_colors([r["run_id"] for r in runs])
"""
import re
import statistics
import zlib
from datetime import datetime

from .comparison import _config_key, compare, moving_average

# Метрики графіка: ключ → підпис осі Y
METRICS = {
    "reward": "Винагорода за епізод",
    "steps": "Кроків за епізод",
    "success": "Дійшов до фінішу (частка)",
    "optimal": "Жадібний шлях: винагорода / V*",
}
METRIC_BUTTONS = {"reward": "Винагорода", "steps": "Кроки", "success": "Успіх", "optimal": "Оптимальність шляху"}
_EPISODE_CURVES = {"reward": "reward_per_episode", "steps": "steps_per_episode", "success": "reached_goal"}

# Стовпці таблиці: ключ → заголовок
TABLE_COLUMNS = [
    ("agent", "Агент"), ("seed", "Seed"), ("map", "Карта"),
    ("first_success_episode", "1-й успіх"), ("convergence_episode", "Збіжність"),
    ("return_ratio", "Шлях / V*"), ("cost_ratio", "Вартість / мін."),
    ("mean_reward_last_50", "Винаг. ост. 50"), ("success_rate_last_50", "Успіх ост. 50"),
    ("total_env_steps", "Кроків"), ("duration_sec", "Час, с"), ("note", "Підпис"),
]
AGENT_LABELS = {"tabular": "Таблиця", "dqn": "Нейромережа"}
PARAM_LABELS = {"alpha": "α", "gamma": "γ", "epsilon": "ε"}

# Палітра ліній: добре видно на темному тлі тренажера, кольори легко розрізнити
PALETTE = ["#4f8cff", "#f2994a", "#27ae60", "#eb5757", "#bb6bd9", "#2dd4bf",
           "#f2c94c", "#ff7eb6", "#a3e635", "#94a3b8"]

DASH = "—"


# ------------------------------------------------------------------ таблиця
def _map_text(run):
    m = run["env"]["map"]
    return f"{m['cols']}×{m['rows']}"


def _num(value, fmt):
    return DASH if value is None else format(value, fmt)


def cell_text(key, run):
    """Текст однієї клітинки таблиці; «—», якщо значення немає (наприклад, не зійшовся)."""
    s = run["summary"]
    if key == "agent":
        return AGENT_LABELS.get(run["agent"], run["agent"])
    if key == "seed":
        return DASH if run["seed"] is None else str(run["seed"])
    if key == "map":
        return _map_text(run)
    if key == "note":
        note = run.get("note") or ""
        changed = run.get("settings", {}).get("changed_during_run") or []
        if changed:
            extra = "змінено посеред запуску: " + ", ".join(PARAM_LABELS.get(c, c) for c in changed)
            note = f"{note} · {extra}" if note else extra
        return note or DASH
    value = s.get(key)
    if key in ("return_ratio", "cost_ratio"):
        return _num(value, ".3f")
    if key == "mean_reward_last_50":
        return _num(value, ".1f")
    if key == "duration_sec":
        return _num(value, ".1f")
    if key == "success_rate_last_50":
        return DASH if value is None else f"{value * 100:.0f} %"
    return DASH if value is None else str(value)


def rows_for_view(runs):
    """Рядки таблиці метрик (списки текстів у порядку TABLE_COLUMNS)."""
    return [[cell_text(key, run) for key, _title in TABLE_COLUMNS] for run in runs]


# ------------------------------------------------------------------- криві
def downsample(xs, ys, max_points=500):
    """Не більше max_points точок, рівномірно; перша й остання точки завжди лишаються.
    Значення не усереднюються — кожна точка точно та, що була в повній кривій."""
    n = len(xs)
    if n <= max_points:
        return list(xs), list(ys)
    idx = sorted({round(i * (n - 1) / (max_points - 1)) for i in range(max_points)})
    return [xs[i] for i in idx], [ys[i] for i in idx]


def _raw_series(run, metric):
    if metric in _EPISODE_CURVES:
        ys = list(run["metrics"][_EPISODE_CURVES[metric]])
        return list(range(1, len(ys) + 1)), ys
    if metric == "optimal":
        v_star = run["env"].get("optimal_return")
        xs, ys = [], []
        for e in run.get("greedy_eval", []):
            xs.append(e["episode"])
            ok = e["reaches_finish"] and v_star
            ys.append(e["path_return"] / v_star if ok else 0.0)
        return xs, ys
    raise ValueError(f"невідома метрика {metric!r} (можливі: {', '.join(METRICS)})")


def run_series(run, metric, window=1, max_points=500):
    """Крива запуску: (номери епізодів, значення). Спершу згладжування (ковзне середнє), потім проріджування."""
    xs, ys = _raw_series(run, metric)
    if window > 1:
        ys = moving_average(ys, window)
    return downsample(xs, ys, max_points)


# ------------------------------------------------------------------ кольори
def assign_colors(run_ids):
    """Колір кожного запуску. Бажаний колір залежить лише від run_id (той самий запуск — той самий колір
    у будь-якому сеансі); якщо в цьому виборі він уже зайнятий — береться наступний вільний."""
    colors, used = {}, set()
    for run_id in run_ids:
        start = zlib.crc32(run_id.encode("utf-8")) % len(PALETTE)
        for k in range(len(PALETTE)):
            color = PALETTE[(start + k) % len(PALETTE)]
            if color not in used:
                break
        else:                       # запусків більше, ніж кольорів — повтор неминучий
            color = PALETTE[start]
        colors[run_id] = color
        used.add(color)
    return colors


# -------------------------------------------------------------- попередження
def mismatch_banner(runs):
    """Текст попередження, якщо запуски непорівнянні (різні карти чи γ); порожній рядок, якщо все гаразд."""
    if len(runs) < 2:
        return ""
    return " ".join(compare(runs)["warnings"])


# --------------------------------------------------------------- групування
def group_series(runs, metric, window=1, max_points=500):
    """«Групувати за seed»: запуски однакової конфігурації (агент, карта, γ, параметри) — одна лінія.

    Повертає список груп у порядку першої появи: {"key", "label", "n_runs", "run_ids", "xs", "mean", "low", "high"},
    де low/high = середнє ∓ вибіркове σ (для одного запуску σ = 0, тож смуги немає).
    """
    groups = {}
    for run in runs:
        groups.setdefault(_config_key(run), []).append(run)
    result = []
    for key, items in groups.items():
        curves = [_raw_series(r, metric) for r in items]
        smooth = [moving_average(ys, window) if window > 1 else ys for _xs, ys in curves]
        xs = max((c[0] for c in curves), key=len)
        mean, low, high = [], [], []
        for i in range(len(xs)):
            vals = [c[i] for c in smooth if i < len(c)]
            m = sum(vals) / len(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            mean.append(m)
            low.append(m - sd)
            high.append(m + sd)
        idx_xs, _ = downsample(list(range(len(xs))), list(range(len(xs))), max_points)
        first = items[0]
        n = len(items)
        label = f"{AGENT_LABELS.get(first['agent'], first['agent'])} · {_map_text(first)} · {n} seed"
        result.append({"key": "|".join(str(k) for k in key), "label": label, "n_runs": n,
                       "run_ids": [r["run_id"] for r in items],
                       "xs": [xs[i] for i in idx_xs], "mean": [mean[i] for i in idx_xs],
                       "low": [low[i] for i in idx_xs], "high": [high[i] for i in idx_xs]})
    return result


# ----------------------------------------------------------------- підписи
def run_label(run):
    """Короткий підпис лінії в легенді: «Таблиця · 7×6 · seed 3»."""
    seed = "без seed" if run["seed"] is None else f"seed {run['seed']}"
    return f"{AGENT_LABELS.get(run['agent'], run['agent'])} · {_map_text(run)} · {seed}"


def _when(run):
    """Дата й час запуску (місцевий час — з run_id; інакше з поля created)."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})", run["run_id"])
    if m:
        _y, mo, d, h, mi = m.groups()
        return f"{d}.{mo} {h}:{mi}"
    try:
        return datetime.fromisoformat(run.get("created", "")).strftime("%d.%m %H:%M")
    except ValueError:
        return ""


def list_label(run):
    """Рядок у списку запусків: «19.11 18:24 · Таблиця · 7×6 · seed 3 · підпис»."""
    parts = [_when(run), run_label(run)]
    if run.get("note"):
        parts.append(run["note"])
    return " · ".join(p for p in parts if p)
