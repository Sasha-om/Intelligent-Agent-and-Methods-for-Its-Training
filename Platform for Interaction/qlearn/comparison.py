"""Порівняння збережених запусків (схема v1): таблиця метрик, середнє ± σ по seed, криві навчання, CSV для Excel.

    rows/warnings = compare(runs)["rows"], compare(runs)["warnings"]
    groups        = aggregate(runs)        # одна група = агент + карта + γ + параметри; seed-и всередині
    curve         = run_curve(run, "reward", window=20)
    write_csv(path, rows, ROW_FIELDS)

Порівнювати має сенс лише запуски на одній карті (env.fingerprint) і з однаковим γ — інакше compare попереджає.
"""
import json
import statistics
from pathlib import Path

ROW_FIELDS = ["run_id", "agent", "seed", "gamma", "map", "fingerprint", "episodes", "first_success_episode",
              "convergence_episode", "final_optimal", "return_ratio", "return_gap", "final_path_cost", "cost_ratio",
              "mean_reward_last_50", "success_rate_last_50", "total_env_steps", "duration_sec", "note"]
AGG_METRICS = ["first_success_episode", "convergence_episode", "return_ratio", "return_gap", "final_path_cost",
               "cost_ratio", "mean_reward_last_50", "success_rate_last_50", "total_env_steps", "duration_sec"]
CURVES = {"reward": "reward_per_episode", "steps": "steps_per_episode", "success": "reached_goal"}


def _map_label(run):
    m = run["env"]["map"]
    return f"{m['cols']}x{m['rows']}"


def row_of(run):
    """Один рядок таблиці порівняння для запуску."""
    s = run["summary"]
    base = {"run_id": run["run_id"], "agent": run["agent"], "seed": run["seed"], "gamma": run["gamma"],
            "map": _map_label(run), "fingerprint": run["env"]["fingerprint"], "note": run.get("note", "")}
    return {f: base[f] if f in base else s.get(f) for f in ROW_FIELDS}


def compare(runs):
    """Таблиця (рядок на запуск) і попередження, якщо запуски непорівнянні."""
    warnings = []
    maps = {(r["env"]["fingerprint"], _map_label(r)) for r in runs}
    if len(maps) > 1:
        sizes = sorted(label for _fp, label in maps)
        same = " — розмір однаковий, але різні перешкоди, старт або фініш" if len(set(sizes)) < len(sizes) else ""
        warnings.append(f"Запуски на різних картах ({len(maps)}: {', '.join(sizes)}{same}) — "
                        "порівнюйте метрики в межах однієї карти.")
    gammas = sorted({r["gamma"] for r in runs})
    if len(gammas) > 1:
        warnings.append(f"Запуски з різними γ ({', '.join(str(g) for g in gammas)}) — оптимальні шляхи можуть відрізнятися (INS-008).")
    return {"rows": [row_of(r) for r in runs], "warnings": warnings}


def _config_key(run):
    params = {k: v for k, v in run["params"].items() if k != "seed"}
    return (run["agent"], run["env"]["fingerprint"], run["gamma"], json.dumps(params, sort_keys=True))


def _mean_std(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    return sum(vals) / len(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def aggregate(runs):
    """Групи однакових конфігурацій (у порядку першої появи): середнє й вибіркове σ кожної метрики по seed."""
    groups = {}
    for run in runs:
        groups.setdefault(_config_key(run), []).append(run)
    result = []
    for (agent, fp, gamma, _params), items in groups.items():
        n = len(items)
        g = {"agent": agent, "map": _map_label(items[0]), "fingerprint": fp, "gamma": gamma, "n_runs": n,
             "seeds": [r["seed"] for r in items],
             "converged": f"{sum(r['summary']['convergence_episode'] is not None for r in items)} з {n}",
             "final_optimal": f"{sum(bool(r['summary']['final_optimal']) for r in items)} з {n}"}
        for m in AGG_METRICS:
            g[f"{m}_mean"], g[f"{m}_std"] = _mean_std([r["summary"].get(m) for r in items])
        result.append(g)
    return result


def aggregate_fields(groups):
    """Порядок стовпців для CSV підсумку."""
    head = ["agent", "map", "fingerprint", "gamma", "n_runs", "seeds", "converged", "final_optimal"]
    return head + [f"{m}_{k}" for m in AGG_METRICS for k in ("mean", "std")]


# ----------------------------------------------------------------- криві
def moving_average(values, window):
    """Ковзне середнє «назад»: точка i — середнє останніх window значень до i включно."""
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def mean_curve(curves):
    """Поелементне середнє кривих різної довжини (у кожній точці — серед тих, що до неї дійшли)."""
    length = max((len(c) for c in curves), default=0)
    out = []
    for i in range(length):
        vals = [c[i] for c in curves if i < len(c)]
        out.append(sum(vals) / len(vals))
    return out


def run_curve(run, metric, window=1):
    """Крива навчання запуску: "reward", "steps" або "success" (0/1), за потреби згладжена."""
    if metric not in CURVES:
        raise ValueError(f"невідома крива {metric!r} (можливі: {', '.join(CURVES)})")
    values = list(run["metrics"][CURVES[metric]])
    return moving_average(values, window) if window > 1 else values


# ------------------------------------------------------------------ CSV
def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return "так" if v else "ні"
    if isinstance(v, float):
        return ("%.6f" % v).rstrip("0").rstrip(".").replace(".", ",")
    if isinstance(v, (list, tuple)):
        return " ".join(_fmt(x) for x in v)
    return str(v).replace(";", ",").replace("\r", " ").replace("\n", " ")


def to_csv(rows, fields):
    """CSV для Excel (як експорт таблиці Q): роздільник «;», десяткова кома, рядки через CRLF."""
    lines = [";".join(fields)] + [";".join(_fmt(r.get(f)) for f in fields) for r in rows]
    return "\r\n".join(lines)


def write_csv(path, rows, fields):
    """Записати CSV з BOM (utf-8-sig), щоб Excel правильно показав кирилицю."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(to_csv(rows, fields))
    return path
