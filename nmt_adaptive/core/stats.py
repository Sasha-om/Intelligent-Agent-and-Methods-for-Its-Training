"""Статистика для пілоту (етап 10): t-критичні, bootstrap-ДІ, Hedges g, ANCOVA. Без streamlit.

Лежить у core/, бо нею користуються і симуляція (sim/pilot_sim.py), і застосунок (app/pilot_storage.py, звіт).
"""
from statistics import NormalDist

import numpy as np

import config


def t_critical(df: int, alpha: float) -> float:
    """Критичне значення t (двобічне): точна таблиця для df ≤ 30 (α = 0,05 і 0,025), далі — Корніш — Фішер."""
    df = max(int(df), 1)
    table = config.T_TABLE.get(round(alpha, 4))
    if table is not None and df <= len(table):
        return table[df - 1]
    z = NormalDist().inv_cdf(1 - alpha / 2)
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


def _percentiles(boots: np.ndarray, alpha: float) -> tuple:
    """Межі перцентильного ДІ рівня 1 − alpha."""
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def mean_ci(values: list, rng: np.random.Generator, n_boot: int = config.PILOT_BOOTSTRAP,
            alpha: float = 0.05) -> tuple:
    """Середнє і ДІ рівня 1 − alpha (bootstrap). Порожньо → (None, None, None)."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return None, None, None
    boots = data[rng.integers(0, data.size, (n_boot, data.size))].mean(axis=1)
    return (float(data.mean()), *_percentiles(boots, alpha))


def diff_ci(a: list, b: list, rng: np.random.Generator, n_boot: int = config.PILOT_BOOTSTRAP,
            alpha: float = 0.05) -> tuple:
    """Різниця середніх a − b (незалежні групи) і ДІ рівня 1 − alpha (bootstrap кожної групи окремо)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    boot_a = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    boot_b = b[rng.integers(0, b.size, (n_boot, b.size))].mean(axis=1)
    return (float(a.mean() - b.mean()), *_percentiles(boot_a - boot_b, alpha))


def hedges_g(a: list, b: list) -> float | None:
    """Розмір ефекту Hedges g (різниця середніх у стандартних відхиленнях з поправкою на малу вибірку)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 2 or b.size < 2:
        return None
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    if pooled == 0:
        return None
    return float((a.mean() - b.mean()) / pooled * (1 - 3 / (4 * (a.size + b.size) - 9)))


def ancova(pre: list, post: list, is_target: list, alpha: float = config.PILOT_ALPHA) -> dict | None:
    """ANCOVA: вихід = β0 + β1·вхід + β2·[режим = цільовий]. β2 — різниця режимів «при однаковому вході».

    Повертає {"effect", "low", "high", "t", "df", "significant"} (ДІ рівня 1 − alpha) або None,
    якщо даних замало (< 4) чи режим один. significant — |t| > t_critical (двобічно).
    """
    y, pre, grp = (np.asarray(v, dtype=float) for v in (post, pre, is_target))
    if y.size < 4 or grp.min() == grp.max():
        return None
    x = np.column_stack([np.ones_like(y), pre, grp])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    df = y.size - 3
    sigma2 = float(((y - x @ coef) ** 2).sum()) / max(df, 1)
    se = float(np.sqrt(sigma2 * np.linalg.pinv(x.T @ x)[2, 2]))
    crit = t_critical(df, alpha)
    t = float(coef[2] / se) if se > 0 else float("inf") * float(np.sign(coef[2]))
    return {"effect": float(coef[2]), "low": float(coef[2] - crit * se), "high": float(coef[2] + crit * se),
            "t": t, "df": df, "significant": bool(abs(t) > crit)}
