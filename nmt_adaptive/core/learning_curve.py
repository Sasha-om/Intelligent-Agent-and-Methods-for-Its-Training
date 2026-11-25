"""Крива навчання з тренувальних відповідей (етап 12).

Ідея: якщо учень вчиться, то з кожною наступною спробою з тієї самої теми він відповідає правильно частіше,
ніж очікувалося б за його знаннями ДО тренування (θ̂ з вхідного тесту) і складністю завдання.

    залишок = правильно (0/1) − p(θ̂_вхід, b)       p — та сама 3PL, що в оцінювачі
    спроба  = котрий раз ця тема трапилась у тренуванні (1, 2, 3, …)

Швидкість навчання — нахил прямої «залишок від спроби» (відсоткових пунктів за спробу).
Різниця складності завдань у режимах уже врахована в p, тому режими можна порівнювати.
Обмеження: повторення слабкої теми (review) обирає теми після помилки — частина зростання може бути
поверненням до середнього, а не навчанням; це перевіряється на симуляції.
"""
import numpy as np

import config
from core.ability import AbilityEstimator
from core.stats import t_critical


def pre_estimator(test_answers: list) -> AbilityEstimator:
    """θ̂ після вхідного тесту: test_answers — [(підтема, рівень, правильно?)] у порядку відповідей."""
    estimator = AbilityEstimator()
    for subtopic, level, correct in test_answers:
        estimator.update(subtopic, float(level), bool(correct))
    return estimator


def session_points(train_answers: list, estimator: AbilityEstimator) -> list:
    """Точки однієї сесії: [(спроба, правильно 0/1, очікувано p)]; train_answers — [(підтема, рівень, правильно?)]."""
    seen, points = {}, []
    for subtopic, level, correct in train_answers:
        seen[subtopic] = seen.get(subtopic, 0) + 1
        points.append((seen[subtopic], float(bool(correct)), estimator.p_hat_blended(subtopic, float(level))))
    return points


def curve_rows(sessions: list, cap: int = config.CURVE_MAX_OPPORTUNITY) -> list:
    """Крива: для спроби 1…cap (останнє — «cap і більше») частка правильних, очікувана, залишок, кількість."""
    rows = []
    for k in range(1, cap + 1):
        part = [(ok, p) for points in sessions for n, ok, p in points if min(n, cap) == k]
        if part:
            ok, p = np.array(part).T
            rows.append({"спроба": k, "відповідей": len(part), "правильно": float(ok.mean()),
                         "очікувано": float(p.mean()), "залишок": float((ok - p).mean())})
    return rows


def slope(sessions: list, cap: int = config.CURVE_MAX_OPPORTUNITY) -> float | None:
    """Нахил «залишок від спроби» (метод найменших квадратів) — частка за одну спробу; None, якщо спроба одна."""
    points = [(min(n, cap), ok - p) for s in sessions for n, ok, p in s]
    if len(points) < 2:
        return None
    x, y = np.array(points).T
    if np.ptp(x) == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def slope_ci(sessions: list, rng: np.random.Generator, alpha: float = config.PILOT_ALPHA,
             repeats: int = config.PILOT_BOOTSTRAP) -> tuple:
    """Нахил і бутстреп-ДІ (перевибірка УЧНІВ, а не окремих відповідей): (нахил, низ, верх)."""
    value = slope(sessions)
    if value is None or len(sessions) < 2:
        return value, None, None
    picks = (slope([sessions[i] for i in rng.integers(0, len(sessions), len(sessions))]) for _ in range(repeats))
    boot = [b for b in picks if b is not None]
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return value, float(low), float(high)


def slope_difference(a: list, b: list, rng: np.random.Generator, alpha: float = config.PILOT_ALPHA,
                     repeats: int = config.PILOT_BOOTSTRAP) -> tuple:
    """Різниця нахилів «a − b» з бутстреп-ДІ (учні в кожній групі перевибираються окремо)."""
    sa, sb = slope(a), slope(b)
    if sa is None or sb is None or len(a) < 2 or len(b) < 2:
        return None, None, None
    boot = []
    for _ in range(repeats):
        da = slope([a[i] for i in rng.integers(0, len(a), len(a))])
        db = slope([b[i] for i in rng.integers(0, len(b), len(b))])
        if da is not None and db is not None:
            boot.append(da - db)
    low, high = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return sa - sb, float(low), float(high)


def session_sums(points: list, cap: int = config.CURVE_MAX_OPPORTUNITY) -> np.ndarray:
    """Суми однієї сесії для швидкого нахилу: [кількість, Σx, Σy, Σx², Σxy], x — спроба, y — залишок."""
    if not points:
        return np.zeros(5)
    x = np.array([min(n, cap) for n, _, _ in points], dtype=float)
    y = np.array([ok - p for _, ok, p in points])
    return np.array([len(x), x.sum(), y.sum(), (x * x).sum(), (x * y).sum()])


def slope_from_sums(sums: np.ndarray) -> float | None:
    """Нахил МНК із сум (той самий, що slope(), але без перебору точок)."""
    n, sx, sy, sxx, sxy = sums
    denominator = n * sxx - sx * sx
    return None if n < 2 or denominator <= 1e-12 else float((n * sxy - sx * sy) / denominator)


def jackknife(sums: list) -> tuple:
    """Нахил групи і його стандартна похибка «складаним ножем» (по одному учню виключаємо): (нахил, SE)."""
    total = np.sum(sums, axis=0)
    value = slope_from_sums(total)
    k = len(sums)
    if value is None or k < 3:
        return value, None
    leave = [slope_from_sums(total - s) for s in sums]
    leave = np.array([v for v in leave if v is not None])
    return value, float(np.sqrt((k - 1) / k * ((leave - leave.mean()) ** 2).sum()))


def difference_test(a: list, b: list, alpha: float = config.PILOT_ALPHA) -> tuple:
    """Різниця нахилів «a − b» (суми сесій) і чи вона значуща (|z| > t-критичного): (різниця, значущо?)."""
    (sa, ea), (sb, eb) = jackknife(a), jackknife(b)
    if sa is None or sb is None or ea is None or eb is None or ea + eb == 0:
        return None, False
    diff = sa - sb
    return diff, abs(diff) / np.hypot(ea, eb) > t_critical(len(a) + len(b) - 2, alpha)
