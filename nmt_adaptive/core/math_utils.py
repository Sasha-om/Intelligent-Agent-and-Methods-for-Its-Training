"""Допоміжна математика для банку завдань: точні дроби, проміжки, метод інтервалів, гарний текст.

Навіщо: розв'язок кожного завдання рахується програмою з тих самих чисел, що й умова.
Щоб відповідь не мала похибок округлення, усі обчислення — у звичайних дробах (fractions.Fraction).

Проміжок (interval) — кортеж (lo, hi, lo_closed, hi_closed):
    (2, inf, True, False) означає [2; +∞).
Множина розв'язків — список проміжків, відсортований зліва направо.
"""
import math
from fractions import Fraction

INF = math.inf
MINUS = "−"                      # типографський мінус «−»
OP_TEXT = {">": ">", ">=": "≥", "<": "<", "<=": "≤"}
FLIP_OP = {">": "<", ">=": "<=", "<": ">", "<=": ">="}

Interval = tuple   # (lo, hi, lo_closed, hi_closed)


# ----------------------------------------------------------------------------- текст чисел і виразів
def fmt_num(x: object) -> str:
    """Число як текст: «3», «−3/2». Дріб завжди нескоротний, бо так працює Fraction."""
    x = Fraction(x)
    sign = MINUS if x < 0 else ""
    x = abs(x)
    body = str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"
    return sign + body


def fmt_bound(x: object) -> str:
    """Межа проміжку: число або ±∞."""
    if x == INF:
        return "+∞"
    if x == -INF:
        return MINUS + "∞"
    return fmt_num(x)


def fmt_point(x: object, y: object) -> str:
    """Точка на площині: «(2; −1)»."""
    return f"({fmt_num(x)}; {fmt_num(y)})"


def fmt_poly(coefs: list) -> str:
    """Многочлен за коефіцієнтами від старшого: [1, -5, 6] → «x² − 5x + 6»."""
    degree = len(coefs) - 1
    parts = []
    for i, c in enumerate(coefs):
        c = Fraction(c)
        power = degree - i
        if c == 0:
            continue
        var = ("", "x", "x²")[power]
        size = "" if (abs(c) == 1 and var) else fmt_num(abs(c))
        parts.append((c < 0, size + var))
    if not parts:
        return "0"
    negative, first = parts[0]
    text = (MINUS if negative else "") + first
    for negative, term in parts[1:]:
        text += f" {MINUS if negative else '+'} {term}"
    return text


def fmt_factor(root: object) -> str:
    """Множник (x − r): fmt_factor(3) → «(x − 3)», fmt_factor(-2) → «(x + 2)», fmt_factor(0) → «x»."""
    root = Fraction(root)
    if root == 0:
        return "x"
    return f"(x {MINUS if root > 0 else '+'} {fmt_num(abs(root))})"


def fmt_interval(iv: Interval) -> str:
    """Один проміжок: «[2; +∞)»; вироджений проміжок з однієї точки — «{2}»."""
    lo, hi, lo_closed, hi_closed = iv
    if lo == hi:
        return "{" + fmt_num(lo) + "}"
    left = "[" if lo_closed else "("
    right = "]" if hi_closed else ")"
    return f"{left}{fmt_bound(lo)}; {fmt_bound(hi)}{right}"


def fmt_intervals(ivs: list) -> str:
    """Множина розв'язків: «(−∞; 2] ∪ [5; +∞)»; порожня множина — «∅»."""
    if not ivs:
        return "∅"
    return " ∪ ".join(fmt_interval(iv) for iv in ivs)


# ----------------------------------------------------------------------------- метод інтервалів
def _probe(lo: object, hi: object) -> Fraction:
    """Точка всередині проміжку (lo; hi), у якій перевіряємо знак."""
    if lo == -INF:
        return Fraction(hi) - 1
    if hi == INF:
        return Fraction(lo) + 1
    return (Fraction(lo) + Fraction(hi)) / 2


def _sign_at(x: Fraction, roots: list) -> int:
    """Знак добутку (x − r1)(x − r2)… у точці x, яка не є коренем."""
    value = Fraction(1)
    for r in roots:
        value *= x - r
    return 1 if value > 0 else -1


def _condition_holds(sign: int, op: str) -> bool:
    """Чи підходить знак до нерівності: «> 0» і «≥ 0» хочуть плюс, «< 0» і «≤ 0» — мінус."""
    return sign > 0 if op in (">", ">=") else sign < 0


def sign_method(num_roots: list, den_roots: list, op: str) -> list:
    """Метод інтервалів для (x − a)(x − b)… / (x − c)… op 0 зі старшим коефіцієнтом +1.

    Корені чисельника входять у відповідь лише при нестрогій нерівності (≥, ≤);
    корені знаменника не входять ніколи (на нуль ділити не можна).
    Корені мають бути різні (кожен множник у першому степені).
    """
    points = sorted(set(num_roots) | set(den_roots))
    bounds = [-INF] + points + [INF]
    roots = list(num_roots) + list(den_roots)
    result = []
    for lo, hi in zip(bounds, bounds[1:]):
        if _condition_holds(_sign_at(_probe(lo, hi), roots), op):
            result.append((lo, hi, _is_closed(lo, num_roots, den_roots, op),
                           _is_closed(hi, num_roots, den_roots, op)))
    return result


def interval_signs(roots: list) -> list:
    """Знаки добутку (x − r1)(x − r2)… на проміжках між коренями, зліва направо: ['+', '−', '+']."""
    bounds = [-INF] + sorted(set(roots)) + [INF]
    return ["+" if _sign_at(_probe(lo, hi), roots) > 0 else MINUS for lo, hi in zip(bounds, bounds[1:])]


def _is_closed(point: object, num_roots: list, den_roots: list, op: str) -> bool:
    """Чи входить кінець проміжку у відповідь (див. sign_method)."""
    if point in (INF, -INF) or op in (">", "<"):
        return False
    return point in num_roots and point not in den_roots


# ----------------------------------------------------------------------------- лінійні нерівності
def solve_linear(a: object, b: object, c: object, op: str) -> tuple:
    """a·x + b op c  →  (новий знак, межа). Ділення на від'ємне a змінює знак нерівності."""
    a, b, c = Fraction(a), Fraction(b), Fraction(c)
    if a == 0:
        raise ValueError("коефіцієнт при x не може бути нулем")
    new_op = op if a > 0 else FLIP_OP[op]
    return new_op, (c - b) / a


def ray(op: str, bound: object) -> list:
    """Розв'язок нерівності x op bound як множина з одного променя."""
    bound = Fraction(bound)
    if op in (">", ">="):
        return [(bound, INF, op == ">=", False)]
    return [(-INF, bound, False, op == "<=")]


# ----------------------------------------------------------------------------- операції з множинами
def _intersect_two(p: Interval, q: Interval) -> Interval | None:
    """Перетин двох проміжків або None, якщо він порожній."""
    # Ліва межа — більша з двох; при рівних значеннях сильніша кругла дужка (open = True).
    lo, lo_open = max((p[0], not p[2]), (q[0], not q[2]))
    # Права межа — менша з двох; при рівних значеннях min обере False, тобто теж круглу дужку.
    hi, hi_closed = min((p[1], p[3]), (q[1], q[3]))
    lo_closed = not lo_open
    if lo < hi or (lo == hi and lo_closed and hi_closed):
        return (lo, hi, lo_closed, hi_closed)
    return None


def intersect(a: list, b: list) -> list:
    """Перетин двох множин розв'язків (як у системі нерівностей)."""
    out = [iv for p in a for q in b if (iv := _intersect_two(p, q)) is not None]
    return sorted(out, key=lambda iv: (iv[0], not iv[2]))


def complement(ivs: list) -> list:
    """Доповнення до всієї числової прямої (те, що НЕ входить у множину)."""
    out, lo, lo_closed = [], -INF, False
    for a, b, a_closed, b_closed in ivs:
        if lo < a or (lo == a and lo_closed and not a_closed):
            out.append((lo, a, lo_closed, not a_closed))
        lo, lo_closed = b, not b_closed
    if lo < INF:
        out.append((lo, INF, lo_closed, False))
    return out


def union(a: list, b: list) -> list:
    """Об'єднання двох множин: доповнення перетину доповнень."""
    return complement(intersect(complement(a), complement(b)))


def mirror(ivs: list) -> list:
    """Дзеркало x → −x (так виглядає відповідь, якщо переплутати знак)."""
    return [(-b, -a, bc, ac) for a, b, ac, bc in reversed(ivs)]


def shift(ivs: list, d: int) -> list:
    """Зсунути всі скінченні межі на d (для «майже правильних» неправильних варіантів)."""
    return [(a + d if a != -INF else a, b + d if b != INF else b, ac, bc) for a, b, ac, bc in ivs]


def toggle_closed(ivs: list) -> list:
    """Поміняти круглі й квадратні дужки на скінченних межах (типова помилка з кінцями)."""
    return [(a, b, (not ac) if a != -INF else ac, (not bc) if b != INF else bc) for a, b, ac, bc in ivs]


def contains(ivs: list, x: object) -> bool:
    """Чи належить число x множині ivs (з урахуванням дужок на кінцях)."""
    for a, b, ac, bc in ivs:
        left_ok = a < x or (ac and a == x)
        right_ok = x < b or (bc and b == x)
        if left_ok and right_ok:
            return True
    return False


def count_integers(ivs: list) -> int | None:
    """Скільки цілих чисел у множині; None, якщо їх нескінченно багато."""
    total = 0
    for a, b, ac, bc in ivs:
        if a == -INF or b == INF:
            return None
        first = math.ceil(a) if (ac or a != math.ceil(a)) else math.ceil(a) + 1
        last = math.floor(b) if (bc or b != math.floor(b)) else math.floor(b) - 1
        total += max(0, last - first + 1)
    return total
