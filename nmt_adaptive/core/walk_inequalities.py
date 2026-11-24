"""Покроковий розбір нерівностей: лінійні, квадратні, метод інтервалів, системи.

Кожна walk_<підтема>(p) повертає кроки; *_conditions(p) — умову задачі для «перевірки на прикладі»
(підставляємо число з відповіді й число не з відповіді). Кінцеву відповідь додає core/walkthrough.py.
"""
from fractions import Fraction

from core import figures
from core import latex as L
from core.item_latex import fraction_expr, linear_ineq_sides, quadratic_ineq_left, quadratic_ineq_right, system_line
from core.item_templates import _ineq_set, _linear_abc
from core.math_utils import FLIP_OP, INF, count_integers, fmt_num, interval_signs, intersect, sign_method
from core.walk_common import step

F = Fraction
LOOSE = (">=", "<=")


def divide_step(a: object, c: object, op: str, var_text: str = "x") -> tuple:
    """Останнє перетворення A·x op C → x op' C/A. Повертає (крок або None, op', межа)."""
    a, c = F(a), F(c)
    bound = c / a
    if a == 1:
        return None, op, bound
    new_op = op if a > 0 else FLIP_OP[op]
    same = a > 0 and c.denominator == 1 and bound.numerator == c and bound.denominator == a
    tail = "" if same else f" = {L.num(bound)}"
    line = rf"{var_text} {L.OPS[new_op]} \frac{{{L.num(c)}}}{{{L.num(a)}}}{tail}"
    if a > 0:
        return step("Ділимо на коефіцієнт біля x", f"Ділимо обидві частини на ${L.num(a)} > 0$ — **знак нерівності "
                    "не змінюється**:", [line]), new_op, bound
    return step("Ділимо на від'ємне число", f"Ділимо обидві частини на ${L.num(a)} < 0$. **Ділимо на від'ємне — "
                f"знак змінюється на протилежний** (${L.OPS[op]}$ стає ${L.OPS[new_op]}$):", [line]), new_op, bound


def ray_text(op: str, bound: object) -> str:
    """Пояснення дужки: «Межа входить — квадратна дужка» / «не входить — кругла»."""
    inside = "входить у відповідь (нерівність нестрога) — дужка **квадратна**" if op in LOOSE else \
        "не входить у відповідь (нерівність строга) — дужка **кругла**"
    return f"Число ${L.num(bound)}$ {inside}; біля $\\infty$ дужка завжди кругла."


# ============================================================================= 6. лінійні нерівності
def linear_ineq_conditions(p: dict) -> list:
    """Умова задачі як (ліва, op, права, f)."""
    left, right = linear_ineq_sides(p)
    lv = p["level"]
    funcs = {-2: lambda x: (x + p["m"], F(p["n"])), -1: lambda x: (p["k"] * x, F(p["n"])),
             0: lambda x: (p["m"] - p["k"] * x, F(p["n"])),
             1: lambda x: (p["a"] * (x - p["p"]), p["b"] * x + p["c"]),
             2: lambda x: ((x - p["p"]) / p["m"] - (x + p["q"]) / p["n"], F(p["r"]))}
    return [(left, p["op"], right, funcs[lv])]


def walk_linear_ineq(p: dict) -> list:
    """Лінійна нерівність: розкриваємо дужки/дроби, x — ліворуч, числа — праворуч, ділимо."""
    left, right = linear_ineq_sides(p)
    op, lv = p["op"], p["level"]
    steps = [step("Що робимо", "Мета — залишити $x$ сам ліворуч. Правила: доданок переносимо в іншу частину "
                  "**зі зміною знака**; ділимо на від'ємне число — **знак нерівності змінюється**.",
                  [f"{left} {L.OPS[op]} {right}"])]
    steps += {-2: _lin_move, -1: lambda q: [], 0: _lin_move, 1: _lin_brackets, 2: _lin_fractions}[lv](p)
    a, b, c = _linear_abc(p)
    if lv in (-2, 0):
        a, c = F(a), F(c) - b
    division, new_op, bound = divide_step(a, c, op)
    if division:
        steps.append(division)
    steps.append(step("Записуємо проміжком", ray_text(new_op, bound),
                      [rf"x {L.OPS[new_op]} {L.num(bound)}"]))
    return steps


def _lin_move(p: dict) -> list:
    """x + m op n  або  m − kx op n: переносимо число m праворуч."""
    m, n = p["m"], p["n"]
    x_part = "x" if p["level"] == -2 else f"-{p['k']}x"
    return [step("Переносимо число", f"Переносимо ${L.num(m)}$ у праву частину зі зміною знака:",
                 [rf"{x_part} {L.OPS[p['op']]} {L.num(n)} {'-' if m > 0 else '+'} {abs(m)}",
                  rf"{x_part} {L.OPS[p['op']]} {L.num(n - m)}"])]


def _lin_brackets(p: dict) -> list:
    """a(x − p) op bx + c: розкриваємо дужки, збираємо x ліворуч."""
    a, b, q, c, op = p["a"], p["b"], p["p"], p["c"], L.OPS[p["op"]]
    return [step("Розкриваємо дужки", "Множимо число перед дужкою на кожен доданок у дужках:",
                 [rf"{a}x {'-' if a * q > 0 else '+'} {abs(a * q)} {op} {L.poly([b, c])}"]),
            step("Збираємо x ліворуч, числа праворуч", "Переносимо зі зміною знака:",
                 [rf"{a}x {'-' if b > 0 else '+'} {abs(b)}x {op} {L.num(c)} {'+' if a * q > 0 else '-'} {abs(a * q)}",
                  rf"{L.poly([a - b, 0])} {op} {L.num(c + a * q)}"])]


def _lin_fractions(p: dict) -> list:
    """(x − p)/m − (x + q)/n op r: множимо на m·n, розкриваємо дужки, зводимо подібні."""
    m, n, q, r, op = p["m"], p["n"], p["q"], p["r"], L.OPS[p["op"]]
    pp = p["p"]
    return [step("Позбуваємося дробів", f"Множимо **обидві частини** на ${m * n}$ (спільний знаменник). Це додатне "
                 "число, тому знак нерівності не змінюється:",
                 [rf"{n}{L.factor(pp)} - {m}{L.factor(-q)} {op} {L.num(r * m * n)}"]),
            step("Розкриваємо дужки", "Увага: мінус перед другою дужкою змінює знаки **обох** доданків у ній:",
                 [rf"{n}x {'-' if n * pp > 0 else '+'} {abs(n * pp)} - {m}x {'-' if m * q > 0 else '+'} {abs(m * q)} "
                  rf"{op} {L.num(r * m * n)}"]),
            step("Зводимо подібні", "Збираємо $x$ ліворуч, числа — праворуч:",
                 [rf"{L.poly([n - m, 0])} {op} {L.num(r * m * n + n * pp + m * q)}"])]


# ============================================================================= 7. квадратні нерівності
def quadratic_ineq_conditions(p: dict) -> list:
    """Умова задачі як (ліва, op, права, f)."""
    r1, r2, lead = p["r1"], p["r2"], p["lead"]
    if p["level"] in (-2, -1):
        f = lambda x: (x * x, F(r2 * r2))        # noqa: E731
    else:
        f = lambda x: (lead * (x - r1) * (x - r2), F(0))     # noqa: E731
    return [(quadratic_ineq_left(p), p["op"], quadratic_ineq_right(p), f)]


def walk_quadratic_ineq(p: dict) -> list:
    """Квадратна нерівність: корені → парабола → знаки → потрібні проміжки."""
    r1, r2, lead, lv = p["r1"], p["r2"], p["lead"], p["level"]
    op = p["op"] if lead > 0 else FLIP_OP[p["op"]]
    steps = [step("План", "1) знайти корені тричлена; 2) подивитися, де парабола вище/нижче осі $x$; "
                  "3) вибрати потрібні проміжки.", [f"{quadratic_ineq_left(p)} {L.OPS[p['op']]} {quadratic_ineq_right(p)}"])]
    steps += _quad_roots(p)
    inside = op in ("<", "<=")
    want = "від'ємний (нижче осі)" if inside else "додатний (вище осі)"
    closed = op in LOOSE
    steps.append(step("Знаки на проміжках", "Гілки параболи $y = (x - x_1)(x - x_2)$ вгору: вона **вище осі зліва і "
                      f"справа від коренів** і **нижче між ними**. Потрібно, щоб вираз був {want}"
                      + (" або нуль — корені входять (зафарбовані точки)." if closed else
                         " — корені не входять (порожні точки)."),
                      figure=figures.number_line(sign_method([r1, r2], [], op), [(F(r1), closed), (F(r2), closed)],
                                                 ["+", "−", "+"])))
    steps.append(step("Вибираємо проміжки", "Беремо " + ("проміжок **між** коренями." if inside else
                                                         "**два крайні** проміжки (зліва і справа)."),
                      figure=figures.parabola(1, -(r1 + r2), r1 * r2, [(r1, 0, fmt_num(r1)), (r2, 0, fmt_num(r2))])))
    return steps


def _quad_roots(p: dict) -> list:
    """Корені: x² = n² → ±n; добуток → з множників; тричлен → дискримінант (при a < 0 спершу ділимо)."""
    r1, r2, lead, lv = p["r1"], p["r2"], p["lead"], p["level"]
    if lv in (-2, -1):
        return [step("Корені", f"Розв'язуємо $x^{{2}} = {r2 * r2}$: квадрат дорівнює ${r2 * r2}$ у двох чисел:",
                     [rf"x = -\sqrt{{{r2 * r2}}} = {r1}\quad\text{{або}}\quad x = \sqrt{{{r2 * r2}}} = {r2}"])]
    if lv == 0:
        return [step("Корені", "Добуток дорівнює нулю, коли **хоча б один множник** дорівнює нулю:",
                     [rf"{L.factor(r1)[1:-1] if r1 else 'x'} = 0\ \Rightarrow\ x_1 = {r1};\qquad "
                      rf"{L.factor(r2)[1:-1] if r2 else 'x'} = 0\ \Rightarrow\ x_2 = {r2}"])]
    out = []
    if lead < 0:
        out.append(step("Робимо коефіцієнт при x² додатним", f"Ділимо обидві частини на ${lead} < 0$ — **знак "
                        "змінюється на протилежний**:",
                        [rf"{L.poly([1, -(r1 + r2), r1 * r2])} {L.OPS[FLIP_OP[p['op']]]} 0"]))
    b, c = -(r1 + r2), r1 * r2
    d = b * b - 4 * c
    out.append(step("Корені через дискримінант", "", [rf"D = {L.paren(b)}^{{2}} - 4\cdot {L.paren(c)} = {d}",
                                                      rf"x_{{1,2}} = \frac{{{L.num(-b)} \pm {int(d ** 0.5)}}}{{2}}"
                                                      rf"\quad\Rightarrow\quad x_1 = {r1},\ \ x_2 = {r2}"]))
    return out


# ============================================================================= 8. метод інтервалів
def interval_conditions(p: dict) -> list:
    """Умова: дріб (добуток) op 0; у нулі знаменника — не визначено."""
    def f(x: Fraction) -> tuple | None:
        """Значення дробу в точці x і нуль; None — у нулі знаменника."""
        value = F(1)
        for r in p["num"]:
            value *= x - r
        for r in p["den"]:
            if x == r:
                return None
            value /= x - r
        return value, F(0)
    return [(fraction_expr(p), p["op"], "0", f)]


def walk_interval_method(p: dict) -> list:
    """Метод інтервалів: нулі → точки на прямій → знаки (правий «+», далі чергуються) → вибір."""
    num, den, op = p["num"], p["den"], p["op"]
    roots = sorted(num + den)
    closed = op in LOOSE
    points = [(F(r), closed and r in num) for r in roots]
    signs = [s.replace("-", "−") for s in interval_signs(roots)]
    want = "+" if op in (">", ">=") else "−"
    steps = [step("Ідея методу", "Добуток (дріб) змінює знак лише там, де якийсь множник дорівнює нулю. Тому "
                  "знаходимо ці точки, ділимо ними пряму на проміжки і дивимося на знак у кожному.",
                  [f"{fraction_expr(p)} {L.OPS[op]} 0"]),
             step("Нулі чисельника", "Кожен множник прирівнюємо до нуля:",
                  [r",\quad ".join(f"x = {r}" for r in num)])]
    if den:
        steps.append(step("Нулі знаменника", "На нуль ділити не можна — ці точки **завжди виколюємо** "
                          "(порожній кружечок), навіть якщо нерівність нестрога:", [r",\quad ".join(f"x \\ne {r}" for r in den)]))
    steps.append(step("Знаки на проміжках", f"Беремо число, більше за всі точки (наприклад, ${roots[-1] + 1}$): "
                      "усі множники додатні, тому праворуч «+». Кожен множник у першому степені, тож при переході "
                      "через кожну точку знак **змінюється**.",
                      figure=figures.number_line(sign_method(num, den, op), points, signs)))
    steps.append(step("Вибираємо проміжки", f"Нам потрібно ${L.OPS[op]} 0$ — беремо проміжки зі знаком «{want}»"
                      + (" разом із зафарбованими точками." if closed else ".")))
    return steps


# ============================================================================= 9. системи нерівностей
def system_conditions(p: dict) -> list:
    """Усі нерівності системи як умови."""
    out = []
    for q in p["ineqs"]:
        if q["kind"] == "sq":
            out.append((f"x^{{2}}", q["op"], str(q["n"] ** 2), lambda x, q=q: (x * x, F(q["n"] ** 2))))
        else:
            out.append((L.poly([q["a"], q["b"]]), q["op"], L.num(q["c"]), lambda x, q=q: (q["a"] * x + q["b"], F(q["c"]))))
    return out


def walk_ineq_system(p: dict) -> list:
    """Система: розв'язуємо кожну нерівність окремо, на прямій шукаємо СПІЛЬНУ частину."""
    parts = [_ineq_set(q) for q in p["ineqs"]]
    common = intersect(parts[0], parts[1])
    steps = [step("Що означає система", "Фігурна дужка означає **«і»**: потрібні $x$, які задовольняють **обидві** "
                  "нерівності одночасно. Розв'язуємо кожну окремо, потім шукаємо спільну частину.",
                  [r"\begin{cases} " + r" \\ ".join(system_line(q) for q in p["ineqs"]) + r" \end{cases}"])]
    for k, (q, part) in enumerate(zip(p["ineqs"], parts), 1):
        steps.append(step(f"{'Перша' if k == 1 else 'Друга'} нерівність", "", _system_part(q),
                          figures.number_line(part)))
    steps.append(step("Спільна частина", "Відповідь системи — там, де проміжки **перекриваються** (перетин):",
                      [L.intervals(common)], figures.number_line(common)))
    if p["ask"] == "count":
        steps.append(_count_step(common))
    return steps


def _system_part(q: dict) -> list:
    """Рядки розв'язання однієї нерівності системи."""
    if q["kind"] == "sq":
        n = q["n"]
        return [rf"x^{{2}} \le {n * n}\quad\Rightarrow\quad -{n} \le x \le {n}"]
    a, b, c, op = q["a"], q["b"], q["c"], q["op"]
    lines = [f"{L.poly([a, b])} {L.OPS[op]} {L.num(c)}"]
    if b:
        lines.append(f"{L.poly([a, 0])} {L.OPS[op]} {L.num(c)} {'-' if b > 0 else '+'} {abs(b)} = {L.num(c - b)}")
    new_op = op if a > 0 else FLIP_OP[op]
    note = r"\quad(\text{ділимо на від'ємне — знак змінюється})" if a < 0 else ""
    if a != 1:
        lines.append(rf"x {L.OPS[new_op]} \frac{{{L.num(c - b)}}}{{{a}}} = {L.num(F(c - b, a))}{note}")
    return lines


def _count_step(common: list) -> dict:
    """Перелічуємо цілі числа спільного проміжку (з урахуванням дужок)."""
    lo, hi, lo_closed, hi_closed = common[0]
    ints = [n for n in range(int(lo) - 1, int(hi) + 2)
            if (lo < n < hi) or (n == lo and lo_closed) or (n == hi and hi_closed)]
    return step("Рахуємо цілі числа", "Кругла дужка — кінець **не** береться, квадратна — береться. Цілі числа "
                "в проміжку:", [r"\{" + ";\\ ".join(str(n) for n in ints) + r"\}" + rf"\quad\Rightarrow\quad {len(ints)}"
                                ] if count_integers(common) == len(ints) else [])
