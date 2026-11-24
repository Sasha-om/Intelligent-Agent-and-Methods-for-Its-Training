"""Умова завдання для показу: текст + формула LaTeX окремим рядком (система — з фігурною дужкою, дроби — рискою).

question_view(item) → {"lead": текст перед формулою, "formula": LaTeX, "tail": текст після}.
Формула будується з тих самих params, що й текст умови (core/item_templates.py), тому числа збігаються —
це перевіряє tests/test_item_latex.py для всіх завдань банку. Текст умови в банку не змінюється.
"""
from core import latex as L
from core.item_templates import _quad_coefs

WHAT = {"up": "зростання", "down": "спадання"}


def lin(root: object) -> str:
    """«x − r» без дужок (для знаменника й кореня): lin(3) → «x - 3», lin(-2) → «x + 2», lin(0) → «x»."""
    text = L.factor(root)
    return text[1:-1] if text.startswith("(") else text


def quad_expr(p: dict) -> str:
    """Парабола так, як в умові: «-(x + 2)^{2} - 3» або «2x^{2} - 4x + 1»."""
    if "h" not in p:
        return L.poly([p["a"], p["b"], p["c"]])
    text = ("" if p.get("a", 1) > 0 else "-") + f"{L.factor(p['h'])}^{{2}}"
    k = p["k"]
    return text + (f" {'+' if k > 0 else '-'} {abs(k)}" if k else "")


def domain_expr(p: dict) -> str:
    """Вираз функції в задачі на область визначення."""
    lv = p["level"]
    if lv == -2:
        return rf"\frac{{1}}{{{lin(p['a'])}}}"
    if lv == -1:
        return rf"\sqrt{{{lin(p['a'])}}}"
    if lv == 0:
        return rf"\sqrt{{{L.num(p['m'])} - {p['k']}x}}"
    if lv == 1:
        return rf"\sqrt{{{lin(p['a'])}}} + \frac{{1}}{{{lin(p['c'])}}}"
    return rf"\sqrt{{{L.poly([1, -(p['r1'] + p['r2']), p['r1'] * p['r2']])}}}"


def linear_ineq_sides(p: dict) -> tuple:
    """Ліва і права частини лінійної нерівності (LaTeX)."""
    lv = p["level"]
    if lv == -2:
        return L.poly([1, p["m"]]), L.num(p["n"])
    if lv == -1:
        return f"{p['k']}x", L.num(p["n"])
    if lv == 0:
        return f"{L.num(p['m'])} - {p['k']}x", L.num(p["n"])
    if lv == 1:
        return f"{p['a']}{L.factor(p['p'])}", L.poly([p["b"], p["c"]])
    left = rf"\frac{{{lin(p['p'])}}}{{{p['m']}}} - \frac{{{lin(-p['q'])}}}{{{p['n']}}}"
    return left, L.num(p["r"])


def quadratic_ineq_left(p: dict) -> str:
    """Ліва частина квадратної нерівності."""
    lv, r1, r2 = p["level"], p["r1"], p["r2"]
    if lv in (-2, -1):
        return "x^{2}"
    if lv == 0:
        return L.factor(r1) + L.factor(r2)
    lead = p["lead"]
    return L.poly([lead, -lead * (r1 + r2), lead * r1 * r2])


def quadratic_ineq_right(p: dict) -> str:
    """Права частина квадратної нерівності."""
    return str(p["r2"] ** 2) if p["level"] in (-2, -1) else "0"


def fraction_expr(p: dict) -> str:
    """Ліва частина для методу інтервалів: добуток або дріб (множник «x» першим)."""
    num = "".join(L.factor(r) for r in sorted(p["num"], key=lambda r: r != 0))
    if not p["den"]:
        return num
    den = "".join(L.factor(r) for r in p["den"])
    den = den[1:-1] if len(p["den"]) == 1 and den.startswith("(") else den
    num = num[1:-1] if len(p["num"]) == 1 and num.startswith("(") else num
    return rf"\frac{{{num}}}{{{den}}}"


def system_line(q: dict) -> str:
    """Одна нерівність системи."""
    if q["kind"] == "sq":
        return f"x^{{2}} {L.OPS[q['op']]} {q['n'] ** 2}"
    return f"{L.poly([q['a'], q['b']])} {L.OPS[q['op']]} {L.num(q['c'])}"


def system_expr(p: dict) -> str:
    """Система з великою фігурною дужкою, нерівності одна під одною."""
    return r"\begin{cases} " + r" \\ ".join(system_line(q) for q in p["ineqs"]) + r" \end{cases}"


# ----------------------------------------------------------------------------- умови за підтемами
def _view_domain(p: dict) -> tuple:
    """Область визначення: «Знайдіть …» + y = вираз."""
    return "Знайдіть область визначення функції", f"y = {domain_expr(p)}", ""


def _view_range(p: dict) -> tuple:
    """Область значень параболи."""
    return "Знайдіть область значень функції", f"y = {quad_expr(p)}", ""


def _view_monotonic(p: dict) -> tuple:
    """Як змінюється лінійна функція / проміжок зростання чи спадання параболи."""
    if p["level"] == -2:
        return "Як змінюється функція", f"y = {L.poly([p['k'], p['b']])}", "?"
    return f"Укажіть проміжок {WHAT[p['ask']]} функції", f"y = {quad_expr(p)}", ""


def _view_linear_func(p: dict) -> tuple:
    """П'ять задач про лінійну функцію."""
    lv = p["level"]
    if lv == -2:
        return "Обчисліть значення функції", f"y = {L.poly([p['k'], p['b']])}", f"при $x = {L.num(p['x0'])}$."
    if lv == -1:
        return "Знайдіть нуль функції", f"y = {L.poly([p['k'], p['b']])}", ""
    if lv == 0:
        a, b = (L.point(x, p["k"] * x + p["b"]) for x in (p["x1"], p["x2"]))
        return "Графік функції $y = kx + b$ проходить через точки", rf"A{a},\quad B{b}", "Знайдіть $k$."
    if lv == 1:
        f1, f2 = (L.poly([k, p["y0"] - k * p["x0"]]) for k in (p["k1"], p["k2"]))
        return ("Знайдіть координати точки перетину графіків функцій", rf"y = {f1} \quad\text{{і}}\quad y = {f2}", "")
    return ("Задайте формулою лінійну функцію, графік якої паралельний прямій", f"y = {L.poly([p['k'], p['c']])}",
            f"і проходить через точку $M{L.point(p['x0'], p['y0'])}$.")


def _view_quadratic_func(p: dict) -> tuple:
    """П'ять задач про параболу."""
    lv = p["level"]
    if lv == -2:
        return "Знайдіть координати вершини параболи", f"y = {quad_expr({'h': p['h'], 'k': p['k']})}", ""
    if lv == -1:
        return "Обчисліть значення функції", f"y = {L.poly([1, p['b'], p['c']])}", f"при $x = {L.num(p['x0'])}$."
    if lv == 0:
        return "Знайдіть координати вершини параболи", f"y = {L.poly([1, p['b'], p['c']])}", ""
    if lv == 1:
        return "Знайдіть нулі функції", f"y = {L.poly([1, -(p['r1'] + p['r2']), p['r1'] * p['r2']])}", ""
    return ("При якому значенні $c$ парабола", f"y = {L.poly([1, p['p'], 0])} + c",
            "має з віссю абсцис рівно одну спільну точку?")


def _view_linear_ineq(p: dict) -> tuple:
    """Лінійна нерівність."""
    left, right = linear_ineq_sides(p)
    return "Розв'яжіть нерівність", f"{left} {L.OPS[p['op']]} {right}", ""


def _view_quadratic_ineq(p: dict) -> tuple:
    """Квадратна нерівність."""
    return "Розв'яжіть нерівність", f"{quadratic_ineq_left(p)} {L.OPS[p['op']]} {quadratic_ineq_right(p)}", ""


def _view_interval_method(p: dict) -> tuple:
    """Метод інтервалів: добуток або дріб."""
    return "Розв'яжіть методом інтервалів нерівність", f"{fraction_expr(p)} {L.OPS[p['op']]} 0", ""


def _view_ineq_system(p: dict) -> tuple:
    """Система нерівностей з фігурною дужкою."""
    if p["ask"] == "count":
        return "Скільки цілих розв'язків має система нерівностей", system_expr(p), "?"
    return "Розв'яжіть систему нерівностей", system_expr(p), ""


VIEWS = {"domain": _view_domain, "range": _view_range, "monotonic": _view_monotonic,
         "linear_func": _view_linear_func, "quadratic_func": _view_quadratic_func,
         "linear_ineq": _view_linear_ineq, "quadratic_ineq": _view_quadratic_ineq,
         "interval_method": _view_interval_method, "ineq_system": _view_ineq_system}


def question_view(item: dict) -> dict:
    """Умова для показу: {"lead": текст (може містити $…$), "formula": LaTeX, "tail": текст}."""
    lead, formula, tail = VIEWS[item["template"]](item["params"])
    if tail == "?":                       # «Скільки … має система нерівностей?» — знак питання при тексті
        lead, tail = lead + "?", ""
    return {"lead": lead, "formula": formula, "tail": tail}


def option_markdown(text: str) -> str:
    """Варіант відповіді для радіокнопки: математика — у «$…$», слова — текстом."""
    return L.markdown(text)
