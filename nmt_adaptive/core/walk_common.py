"""Спільне для покрокового розбору: крок, підстановка числа у вираз, «перевірка на прикладі», крок «Відповідь».

Крок розбору — dict: {"title": "Крок 2. …", "text": пояснення (може містити $…$), "math": [рядки LaTeX],
"figure": SVG або None}. Нумерацію кроків ставить core/walkthrough.py.
"""
import math
import re
from fractions import Fraction

from core import figures
from core import latex as L
from core.math_utils import INF, complement, contains

CHECK, CROSS = r"\ \checkmark", r"\ \times"
OP_TEST = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
           "!=": lambda a, b: a != b, "=": lambda a, b: a == b}
OP_TEX = {**L.OPS, "!=": r"\ne", "=": "="}


def step(title: str, text: str = "", math_lines: list | None = None, figure: str | None = None) -> dict:
    """Один крок розбору."""
    return {"title": title, "text": text, "math": list(math_lines or []), "figure": figure}


def subst(expr: str, x: object) -> str:
    """Підставляє число замість x у LaTeX-вираз: «3x - 2» при x = −1 → «3\\cdot(-1) - 2»."""
    value = L.paren(x) if Fraction(x).denominator == 1 else f"\\left({L.num(x)}\\right)"
    expr = re.sub(r"(?<=[\d)])x", lambda _m: r"\cdot " + value, expr)
    return re.sub(r"(?<![a-z\\])x", lambda _m: value, expr)


def answer_step(answer: object, text: str = "") -> dict:
    """Останній крок: відповідь формулою і (для проміжків) на числовій прямій."""
    if isinstance(answer, list):
        figure = figures.number_line(answer) if answer else None
        return step("Відповідь", text, [L.intervals(answer)], figure)
    tex = L.num(answer) if isinstance(answer, (int, Fraction)) else L.from_text(str(answer))
    return step("Відповідь", text, [tex])


# ----------------------------------------------------------------------------- перевірка на прикладі
def _inside(iv: tuple) -> Fraction:
    """Зручне число всередині проміжку (ціле, якщо можна; ближче до нуля)."""
    lo, hi, lo_closed, hi_closed = iv
    if lo == hi:
        return Fraction(lo)
    if lo == -INF and hi == INF:
        return Fraction(0)
    if lo == -INF:
        return Fraction(math.ceil(hi) - 1) if not hi_closed or hi != int(hi) else Fraction(int(hi) - 1)
    if hi == INF:
        return Fraction(math.floor(lo) + 1)
    ints = [n for n in range(math.floor(lo), math.ceil(hi) + 1) if contains([iv], n)]
    return Fraction(min(ints, key=abs)) if ints else (Fraction(lo) + Fraction(hi)) / 2


def sample_points(answer: list, exclude: tuple = ()) -> tuple:
    """(число з відповіді, число не з відповіді) — для перевірки; None, якщо такого немає."""
    inside = [_inside(iv) for iv in answer]
    outside = [x for x in (_inside(iv) for iv in complement(answer)) if x not in exclude]
    pick = lambda xs: min(xs, key=abs) if xs else None      # noqa: E731
    return pick(inside), pick(outside)


def condition_lines(conditions: list, x: Fraction) -> tuple:
    """Рядки підстановки для всіх умов і чи виконуються всі. Умова: (ліва, op, права, f(x) → (л, п) або None)."""
    lines, ok = [], True
    for left, op, right, evaluate in conditions:
        values = evaluate(x)
        if values is None:
            lines.append(rf"{subst(left, x)}\ \text{{— не має змісту (ділення на нуль)}}{CROSS}")
            ok = False
            continue
        lv, rv = values
        good = OP_TEST[op](lv, rv)
        ok = ok and good
        if left.strip() == "x" and right == L.num(rv):       # умова «x < 3»: без зайвого «(−4) = −4»
            lines.append(rf"{L.num(lv)} {OP_TEX[op]} {L.num(rv)}" + (CHECK if good else CROSS))
            continue
        shown_right = L.num(rv) if right == L.num(rv) else f"{subst(right, x)} = {L.num(rv)}"
        lines.append(rf"{subst(left, x)} = {L.num(lv)},\quad {L.num(lv)} {OP_TEX[op]} {L.num(rv)}"
                     if shown_right == L.num(rv) else
                     rf"{subst(left, x)} = {L.num(lv)},\ \ {shown_right}:\ \ {L.num(lv)} {OP_TEX[op]} {L.num(rv)}")
        lines[-1] += CHECK if good else CROSS
    return lines, ok


def check_step(conditions: list, answer: list, exclude: tuple = ()) -> dict | None:
    """«Перевірка на прикладі»: число з відповіді задовольняє умову, число поза відповіддю — ні."""
    inside, outside = sample_points(answer, exclude)
    if inside is None:
        return None
    lines_in, _ = condition_lines(conditions, inside)
    text = f"Візьмемо число з відповіді, $x = {L.num(inside)}$, і підставимо в умову:"
    math_lines = lines_in
    if outside is not None:
        lines_out, _ = condition_lines(conditions, outside)
        math_lines = lines_in + [rf"\text{{А число не з відповіді, }} x = {L.num(outside)}:"] + lines_out
        text += " умова виконується. Для порівняння — число не з відповіді: умова не виконується."
    return step("Перевірка на прикладі", text, math_lines)
