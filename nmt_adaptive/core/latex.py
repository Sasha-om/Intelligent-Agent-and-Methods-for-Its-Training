"""Математичний набір (LaTeX) для показу задач, варіантів і розбору — як у підручнику. Без streamlit.

Streamlit показує LaTeX через KaTeX: st.latex(формула) — окремим рядком; «$…$» у тексті й варіантах — у рядку.
Тут — лише «як записати»; що саме записати, вирішують core/item_latex.py і core/walkthrough.py.
"""
import re
from fractions import Fraction

from core.math_utils import INF

OPS = {">": ">", ">=": r"\ge", "<": "<", "<=": r"\le"}
WORD = re.compile(r"[А-Яа-яІіЇїЄєҐґ'’]+(?:[\s,]+[А-Яа-яІіЇїЄєҐґ'’]+)*")
NUMBER = re.compile(r"(\d+)/(\d+)")
SYMBOLS = (("−", "-"), ("∞", r"\infty "), ("∪", r"\cup "), ("∅", r"\varnothing "), ("≤", r"\le "),
           ("≥", r"\ge "), ("≠", r"\ne "), ("·", r"\cdot "), ("²", "^{2}"), ("₀", "_0"), ("₁", "_1"), ("₂", "_2"),
           (";", r";\,"))


def num(x: object) -> str:
    """Число: «3», «-3», «\\frac{7}{2}», «-\\frac{3}{2}»."""
    x = Fraction(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    return sign + (str(x.numerator) if x.denominator == 1 else rf"\frac{{{x.numerator}}}{{{x.denominator}}}")


def bound(x: object) -> str:
    """Межа проміжку: число або ±∞."""
    if x == INF:
        return r"+\infty"
    if x == -INF:
        return r"-\infty"
    return num(x)


def interval(iv: tuple) -> str:
    """Проміжок «[3;\\,5)»; точка — «\\{3\\}»."""
    lo, hi, lo_closed, hi_closed = iv
    if lo == hi:
        return rf"\{{{num(lo)}\}}"
    return f"{'[' if lo_closed else '('}{bound(lo)};\\,{bound(hi)}{']' if hi_closed else ')'}"


def intervals(ivs: list) -> str:
    """Множина розв'язків: «(-\\infty;\\,2] \\cup [5;\\,+\\infty)»; порожня — «\\varnothing»."""
    return r" \cup ".join(interval(iv) for iv in ivs) if ivs else r"\varnothing"


def point(x: object, y: object) -> str:
    """Точка «(2;\\,-1)»."""
    return f"({num(x)};\\,{num(y)})"


def poly(coefs: list, var: str = "x") -> str:
    """Многочлен від старшого коефіцієнта: [1, -5, 6] → «x^{2} - 5x + 6»."""
    degree, parts = len(coefs) - 1, []
    for i, c in enumerate(coefs):
        c, power = Fraction(c), degree - i
        if c == 0:
            continue
        name = ("", var, f"{var}^{{2}}")[power]
        size = "" if (abs(c) == 1 and name) else num(abs(c))
        parts.append((c < 0, size + name))
    if not parts:
        return "0"
    text = ("-" if parts[0][0] else "") + parts[0][1]
    return text + "".join(f" {'-' if neg else '+'} {term}" for neg, term in parts[1:])


def factor(root: object) -> str:
    """Множник (x − r): «(x - 3)», «(x + 2)», «x»."""
    root = Fraction(root)
    if root == 0:
        return "x"
    return f"(x {'-' if root > 0 else '+'} {num(abs(root))})"


def paren(x: object) -> str:
    """Число в дужках, якщо воно від'ємне: «3», «(-3)» — для підстановки «2·(-3)»."""
    return f"({num(x)})" if Fraction(x) < 0 else num(x)


def from_text(text: str) -> str:
    """Готовий математичний текст програми (варіант відповіді, «−2 і 3», «y = 2x − 3») → LaTeX."""
    text = NUMBER.sub(lambda m: rf"\dfrac{{{m.group(1)}}}{{{m.group(2)}}}", text)   # дріб повного розміру
    for plain, tex in SYMBOLS:
        text = text.replace(plain, tex)
    text = re.sub(r"\{([^{}]*\\dfrac[^{}]*|[-\d]+)\}", r"\\{\1\\}", text) if text.startswith("{") else text
    return text.replace("√", r"\sqrt")


def markdown(text: str) -> str:
    """Текст для st.markdown / варіанта радіокнопки: слова — звичайним текстом, математика — у «$…$»."""
    if not text:
        return text
    out, last = [], 0
    for match in WORD.finditer(text):
        out.append(_math_chunk(text[last:match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(_math_chunk(text[last:]))
    return "".join(out)


def _math_chunk(chunk: str) -> str:
    """Шматок між словами: якщо в ньому є математика — у «$…$» (пробіли по краях зберігаємо)."""
    core = chunk.strip(" .,:")
    if not core or not re.search(r"[\dx∞∅√=<>≤≥(\[{]", core):
        return chunk
    start = chunk.index(core)
    return f"{chunk[:start]}${from_text(core)}${chunk[start + len(core):]}"
