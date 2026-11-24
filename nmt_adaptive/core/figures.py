"""Малюнки для розбору (SVG-текст, без streamlit): числова пряма з відповіддю і знаками, ескіз параболи.

У застосунку: st.image(svg). Підписи — звичайним текстом із типографським мінусом.
"""
from fractions import Fraction

from core.math_utils import INF, fmt_num

BLUE, GREY, RED = "#1f6fb4", "#444444", "#c0392b"
WIDTH, HEIGHT, MARGIN = 520, 86, 40
FONT = 'font-family="sans-serif" font-size="14"'


def _span(values: list) -> tuple:
    """Межі видимої частини прямої: усі точки + запас по краях."""
    finite = [float(v) for v in values if abs(v) != INF] or [0.0]
    lo, hi = min(finite), max(finite)
    pad = max(1.0, (hi - lo) * 0.25)
    return lo - pad, hi + pad


def _scale(lo: float, hi: float) -> callable:
    """Функція «число → пікселі» для прямої."""
    return lambda x: MARGIN + (float(x) - lo) / (hi - lo) * (WIDTH - 2 * MARGIN)


def _axis(y: int) -> list:
    """Вісь зі стрілкою і підписом x."""
    end = WIDTH - 12
    return [f'<line x1="8" y1="{y}" x2="{end}" y2="{y}" stroke="{GREY}" stroke-width="1.5"/>',
            f'<polygon points="{end},{y - 5} {end + 10},{y} {end},{y + 5}" fill="{GREY}"/>',
            f'<text x="{end - 4}" y="{y + 22}" {FONT} fill="{GREY}">x</text>']


def _band(iv: tuple, px: callable, y: int) -> str:
    """Зафарбований проміжок (товста синя смуга; до краю, якщо нескінченний)."""
    lo, hi = iv[0], iv[1]
    x1 = 8 if lo == -INF else px(lo)
    x2 = WIDTH - 14 if hi == INF else px(hi)
    return f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="{BLUE}" stroke-width="7" opacity="0.55"/>'


def _dot(x: float, y: int, closed: bool, label: str) -> list:
    """Точка: зафарбована (входить) або порожня (не входить) + підпис під нею."""
    fill = BLUE if closed else "white"
    return [f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="{fill}" stroke="{BLUE}" stroke-width="2.2"/>',
            f'<text x="{x:.1f}" y="{y + 26}" {FONT} text-anchor="middle" fill="{GREY}">{label}</text>']


def number_line(answer: list, points: list | None = None, signs: list | None = None) -> str:
    """Числова пряма: answer — зафарбовані проміжки; points — [(x, входить?)]; signs — «+»/«−» між точками.

    Якщо points не задано, точки — кінці проміжків відповіді.
    """
    if points is None:
        points = sorted({(iv[k], iv[2 + k]) for iv in answer for k in (0, 1) if abs(iv[k]) != INF})
    xs = sorted({Fraction(p[0]) for p in points})
    lo, hi = _span(xs)
    px, y = _scale(lo, hi), 38 if signs else 34
    parts = _axis(y) + [_band(iv, px, y) for iv in answer if iv[0] != iv[1]]
    for x, closed in points:
        parts += _dot(px(x), y, closed, fmt_num(x))
    if signs:
        edges = [lo] + [float(x) for x in xs] + [hi]
        for k, sign in enumerate(signs):
            mid = (px(edges[k]) + px(edges[k + 1])) / 2
            color = BLUE if sign == "+" else RED
            parts.append(f'<text x="{mid:.1f}" y="{y - 12}" font-family="sans-serif" font-size="20" '
                         f'text-anchor="middle" fill="{color}">{sign}</text>')
    return _svg(WIDTH, HEIGHT if signs else HEIGHT - 8, parts)


def parabola(a: object, b: object, c: object, marks: list) -> str:
    """Ескіз параболи y = ax² + bx + c; marks — [(x, y, підпис)] (корені, вершина)."""
    a, b, c = (float(v) for v in (a, b, c))
    x0 = -b / (2 * a)
    xs_marked = [float(m[0]) for m in marks] + [x0]
    half = max(3.0, (max(xs_marked) - min(xs_marked)) / 2 + 2)
    xl, xr = x0 - half, x0 + half
    grid = [xl + (xr - xl) * i / 80 for i in range(81)]
    ys = [a * x * x + b * x + c for x in grid]
    yl, yr = min(ys + [0.0]), max(ys + [0.0])
    w, h, m = 320, 220, 24
    px = lambda x: m + (x - xl) / (xr - xl) * (w - 2 * m)                      # noqa: E731
    py = lambda yy: h - m - (yy - yl) / ((yr - yl) or 1) * (h - 2 * m)          # noqa: E731
    parts = [f'<line x1="{m - 10}" y1="{py(0):.1f}" x2="{w - 6}" y2="{py(0):.1f}" stroke="{GREY}"/>',
             f'<text x="{w - 14}" y="{py(0) + 16:.1f}" {FONT} fill="{GREY}">x</text>']
    if xl < 0 < xr:
        parts.append(f'<line x1="{px(0):.1f}" y1="6" x2="{px(0):.1f}" y2="{h - 6}" stroke="{GREY}"/>')
        parts.append(f'<text x="{px(0) + 6:.1f}" y="16" {FONT} fill="{GREY}">y</text>')
    path = " ".join(f"{px(x):.1f},{py(yy):.1f}" for x, yy in zip(grid, ys))
    parts.append(f'<polyline points="{path}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    for x, yy, label in marks:
        cx, cy = px(float(x)), py(float(yy))
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{RED}"/>')
        tx, ty, anchor = (cx, cy + 20, "middle") if yy == 0 else (cx + 8, cy + 18, "start")   # корінь — під віссю
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" {FONT} text-anchor="{anchor}" fill="{RED}">{label}</text>')
    return _svg(w, h, parts)


def _svg(width: int, height: int, parts: list) -> str:
    """Збирає SVG-документ."""
    body = "\n".join(parts)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n{body}\n</svg>')
