"""Покроковий розбір задач про функції: область визначення, область значень, монотонність, лінійна функція,
квадратична функція. Кожна функція walk_<підтема>(p) повертає список кроків (core/walk_common.step).

Числа — з тих самих params, що й умова; кінцеву відповідь бере core/walkthrough.py з solve(), тож вона завжди
збігається з правильним варіантом.
"""
from fractions import Fraction

from core import figures
from core import latex as L
from core.item_latex import domain_expr, lin, quad_expr
from core.item_templates import _quad_coefs, _vertex
from core.math_utils import INF, fmt_num
from core.walk_common import step

F = Fraction


def _sqrt_ok(value: Fraction) -> tuple:
    """Для перевірки «під коренем ≥ 0»: (значення, 0)."""
    return value, F(0)


# ============================================================================= 1. область визначення
def domain_conditions(p: dict) -> list:
    """Умови існування виразу: (ліва, op, права, f(x)) — для кроків і для перевірки на прикладі."""
    lv = p["level"]
    if lv == -2:
        return [(lin(p["a"]), "!=", "0", lambda x: (x - p["a"], F(0)))]
    if lv == -1:
        return [(lin(p["a"]), ">=", "0", lambda x: _sqrt_ok(x - p["a"]))]
    if lv == 0:
        return [(f"{L.num(p['m'])} - {p['k']}x", ">=", "0", lambda x: _sqrt_ok(p["m"] - p["k"] * x))]
    if lv == 1:
        return [(lin(p["a"]), ">=", "0", lambda x: _sqrt_ok(x - p["a"])),
                (lin(p["c"]), "!=", "0", lambda x: (x - p["c"], F(0)))]
    r1, r2 = p["r1"], p["r2"]
    return [(L.poly([1, -(r1 + r2), r1 * r2]), ">=", "0", lambda x: _sqrt_ok((x - r1) * (x - r2)))]


def walk_domain(p: dict) -> list:
    """Область визначення: шукаємо, де вираз «ламається» (ділення на 0, корінь з від'ємного)."""
    intro = step("Що шукаємо", "Область визначення — це всі значення $x$, при яких вираз має зміст. "
                 "Ламають вираз дві речі: **ділення на нуль** і **корінь з від'ємного числа**.",
                 [f"y = {domain_expr(p)}"])
    lv = p["level"]
    if lv == -2:
        body = [step("Записуємо умову", "Є дріб — знаменник не може дорівнювати нулю:", [rf"{lin(p['a'])} \ne 0"]),
                step("Розв'язуємо", "Переносимо число в праву частину зі зміною знака:", [rf"x \ne {L.num(p['a'])}"]),
                step("Записуємо проміжками", f"Підходять усі числа, крім ${L.num(p['a'])}$. Точку «виколюємо» "
                     "(на прямій — порожній кружечок), тому дужки круглі.")]
    elif lv == -1:
        body = [step("Записуємо умову", "Є квадратний корінь — під ним має бути **невід'ємне** число (≥ 0):",
                     [rf"{lin(p['a'])} \ge 0"]),
                step("Розв'язуємо", "Переносимо число в праву частину зі зміною знака:", [rf"x \ge {L.num(p['a'])}"]),
                step("Записуємо проміжком", f"Число ${L.num(p['a'])}$ підходить (бо $\\sqrt{{0}} = 0$), тому дужка "
                     "квадратна; праворуч — до нескінченності, там дужка завжди кругла.")]
    elif lv == 0:
        body = _domain_level0(p)
    elif lv == 1:
        body = _domain_level1(p)
    else:
        body = _domain_level2(p)
    return [intro] + body


def _domain_level0(p: dict) -> list:
    """√(m − kx): переносимо m, ділимо на від'ємне −k зі зміною знака."""
    m, k = p["m"], p["k"]
    bound = F(m, k)
    return [step("Записуємо умову", "Під коренем має бути невід'ємне число:", [rf"{L.num(m)} - {k}x \ge 0"]),
            step("Переносимо число", f"Переносимо ${L.num(m)}$ у праву частину, змінюючи знак:",
                 [rf"-{k}x \ge {L.num(-m)}"]),
            step("Ділимо на від'ємне число", f"Ділимо обидві частини на $-{k}$. **Ділимо на від'ємне — знак "
                 "нерівності змінюється на протилежний** ($\\ge$ стає $\\le$):",
                 [rf"x \le \frac{{{L.num(-m)}}}{{-{k}}} = {L.num(bound)}"]),
            step("Записуємо проміжком", "Межа входить (≥ 0 допускає нуль), тому дужка біля неї квадратна.")]


def _domain_level1(p: dict) -> list:
    """√(x − a) + 1/(x − c): дві умови одночасно — система."""
    a, c = p["a"], p["c"]
    return [step("Записуємо умови", "Тут і корінь, і дріб — обидві умови мають виконуватися **одночасно** "
                 "(система):", [rf"\begin{{cases}} {lin(a)} \ge 0 \\ {lin(c)} \ne 0 \end{{cases}}"]),
            step("Розв'язуємо кожну", "", [rf"\begin{{cases}} x \ge {L.num(a)} \\ x \ne {L.num(c)} \end{{cases}}"]),
            step("Поєднуємо", f"Беремо промінь від ${L.num(a)}$ праворуч і «виколюємо» з нього точку ${L.num(c)}$ "
                 "— промінь розпадається на два проміжки.",
                 figure=figures.number_line([(F(a), F(c), True, False), (F(c), INF, False, False)],
                                            points=[(F(a), True), (F(c), False)]))]


def _domain_level2(p: dict) -> list:
    """√(квадратного тричлена): корені, парабола гілками вгору, тричлен ≥ 0 зовні коренів."""
    r1, r2 = p["r1"], p["r2"]
    b, c = -(r1 + r2), r1 * r2
    expr = L.poly([1, b, c])
    d = b * b - 4 * c
    return [step("Записуємо умову", "Під коренем має бути невід'ємне число:", [rf"{expr} \ge 0"]),
            step("Знаходимо корені тричлена", "Розв'язуємо рівняння через дискримінант:",
                 [rf"D = b^{{2}} - 4ac = {L.paren(b)}^{{2}} - 4\cdot 1\cdot {L.paren(c)} = {d}",
                  rf"x_{{1,2}} = \frac{{-b \pm \sqrt{{D}}}}{{2a}} = \frac{{{L.num(-b)} \pm {int(d ** 0.5)}}}{{2}}"
                  rf"\quad\Rightarrow\quad x_1 = {r1},\ \ x_2 = {r2}"]),
            step("Дивимося на параболу", "$a = 1 > 0$ — гілки вгору. Тричлен **додатний зліва і справа від коренів** "
                 "і від'ємний між ними. Нам потрібно ≥ 0 — беремо два крайні проміжки разом з коренями.",
                 figure=figures.parabola(1, b, c, [(r1, 0, fmt_num(r1)), (r2, 0, fmt_num(r2))]))]


# ============================================================================= 2. область значень
def walk_range(p: dict) -> list:
    """Область значень параболи: вершина — найнижча (a > 0) або найвища (a < 0) точка."""
    a, b, c = _quad_coefs(p)
    x0, y0 = _vertex(a, b, c)
    up = a > 0
    branches = ("> 0$, тому гілки напрямлені **вгору**: найменше значення — у вершині." if up else
                "< 0$, тому гілки напрямлені **вниз**: найбільше значення — у вершині.")
    steps = [step("Що шукаємо", "Область значень — усі значення, які може набувати $y$. Графік — **парабола**, "
                  f"$a = {L.num(a)} {branches}", [f"y = {quad_expr(p)}"])]
    steps += _vertex_steps(p, a, b, c, x0, y0)
    sign = r"\ge" if up else r"\le"
    steps.append(step("Робимо висновок", f"Від вершини гілки йдуть {'вгору' if up else 'вниз'} без кінця, тому "
                      f"$y$ набуває всіх значень {'від' if up else 'до'} ${L.num(y0)}$ (включно).",
                      [rf"y {sign} {L.num(y0)}"],
                      figures.parabola(a, b, c, [(x0, y0, "вершина")])))
    return steps


def _vertex_steps(p: dict, a: int, b: int, c: int, x0: Fraction, y0: Fraction) -> list:
    """Знаходження вершини: з вершинного вигляду або за формулою x₀ = −b/(2a)."""
    if "h" in p:
        return [step("Знаходимо вершину", "Функція записана у вигляді $a(x - h)^{2} + k$ — вершина в точці $(h;\\,k)$. "
                     "Увага: в дужках «$x$ **мінус** $h$», тому знак числа в дужках протилежний до $h$.",
                     [rf"h = {L.num(p['h'])},\quad k = {L.num(p['k'])}\quad\Rightarrow\quad "
                      rf"\text{{вершина}}\ {L.point(x0, y0)}"])]
    return [step("Знаходимо абсцису вершини", "За формулою:",
                 [rf"x_0 = -\frac{{b}}{{2a}} = -\frac{{{L.num(b)}}}{{2\cdot {L.paren(a)}}} = {L.num(x0)}"]),
            step("Знаходимо ординату вершини", "Підставляємо $x_0$ у формулу функції:",
                 [rf"y_0 = {_poly_at(a, b, c, x0)} = {L.num(y0)}"])]


def _poly_at(a: int, b: int, c: int, x: Fraction) -> str:
    """Підстановка в ax² + bx + c: «2·(−1)² − 3·(−1) + 4» (нульові доданки пропускаємо)."""
    v = L.paren(x) if F(x).denominator == 1 else rf"\left({L.num(x)}\right)"
    lead = {1: "", -1: "-"}.get(a, rf"{L.num(a)}\cdot ")
    text = f"{lead}{v}^{{2}}"
    if b:
        text += rf" {'+' if b > 0 else '-'} {abs(b)}\cdot {v}"
    if c:
        text += f" {'+' if c > 0 else '-'} {abs(c)}"
    return text


# ============================================================================= 3. зростання і спадання
def walk_monotonic(p: dict) -> list:
    """Лінійна: знак k. Парабола: вершина ділить пряму на «спуск» і «підйом»."""
    if p["level"] == -2:
        return _monotonic_linear(p)
    a, b, c = _quad_coefs(p)
    x0, y0 = _vertex(a, b, c)
    up = a > 0
    what = "зростає" if p["ask"] == "up" else "спадає"
    return [step("Що шукаємо", f"Шукаємо, де функція {what}. Графік — парабола; $a = {L.num(a)}$, тому гілки "
                 f"**{'вгору' if up else 'вниз'}**.", [f"y = {quad_expr(p)}"])] + \
        _vertex_steps(p, a, b, c, x0, y0)[:1] + [
        step("Дивимося на графік", "Рухаємось зліва направо. "
             + ("Гілки вгору: до вершини графік **спускається** (спадає), після — **піднімається** (зростає)."
                if up else "Гілки вниз: до вершини графік **піднімається** (зростає), після — **спускається** (спадає).")
             + f" Межа — абсциса вершини $x_0 = {L.num(x0)}$.",
             figure=figures.parabola(a, b, c, [(x0, y0, "вершина")])),
        step("Перевірка на прикладі", "Візьмемо два числа з відповіді — більше $x$ дає "
             + ("більше $y$." if p["ask"] == "up" else "менше $y$."), _monotonic_example(a, b, c, x0, p["ask"], up))]


def _monotonic_example(a: int, b: int, c: int, x0: Fraction, ask: str, up: bool) -> list:
    """Два числа з відповіді і значення функції в них."""
    right = (ask == "up") == up
    x1, x2 = (x0 + 1, x0 + 2) if right else (x0 - 2, x0 - 1)
    y = lambda x: a * x * x + b * x + c      # noqa: E731
    sign = "<" if ask == "up" else ">"
    return [rf"x = {L.num(x1)}:\ y = {L.num(y(x1))};\qquad x = {L.num(x2)}:\ y = {L.num(y(x2))};\qquad "
            rf"{L.num(y(x1))} {sign} {L.num(y(x2))}\ \checkmark"]


def _monotonic_linear(p: dict) -> list:
    """y = kx + b: зростає при k > 0, спадає при k < 0."""
    k, b = p["k"], p["b"]
    y = lambda x: k * x + b      # noqa: E731
    return [step("Що за функція", "Це **лінійна** функція $y = kx + b$; її графік — пряма.",
                 [rf"y = {L.poly([k, b])},\quad k = {L.num(k)},\ \ b = {L.num(b)}"]),
            step("Дивимося на k", "Як поводиться пряма, вирішує **кутовий коефіцієнт** $k$ (число біля $x$), "
                 "а не $b$: $k > 0$ — пряма йде вгору (зростає), $k < 0$ — вниз (спадає).",
                 [rf"k = {L.num(k)} {'> 0' if k > 0 else '< 0'}"]),
            step("Перевірка на прикладі", "Візьмемо $x = 0$ і $x = 1$:",
                 [rf"y(0) = {L.num(y(0))},\quad y(1) = {L.num(y(1))}\quad\Rightarrow\quad "
                  rf"\text{{{'збільшилось' if k > 0 else 'зменшилось'}}}\ \checkmark"])]


# ============================================================================= 4. лінійна функція
def walk_linear_func(p: dict) -> list:
    """Лінійна функція: значення, нуль, k за двома точками, перетин прямих, паралельна пряма."""
    return {-2: _lin_value, -1: _lin_zero, 0: _lin_slope, 1: _lin_cross, 2: _lin_parallel}[p["level"]](p)


def _lin_value(p: dict) -> list:
    """y = kx + b при x = x₀: підставляємо і рахуємо по діях."""
    k, b, x0 = p["k"], p["b"], p["x0"]
    return [step("Що треба зробити", "Обчислити значення функції — це **підставити** число замість $x$.",
                 [f"y = {L.poly([k, b])},\\quad x = {L.num(x0)}"]),
            step("Підставляємо", "Від'ємне число беремо в дужки, щоб не загубити мінус:",
                 [f"y = {L.num(k)}\\cdot {L.paren(x0)} {'+' if b >= 0 else '-'} {abs(b)}"]),
            step("Рахуємо по діях", "Спершу множення, потім додавання:",
                 [f"{L.num(k)}\\cdot {L.paren(x0)} = {L.num(k * x0)}",
                  f"y = {L.num(k * x0)} {'+' if b >= 0 else '-'} {abs(b)} = {L.num(k * x0 + b)}"])]


def _lin_zero(p: dict) -> list:
    """Нуль функції: kx + b = 0."""
    k, b = p["k"], p["b"]
    x = F(-b, k)
    return [step("Що таке нуль функції", "Нуль функції — це таке $x$, при якому $y = 0$. Прирівнюємо до нуля:",
                 [f"{L.poly([k, b])} = 0"]),
            step("Переносимо число", f"Переносимо ${L.num(b)}$ у праву частину зі зміною знака:",
                 [f"{L.poly([k, 0])} = {L.num(-b)}"]),
            step("Ділимо на коефіцієнт біля x", "", [rf"x = \frac{{{L.num(-b)}}}{{{L.num(k)}}} = {L.num(x)}"]),
            step("Перевірка на прикладі", "Підставляємо знайдене $x$ — має вийти нуль:",
                 [rf"{L.num(k)}\cdot {L.paren(x)} {'+' if b >= 0 else '-'} {abs(b)} = {L.num(k * x)} "
                  rf"{'+' if b >= 0 else '-'} {abs(b)} = 0\ \checkmark"])]


def _lin_slope(p: dict) -> list:
    """k за двома точками: k = (y₂ − y₁)/(x₂ − x₁)."""
    k, b, x1, x2 = p["k"], p["b"], p["x1"], p["x2"]
    y1, y2 = k * x1 + b, k * x2 + b
    return [step("Згадуємо формулу", "Кутовий коефіцієнт — це «на скільки змінюється $y$, коли $x$ змінюється на 1»:",
                 [r"k = \frac{y_2 - y_1}{x_2 - x_1}"]),
            step("Виписуємо координати", "", [rf"A{L.point(x1, y1)}:\ x_1 = {x1},\ y_1 = {y1};\qquad "
                                              rf"B{L.point(x2, y2)}:\ x_2 = {x2},\ y_2 = {y2}"]),
            step("Підставляємо", "Вгорі — різниця **y**, внизу — різниця **x** (у тому самому порядку):",
                 [rf"k = \frac{{{y2} - {L.paren(y1)}}}{{{x2} - {L.paren(x1)}}} = \frac{{{y2 - y1}}}{{{x2 - x1}}} "
                  rf"= {L.num(F(y2 - y1, x2 - x1))}"]),
            step("Перевірка на прикладі", f"Знайдемо $b$ з точки A і перевіримо точку B:",
                 [rf"b = y_1 - k x_1 = {y1} - {L.paren(k)}\cdot {L.paren(x1)} = {b}",
                  rf"y(x_2) = {L.num(k)}\cdot {L.paren(x2)} {'+' if b >= 0 else '-'} {abs(b)} = {y2}\ \checkmark"])]


def _lin_cross(p: dict) -> list:
    """Перетин двох прямих: прирівнюємо праві частини."""
    k1, k2, x0, y0 = p["k1"], p["k2"], p["x0"], p["y0"]
    b1, b2 = y0 - k1 * x0, y0 - k2 * x0
    x = F(b2 - b1, k1 - k2)
    y = k1 * x + b1
    return [step("Ідея", "У точці перетину обидва графіки мають **однакові** $x$ і $y$. Тому прирівнюємо праві частини:",
                 [f"{L.poly([k1, b1])} = {L.poly([k2, b2])}"]),
            step("Збираємо x ліворуч, числа праворуч", "Переносимо доданки через знак «=», змінюючи їхні знаки:",
                 [f"{L.poly([k1, 0])} {'-' if k2 > 0 else '+'} {L.poly([abs(k2), 0])} = {L.num(b2)} "
                  f"{'-' if b1 > 0 else '+'} {abs(b1)}", f"{L.poly([k1 - k2, 0])} = {L.num(b2 - b1)}",
                  rf"x = \frac{{{L.num(b2 - b1)}}}{{{L.num(k1 - k2)}}} = {L.num(x)}"]),
            step("Знаходимо y", "Підставляємо $x$ у **будь-яку** з двох формул:",
                 [rf"y = {L.num(k1)}\cdot {L.paren(x)} {'+' if b1 >= 0 else '-'} {abs(b1)} = {L.num(y)}"]),
            step("Перевірка на прикладі", "Підставляємо в другу формулу — має вийти те саме $y$:",
                 [rf"{L.num(k2)}\cdot {L.paren(x)} {'+' if b2 >= 0 else '-'} {abs(b2)} = {L.num(k2 * x + b2)}\ \checkmark"]),
            step("Записуємо точку", "Спочатку $x$, потім $y$:", [rf"{L.point(x, y)}"])]


def _lin_parallel(p: dict) -> list:
    """Паралельна пряма через точку M: той самий k, b — з координат M."""
    k, c, x0, y0 = p["k"], p["c"], p["x0"], p["y0"]
    b = y0 - k * x0
    return [step("Згадуємо правило", "Паралельні прямі мають **однаковий кутовий коефіцієнт** $k$ (різні лише $b$).",
                 [rf"y = {L.poly([k, c])}\quad\Rightarrow\quad k = {L.num(k)}", rf"\text{{шукаємо }} y = {L.poly([k, 0])} + b"]),
            step("Знаходимо b з точки M", f"Точка $M{L.point(x0, y0)}$ лежить на прямій — підставляємо її координати:",
                 [rf"{y0} = {L.num(k)}\cdot {L.paren(x0)} + b", rf"{y0} = {L.num(k * x0)} + b",
                  rf"b = {y0} - {L.paren(k * x0)} = {L.num(b)}"]),
            step("Перевірка на прикладі", "Підставляємо $x = $ абсцису M — має вийти ордината M:",
                 [rf"{L.num(k)}\cdot {L.paren(x0)} {'+' if b >= 0 else '-'} {abs(b)} = {y0}\ \checkmark"])]


# ============================================================================= 5. квадратична функція
def walk_quadratic_func(p: dict) -> list:
    """Парабола: вершина з вершинного вигляду, значення, вершина за формулою, нулі, дотик до осі."""
    return {-2: _quad_vertex_form, -1: _quad_value, 0: _quad_vertex, 1: _quad_zeros, 2: _quad_touch}[p["level"]](p)


def _quad_vertex_form(p: dict) -> list:
    """y = (x − h)² + k: вершина (h; k)."""
    h, k = p["h"], p["k"]
    return [step("Згадуємо правило", "Для параболи $y = (x - h)^{2} + k$ вершина — точка $(h;\\,k)$.",
                 [f"y = {quad_expr({'h': h, 'k': k})}"]),
            step("Читаємо h і k", "Обережно зі знаком: в дужках стоїть «$x$ **мінус** $h$». "
                 + (f"$(x - {h})$ означає $h = {h}$." if h > 0 else f"$(x + {-h})$ — це $(x - ({h}))$, тобто $h = {h}$."),
                 [rf"h = {h},\quad k = {k}"]),
            step("Перевірка на прикладі", "У вершині дужка дорівнює нулю, тому $y = k$:",
                 [rf"y({h}) = ({L.num(h)} - {L.paren(h)})^{{2}} {'+' if k > 0 else '-'} {abs(k)} = {k}\ \checkmark"],
                 figures.parabola(1, -2 * h, h * h + k, [(h, k, "вершина")]))]


def _quad_value(p: dict) -> list:
    """y = x² + bx + c при x = x₀: підставляємо і рахуємо по діях."""
    b, c, x0 = p["b"], p["c"], p["x0"]
    return [step("Що треба зробити", "Підставити число замість $x$ (від'ємне — у дужках!).",
                 [rf"y = {L.poly([1, b, c])},\quad x = {x0}", rf"y = {_poly_at(1, b, c, F(x0))}"]),
            step("Рахуємо по діях", "Квадрат від'ємного числа — **додатний**:",
                 [rf"{L.paren(x0)}^{{2}} = {x0 * x0},\qquad {b}\cdot {L.paren(x0)} = {b * x0}",
                  rf"y = {x0 * x0} {'+' if b * x0 >= 0 else '-'} {abs(b * x0)} {'+' if c >= 0 else '-'} {abs(c)} "
                  rf"= {x0 * x0 + b * x0 + c}"])]


def _quad_vertex(p: dict) -> list:
    """Вершина параболи y = x² + bx + c за формулою."""
    a, b, c = _quad_coefs(p)
    x0, y0 = _vertex(a, b, c)
    return [step("Згадуємо формулу", "Абсциса вершини параболи $y = ax^{2} + bx + c$:", [r"x_0 = -\frac{b}{2a}"]),
            *_vertex_steps(p, a, b, c, x0, y0),
            step("Записуємо точку", "", [L.point(x0, y0)], figures.parabola(a, b, c, [(x0, y0, "вершина")]))]


def _quad_zeros(p: dict) -> list:
    """Нулі x² + bx + c: дискримінант і корені."""
    r1, r2 = p["r1"], p["r2"]
    b, c = -(r1 + r2), r1 * r2
    d = b * b - 4 * c
    return [step("Що таке нулі функції", "Нулі — це $x$, при яких $y = 0$. Розв'язуємо квадратне рівняння:",
                 [f"{L.poly([1, b, c])} = 0"]),
            step("Дискримінант", "", [rf"D = b^{{2}} - 4ac = {L.paren(b)}^{{2}} - 4\cdot 1\cdot {L.paren(c)} = {d}"]),
            step("Корені", "$D > 0$ — два корені:",
                 [rf"x_{{1,2}} = \frac{{-b \pm \sqrt{{D}}}}{{2a}} = \frac{{{L.num(-b)} \pm {int(d ** 0.5)}}}{{2}}",
                  rf"x_1 = {r1},\qquad x_2 = {r2}"]),
            step("Перевірка на прикладі", "За теоремою Вієта: сума коренів $= -b$, добуток $= c$:",
                 [rf"{r1} + {L.paren(r2)} = {r1 + r2} = -b\ \checkmark,\qquad {r1}\cdot {L.paren(r2)} = {c} = c\ \checkmark"],
                 figures.parabola(1, b, c, [(r1, 0, fmt_num(r1)), (r2, 0, fmt_num(r2))]))]


def _quad_touch(p: dict) -> list:
    """Одна спільна точка з віссю Ox ⇔ D = 0."""
    q = p["p"]
    c = F(q * q, 4)
    return [step("Ідея", "Спільні точки з віссю абсцис — це нулі функції. **Рівно одна** точка буває, коли "
                 "дискримінант дорівнює нулю (парабола торкається осі вершиною).", [rf"x^{{2}} {'+' if q > 0 else '-'} "
                                                                                 rf"{abs(q)}x + c = 0"]),
            step("Записуємо D = 0", "", [rf"D = {L.paren(q)}^{{2}} - 4\cdot 1\cdot c = {q * q} - 4c = 0"]),
            step("Знаходимо c", "", [rf"4c = {q * q}\quad\Rightarrow\quad c = \frac{{{q * q}}}{{4}} = {L.num(c)}"]),
            step("Перевірка на прикладі", "Тоді вираз — повний квадрат, і нуль лише один:",
                 [rf"x^{{2}} {'+' if q > 0 else '-'} {abs(q)}x + {L.num(c)} = {L.factor(-q // 2)}^{{2}} = 0"
                  rf"\ \Rightarrow\ x = {-q // 2}\ \checkmark"],
                 figures.parabola(1, q, c, [(F(-q, 2), 0, fmt_num(F(-q, 2)))]))]
