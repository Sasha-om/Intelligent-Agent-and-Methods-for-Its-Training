"""Параметричні шаблони завдань: для кожної з 9 підтем — генерація чисел, умова, розв'язок і неправильні варіанти.

Як це працює (однаково для всіх підтем):
    make_<підтема>(level, rng)  → params   — випадкові, але «гарні» числа для рівня складності level (−2…2);
    text_<підтема>(params)      → умова    — текст завдання з цих чисел;
    solve_<підтема>(params)     → розв'язок — відповідь, кроки, типова помилка (рахується з тих самих params!);
    wrong_<підтема>(params)     → список типових неправильних відповідей (для варіантів).

Відповідь у solve — це «об'єкт відповіді»: множина проміжків (список), число (Fraction) або готовий текст (str).
Перетворює його на текст core/items.py → answer_text().
Тест tests/test_items.py перевіряє: розв'язок, обчислений з params, збігається з правильним варіантом.
"""
import random
from fractions import Fraction

from core.math_utils import (FLIP_OP, INF, MINUS, OP_TEXT, complement, count_integers, fmt_factor,
                             fmt_intervals, fmt_num, fmt_point, fmt_poly, interval_signs, intersect,
                             mirror, ray, sign_method, solve_linear, toggle_closed, union)

F = Fraction


# ============================================================================= спільні помічники
def _nz(rng: random.Random, lo: int, hi: int) -> int:
    """Випадкове ціле з [lo; hi], але не нуль (щоб не ділити на нуль і не мати «порожніх» доданків)."""
    value = 0
    while value == 0:
        value = rng.randint(lo, hi)
    return value


def _roots(rng: random.Random, count: int, lo: int = -6, hi: int = 6) -> list:
    """count різних цілих коренів, відсортованих за зростанням."""
    return sorted(rng.sample(range(lo, hi + 1), count))


def _answer_line(answer: object) -> str:
    """Останній крок розв'язку: «Відповідь: …»."""
    text = fmt_intervals(answer) if isinstance(answer, list) else (
        answer if isinstance(answer, str) else fmt_num(answer))
    return f"Відповідь: {text}."


def _vertex_form(h: int, k: int, sign: int = 1) -> str:
    """Парабола у вигляді ±(x − h)² + k, наприклад «−(x + 2)² − 3»."""
    text = ("" if sign > 0 else MINUS) + f"{fmt_factor(h)}²"
    if k:
        text += f" {'+' if k > 0 else MINUS} {abs(k)}"
    return text


def _quad_coefs(p: dict) -> tuple:
    """Коефіцієнти a, b, c параболи з params (у вершинній або звичайній формі)."""
    if "h" in p:
        a = p.get("a", 1)
        return a, -2 * a * p["h"], a * p["h"] ** 2 + p["k"]
    return p["a"], p["b"], p["c"]


def _vertex(a: int, b: int, c: int) -> tuple:
    """Вершина параболи: x₀ = −b/(2a), y₀ = y(x₀) — точно, у дробах."""
    x0 = F(-b, 2 * a)
    return x0, a * x0 * x0 + b * x0 + c


def _quad_expr(p: dict) -> str:
    """Текст параболи так, як він стоїть в умові."""
    if "h" in p:
        return _vertex_form(p["h"], p["k"], p.get("a", 1))
    return fmt_poly([p["a"], p["b"], p["c"]])


def _point_set(x: Fraction) -> list:
    """Множина з однієї точки {x} — щоб записати «x ≠ x₀» як доповнення."""
    return [(F(x), F(x), True, True)]


# ============================================================================= 1. область визначення
def make_domain(level: int, rng: random.Random) -> dict:
    """Числа для завдання на область визначення."""
    if level == -2 or level == -1:
        return {"level": level, "a": _nz(rng, -6, 6)}
    if level == 0:
        return {"level": level, "k": rng.randint(2, 5), "m": _nz(rng, -9, 9)}
    if level == 1:
        a = rng.randint(-5, 3)
        return {"level": level, "a": a, "c": a + rng.randint(1, 5)}
    r1, r2 = _roots(rng, 2)
    return {"level": level, "r1": r1, "r2": r2}


def text_domain(p: dict) -> str:
    """Умова: «Знайдіть область визначення функції y = …»."""
    lv = p["level"]
    if lv == -2:
        expr = f"1/{fmt_factor(p['a'])}"
    elif lv == -1:
        expr = f"√{fmt_factor(p['a'])}"
    elif lv == 0:
        expr = f"√({fmt_num(p['m'])} {MINUS} {p['k']}x)"
    elif lv == 1:
        expr = f"√{fmt_factor(p['a'])} + 1/{fmt_factor(p['c'])}"
    else:
        expr = f"√({fmt_poly([1, -(p['r1'] + p['r2']), p['r1'] * p['r2']])})"
    return f"Знайдіть область визначення функції y = {expr}."


def solve_domain(p: dict) -> dict:
    """Область визначення: знаменник ≠ 0, під коренем ≥ 0."""
    lv = p["level"]
    if lv == -2:
        ans = complement(_point_set(p["a"]))
        steps = [f"На нуль ділити не можна: {fmt_factor(p['a'])} ≠ 0, тобто x ≠ {fmt_num(p['a'])}."]
        typical = "Записати x = a замість x ≠ a (сплутати «можна» і «не можна»)."
    elif lv == -1:
        ans = ray(">=", p["a"])
        steps = [f"Під коренем має бути невід'ємне число: {fmt_factor(p['a'])} ≥ 0, отже x ≥ {fmt_num(p['a'])}."]
        typical = "Поставити строгу нерівність x > a, хоча √0 = 0 існує."
    elif lv == 0:
        op, bound = solve_linear(-p["k"], p["m"], 0, ">=")
        ans = ray(op, bound)
        steps = [f"Під коренем: {fmt_num(p['m'])} {MINUS} {p['k']}x ≥ 0, тобто {MINUS}{p['k']}x ≥ {fmt_num(-p['m'])}.",
                 f"Ділимо на від'ємне число {MINUS}{p['k']} і змінюємо знак: x ≤ {fmt_num(bound)}."]
        typical = "Не змінити знак нерівності при діленні на від'ємне число."
    elif lv == 1:
        ans = intersect(ray(">=", p["a"]), complement(_point_set(p["c"])))
        steps = [f"Корінь: x ≥ {fmt_num(p['a'])}. Дріб: x ≠ {fmt_num(p['c'])}.",
                 "Обидві умови мають виконуватися разом, тому виключаємо точку з променя."]
        typical = "Забути виключити точку, де знаменник дорівнює нулю."
    else:
        ans = sign_method([p["r1"], p["r2"]], [], ">=")
        steps = [f"Під коренем ≥ 0. Корені квадратного тричлена: {p['r1']} і {p['r2']}.",
                 "Гілки параболи вгору, тому тричлен ≥ 0 зліва від меншого й справа від більшого кореня."]
        typical = "Взяти проміжок між коренями замість двох крайніх."
    return {"answer": ans, "steps": steps + [_answer_line(ans)], "typical_error": typical}


def wrong_domain(p: dict) -> list:
    """Типові неправильні відповіді для області визначення."""
    lv = p["level"]
    if lv == -2:
        return [ray(">", p["a"]), complement(_point_set(-p["a"])), _point_set(p["a"])]
    if lv == -1:
        return [ray(">", p["a"]), ray(">=", -p["a"]), ray("<=", p["a"])]
    if lv == 0:
        bound = F(p["m"], p["k"])
        return [ray(">=", bound), ray("<=", -bound), ray("<", bound)]
    if lv == 1:
        return [ray(">=", p["a"]), [(F(p["a"]), F(p["c"]), True, False)],
                intersect(ray(">", p["a"]), complement(_point_set(p["c"])))]
    return [[(F(p["r1"]), F(p["r2"]), True, True)], sign_method([p["r1"], p["r2"]], [], ">")]


# ============================================================================= 2. область значень
def make_range(level: int, rng: random.Random) -> dict:
    """Числа для завдання на область значень квадратичної функції."""
    if level == -2:
        return {"level": level, "a": 1, "b": 0, "c": _nz(rng, -7, 7)}
    if level == -1:
        return {"level": level, "a": -1, "b": 0, "c": _nz(rng, -7, 7)}
    if level == 0:
        return {"level": level, "h": _nz(rng, -5, 5), "k": rng.randint(-6, 6)}
    if level == 1:
        return {"level": level, "a": 1, "b": 2 * _nz(rng, -4, 4), "c": rng.randint(-6, 8)}
    return {"level": level, "a": rng.choice([-3, -2, 2, 3]), "b": _nz(rng, -8, 8), "c": rng.randint(-5, 5)}


def text_range(p: dict) -> str:
    """Умова: «Знайдіть область значень функції y = …»."""
    return f"Знайдіть область значень функції y = {_quad_expr(p)}."


def _range_of(a: int, y0: Fraction) -> list:
    """Область значень параболи з вершиною на висоті y0."""
    return [(y0, INF, True, False)] if a > 0 else [(-INF, y0, False, True)]


def solve_range(p: dict) -> dict:
    """Область значень параболи — від вершини вгору (a > 0) або вниз (a < 0)."""
    a, b, c = _quad_coefs(p)
    x0, y0 = _vertex(a, b, c)
    where = "вгору" if a > 0 else "вниз"
    steps = [f"Графік — парабола, a = {fmt_num(a)}, тому гілки напрямлені {where}."]
    if "h" in p:
        steps.append(f"Вершина параболи — точка {fmt_point(p['h'], p['k'])}.")
    else:
        steps.append(f"Абсциса вершини: x₀ = −b/(2a) = {fmt_num(x0)}; ордината: y₀ = y(x₀) = {fmt_num(y0)}.")
    ans = _range_of(a, y0)
    best = "найменше" if a > 0 else "найбільше"
    steps += [f"y₀ = {fmt_num(y0)} — {best} значення функції.", _answer_line(ans)]
    return {"answer": ans, "steps": steps,
            "typical_error": "Взяти абсцису вершини x₀ замість ординати y₀ або переплутати напрям гілок."}


def wrong_range(p: dict) -> list:
    """Типові помилки: x₀ замість y₀, інший напрям гілок, c замість y₀."""
    a, b, c = _quad_coefs(p)
    x0, y0 = _vertex(a, b, c)
    return [_range_of(a, x0), _range_of(-a, y0), _range_of(a, F(c)), _range_of(a, -y0)]


# ============================================================================= 3. зростання і спадання
LINEAR_BEHAVIOUR = ["зростає на всій числовій прямій", "спадає на всій числовій прямій",
                    "стала на всій числовій прямій", "зростає при x > 0 і спадає при x < 0"]


def make_monotonic(level: int, rng: random.Random) -> dict:
    """Числа для завдання на проміжки зростання/спадання."""
    if level == -2:
        return {"level": level, "k": _nz(rng, -5, 5), "b": rng.randint(-6, 6)}
    if level == -1:
        return {"level": level, "h": _nz(rng, -5, 5), "k": 0, "ask": "up"}
    if level == 0:
        return {"level": level, "a": -1, "h": _nz(rng, -5, 5), "k": rng.randint(-5, 5), "ask": "up"}
    if level == 1:
        return {"level": level, "a": 1, "b": 2 * _nz(rng, -4, 4), "c": rng.randint(-5, 5), "ask": "down"}
    return {"level": level, "a": rng.choice([-2, -1, 2, 3]), "b": _nz(rng, -7, 7), "c": rng.randint(-5, 5),
            "ask": rng.choice(["up", "down"])}


def text_monotonic(p: dict) -> str:
    """Умова: поведінка лінійної функції або проміжок зростання/спадання параболи."""
    if p["level"] == -2:
        return f"Як змінюється функція y = {fmt_poly([p['k'], p['b']])}?"
    what = "зростання" if p["ask"] == "up" else "спадання"
    return f"Укажіть проміжок {what} функції y = {_quad_expr(p)}."


def _monotone_ray(a: int, x0: Fraction, rising: bool) -> list:
    """Проміжок зростання (rising=True) або спадання параболи з вершиною в x0."""
    right_side = (a > 0) == rising     # при a > 0 парабола зростає праворуч від вершини
    return [(x0, INF, True, False)] if right_side else [(-INF, x0, False, True)]


def solve_monotonic(p: dict) -> dict:
    """Лінійна: знак k. Парабола: вершина ділить пряму на «спуск» і «підйом»."""
    if p["level"] == -2:
        ans = LINEAR_BEHAVIOUR[0] if p["k"] > 0 else LINEAR_BEHAVIOUR[1]
        steps = [f"Це лінійна функція з кутовим коефіцієнтом k = {fmt_num(p['k'])}.",
                 f"k {'> 0' if p['k'] > 0 else '< 0'}, тому функція {ans.split()[0]} скрізь.", _answer_line(ans)]
        return {"answer": ans, "steps": steps, "typical_error": "Дивитися на вільний член b замість коефіцієнта k."}
    a, b, c = _quad_coefs(p)
    x0, _y0 = _vertex(a, b, c)
    ans = _monotone_ray(a, x0, p["ask"] == "up")
    side = "праворуч" if ans[0][1] == INF else "ліворуч"
    steps = [f"Парабола, a = {fmt_num(a)}: гілки {'вгору' if a > 0 else 'вниз'}; абсциса вершини x₀ = {fmt_num(x0)}.",
             f"Функція {'зростає' if p['ask'] == 'up' else 'спадає'} {side} від вершини.", _answer_line(ans)]
    return {"answer": ans, "steps": steps, "brackets_matter": False,
            "typical_error": "Переплутати проміжки зростання і спадання або знак абсциси вершини."}


def wrong_monotonic(p: dict) -> list:
    """Протилежний проміжок і помилка знака вершини."""
    if p["level"] == -2:
        return [t for t in LINEAR_BEHAVIOUR if t != solve_monotonic(p)["answer"]]
    a, b, c = _quad_coefs(p)
    x0, _y0 = _vertex(a, b, c)
    up = p["ask"] == "up"
    return [_monotone_ray(a, x0, not up), _monotone_ray(a, -x0, up), _monotone_ray(a, -x0, not up)]


# ============================================================================= 4. лінійна функція
def make_linear_func(level: int, rng: random.Random) -> dict:
    """Числа для завдань на лінійну функцію y = kx + b."""
    if level == -2:
        return {"level": level, "k": _nz(rng, -5, 5), "b": rng.randint(-9, 9), "x0": _nz(rng, -4, 4)}
    if level == -1:
        return {"level": level, "k": _nz(rng, -6, 6), "b": _nz(rng, -9, 9)}
    if level == 0:
        x1, x2 = rng.sample(range(-4, 5), 2)
        return {"level": level, "k": _nz(rng, -4, 4), "b": rng.randint(-5, 5), "x1": x1, "x2": x2}
    k1, k2 = rng.sample([-3, -2, -1, 1, 2, 3, 4], 2)
    x0, y0 = rng.randint(-4, 4), rng.randint(-5, 5)
    if level == 1:
        return {"level": level, "k1": k1, "k2": k2, "x0": x0, "y0": y0}
    c = y0 - k1 * x0 + _nz(rng, -4, 4)     # c ≠ b: дана пряма НЕ проходить через M
    return {"level": level, "k": k1, "c": c, "x0": x0, "y0": y0}


def text_linear_func(p: dict) -> str:
    """Умова для однієї з п'яти задач про лінійну функцію."""
    lv = p["level"]
    if lv == -2:
        return f"Обчисліть значення функції y = {fmt_poly([p['k'], p['b']])} при x = {fmt_num(p['x0'])}."
    if lv == -1:
        return f"Знайдіть нуль функції y = {fmt_poly([p['k'], p['b']])}."
    if lv == 0:
        a, b = _line_point(p, p["x1"]), _line_point(p, p["x2"])
        return f"Графік функції y = kx + b проходить через точки A{a} і B{b}. Знайдіть k."
    if lv == 1:
        f1, f2 = (fmt_poly([k, p["y0"] - k * p["x0"]]) for k in (p["k1"], p["k2"]))
        return f"Знайдіть координати точки перетину графіків функцій y = {f1} і y = {f2}."
    return (f"Задайте формулою лінійну функцію, графік якої паралельний прямій y = {fmt_poly([p['k'], p['c']])} "
            f"і проходить через точку M{fmt_point(p['x0'], p['y0'])}.")


def _line_point(p: dict, x: int) -> str:
    """Точка (x; kx + b) на прямій з params рівня 0."""
    return fmt_point(x, p["k"] * x + p["b"])


def solve_linear_func(p: dict) -> dict:
    """Розв'язок задач про y = kx + b (значення, нуль, k за двома точками, перетин, паралельна пряма)."""
    lv = p["level"]
    if lv == -2:
        ans = F(p["k"] * p["x0"] + p["b"])
        steps = [f"Підставляємо x = {fmt_num(p['x0'])}: y = {fmt_num(p['k'])}·({fmt_num(p['x0'])}) + ({fmt_num(p['b'])})."]
        typical = "Помилитися зі знаком при множенні на від'ємне число."
    elif lv == -1:
        ans = F(-p["b"], p["k"])
        steps = [f"Нуль функції — це x, при якому y = 0: {fmt_poly([p['k'], p['b']])} = 0.",
                 f"x = {fmt_num(-p['b'])} : {fmt_num(p['k'])}."]
        typical = "Перенести b у праву частину без зміни знака."
    elif lv == 0:
        y1, y2 = (p["k"] * x + p["b"] for x in (p["x1"], p["x2"]))
        ans = F(y2 - y1, p["x2"] - p["x1"])
        steps = [f"k = (y₂ − y₁)/(x₂ − x₁) = ({fmt_num(y2)} − ({fmt_num(y1)}))/({fmt_num(p['x2'])} − ({fmt_num(p['x1'])}))."]
        typical = "Поділити різницю x на різницю y (перевернути дріб)."
    elif lv == 1:
        return _solve_intersection(p)
    else:
        return _solve_parallel(p)
    return {"answer": ans, "steps": steps + [_answer_line(ans)], "typical_error": typical}


def _solve_intersection(p: dict) -> dict:
    """Перетин двох прямих: прирівнюємо праві частини."""
    k1, k2, x0, y0 = p["k1"], p["k2"], p["x0"], p["y0"]
    b1, b2 = y0 - k1 * x0, y0 - k2 * x0
    x = F(b2 - b1, k1 - k2)
    ans = fmt_point(x, k1 * x + b1)
    steps = [f"Прирівнюємо: {fmt_poly([k1, b1])} = {fmt_poly([k2, b2])}, звідси x = {fmt_num(x)}.",
             f"Підставляємо x в будь-яку формулу: y = {fmt_num(k1 * x + b1)}.", _answer_line(ans)]
    return {"answer": ans, "steps": steps, "typical_error": "Знайти лише x або переставити координати місцями."}


def _solve_parallel(p: dict) -> dict:
    """Паралельна пряма: той самий k, а b знаходимо з точки M."""
    k, x0, y0 = p["k"], p["x0"], p["y0"]
    b = y0 - k * x0
    ans = f"y = {fmt_poly([k, b])}"
    steps = [f"Паралельні прямі мають однаковий кутовий коефіцієнт: k = {fmt_num(k)}.",
             f"Точка M на прямій: {fmt_num(y0)} = {fmt_num(k)}·({fmt_num(x0)}) + b, тому b = {fmt_num(b)}.",
             _answer_line(ans)]
    return {"answer": ans, "steps": steps, "typical_error": "Змінити k на протилежний або помилитися зі знаком при знаходженні b."}


def wrong_linear_func(p: dict) -> list:
    """Типові помилки для задач про лінійну функцію."""
    lv = p["level"]
    if lv == -2:
        return [F(p["k"] * p["x0"] - p["b"]), F(-p["k"] * p["x0"] + p["b"]), F(p["k"] * p["x0"])]
    if lv == -1:
        return [F(p["b"], p["k"]), F(-p["k"], p["b"]), F(-p["b"])]
    if lv == 0:
        return [F(1, p["k"]), F(-p["k"]), F(p["b"])]
    if lv == 1:
        x0, y0 = p["x0"], p["y0"]
        return [fmt_point(y0, x0), fmt_point(-x0, y0), fmt_point(x0, -y0), fmt_point(x0 + 1, y0), fmt_point(x0, y0 + 2),
                fmt_point(x0 - 1, y0), fmt_point(x0, y0 - 2)]
    k, x0, y0 = p["k"], p["x0"], p["y0"]
    b = y0 - k * x0
    return [f"y = {fmt_poly([k, y0 + k * x0])}", f"y = {fmt_poly([-k, b])}", f"y = {fmt_poly([k, p['c']])}",
            f"y = {fmt_poly([k, b + 1])}", f"y = {fmt_poly([k, b - 1])}", f"y = {fmt_poly([-k, y0 + k * x0])}"]


# ============================================================================= 5. квадратична функція і графік
def make_quadratic_func(level: int, rng: random.Random) -> dict:
    """Числа для завдань на квадратичну функцію."""
    if level == -2:
        return {"level": level, "h": _nz(rng, -5, 5), "k": _nz(rng, -6, 6)}
    if level == -1:
        return {"level": level, "a": 1, "b": _nz(rng, -6, 6), "c": rng.randint(-8, 8), "x0": _nz(rng, -3, 3)}
    if level == 0:
        return {"level": level, "a": 1, "b": 2 * _nz(rng, -4, 4), "c": rng.randint(-6, 6)}
    if level == 1:
        r1, r2 = _roots(rng, 2)
        while r1 + r2 == 0:                     # інакше «дзеркальна» відповідь збіглася б з правильною
            r1, r2 = _roots(rng, 2)
        return {"level": level, "r1": r1, "r2": r2}
    return {"level": level, "p": 2 * _nz(rng, -5, 5)}


def text_quadratic_func(p: dict) -> str:
    """Умова для однієї з п'яти задач про параболу."""
    lv = p["level"]
    if lv == -2:
        return f"Знайдіть координати вершини параболи y = {_vertex_form(p['h'], p['k'])}."
    if lv == -1:
        return f"Обчисліть значення функції y = {fmt_poly([1, p['b'], p['c']])} при x = {fmt_num(p['x0'])}."
    if lv == 0:
        return f"Знайдіть координати вершини параболи y = {fmt_poly([1, p['b'], p['c']])}."
    if lv == 1:
        return f"Знайдіть нулі функції y = {fmt_poly([1, -(p['r1'] + p['r2']), p['r1'] * p['r2']])}."
    return (f"При якому значенні c парабола y = {fmt_poly([1, p['p'], 0])} + c має з віссю абсцис "
            f"рівно одну спільну точку?")


def _pair(a: object, b: object) -> str:
    """Два корені у порядку зростання: «−2 і 3»."""
    lo, hi = sorted((F(a), F(b)))
    return f"{fmt_num(lo)} і {fmt_num(hi)}"


def solve_quadratic_func(p: dict) -> dict:
    """Розв'язки задач про параболу (вершина, значення, нулі, дотик до осі Ox)."""
    lv = p["level"]
    if lv == -2:
        ans = fmt_point(p["h"], p["k"])
        steps = ["Для y = (x − h)² + k вершина — точка (h; k). Увага: у дужці стоїть «x мінус h»."]
        typical = "Взяти h з протилежним знаком."
    elif lv in (-1, 0):
        return _solve_quad_value_or_vertex(p)
    elif lv == 1:
        ans = _pair(p["r1"], p["r2"])
        steps = [f"Розв'язуємо x² + px + q = 0. За теоремою Вієта: x₁ + x₂ = {fmt_num(p['r1'] + p['r2'])}, "
                 f"x₁·x₂ = {fmt_num(p['r1'] * p['r2'])}."]
        typical = "Взяти корені з протилежними знаками (сплутати знаки у теоремі Вієта)."
    else:
        ans = F(p["p"] ** 2, 4)
        steps = ["Одна спільна точка з віссю Ox — це дискримінант D = 0.",
                 f"D = {fmt_num(p['p'])}² − 4c = 0, тому c = {fmt_num(p['p'] ** 2)}/4."]
        typical = "Помилитися в формулі дискримінанта (наприклад, D = b² + 4ac)."
    return {"answer": ans, "steps": steps + [_answer_line(ans)], "typical_error": typical}


def _solve_quad_value_or_vertex(p: dict) -> dict:
    """Рівень −1: значення параболи в точці; рівень 0: вершина параболи у звичайному вигляді."""
    a, b, c = _quad_coefs(p)
    if p["level"] == -1:
        x = p["x0"]
        ans = F(x * x + b * x + c)
        steps = [f"Підставляємо x = {fmt_num(x)}: y = ({fmt_num(x)})² + {fmt_num(b)}·({fmt_num(x)}) + ({fmt_num(c)})."]
        return {"answer": ans, "steps": steps + [_answer_line(ans)],
                "typical_error": "Вважати, що (−x)² від'ємне."}
    x0, y0 = _vertex(a, b, c)
    ans = fmt_point(x0, y0)
    steps = [f"x₀ = −b/(2a) = {fmt_num(x0)}.", f"y₀ = y(x₀) = {fmt_num(y0)}.", _answer_line(ans)]
    return {"answer": ans, "steps": steps, "typical_error": "Забути мінус у формулі x₀ = −b/(2a)."}


def wrong_quadratic_func(p: dict) -> list:
    """Типові помилки для задач про параболу."""
    lv = p["level"]
    if lv == -2:
        h, k = p["h"], p["k"]
        return [fmt_point(-h, k), fmt_point(h, -k), fmt_point(-h, -k), fmt_point(k, h)]
    a, b, c = _quad_coefs(p) if lv in (-1, 0) else (1, 0, 0)
    if lv == -1:
        x = p["x0"]
        return [F(-x * x + b * x + c), F(x * x - b * x + c), F(2 * x + b * x + c)]
    if lv == 0:
        x0, y0 = _vertex(a, b, c)
        return [fmt_point(-x0, y0), fmt_point(x0, c), fmt_point(y0, x0), fmt_point(-x0, c)]
    if lv == 1:
        r1, r2 = p["r1"], p["r2"]
        return [_pair(-r1, -r2), _pair(r1, -r2), _pair(-r1, r2), "нулів немає", _pair(r1 + 1, r2 + 1),
                _pair(r1 - 1, r2 - 1)]
    return [F(-p["p"] ** 2, 4), F(p["p"] ** 2, 2), F(p["p"], 2)]


# ============================================================================= 6. лінійні нерівності
def make_linear_ineq(level: int, rng: random.Random) -> dict:
    """Числа для лінійної нерівності; знак нерівності теж випадковий."""
    op = rng.choice([">", ">=", "<", "<="])
    if level == -2:
        return {"level": level, "m": _nz(rng, -9, 9), "n": rng.randint(-9, 9), "op": op}
    if level == -1:
        return {"level": level, "k": rng.randint(2, 6), "n": _nz(rng, -12, 12), "op": op}
    if level == 0:
        return {"level": level, "m": _nz(rng, -9, 9), "k": rng.randint(2, 5), "n": rng.randint(-9, 9), "op": op}
    if level == 1:
        a = rng.randint(2, 5)
        b = rng.choice([v for v in range(-3, 7) if v != a])
        return {"level": level, "a": a, "b": b, "p": _nz(rng, -4, 4), "c": rng.randint(-8, 8), "op": op}
    m, n = rng.sample([2, 3, 4, 5], 2)
    return {"level": level, "m": m, "n": n, "p": rng.randint(-5, 5), "q": rng.randint(-5, 5),
            "r": rng.randint(-2, 2), "op": op}


def text_linear_ineq(p: dict) -> str:
    """Умова: «Розв'яжіть нерівність …»."""
    lv, sign = p["level"], OP_TEXT[p["op"]]
    if lv == -2:
        body = f"{fmt_poly([1, p['m']])} {sign} {fmt_num(p['n'])}"
    elif lv == -1:
        body = f"{p['k']}x {sign} {fmt_num(p['n'])}"
    elif lv == 0:
        body = f"{fmt_num(p['m'])} {MINUS} {p['k']}x {sign} {fmt_num(p['n'])}"
    elif lv == 1:
        body = f"{p['a']}{fmt_factor(p['p'])} {sign} {fmt_poly([p['b'], p['c']])}"
    else:
        body = (f"{fmt_factor(p['p'])}/{p['m']} {MINUS} {fmt_factor(-p['q'])}/{p['n']} {sign} {fmt_num(p['r'])}")
    return f"Розв'яжіть нерівність {body}."


def _linear_abc(p: dict) -> tuple:
    """Зводимо нерівність до вигляду A·x + B op C (кожен рівень — своїми перетвореннями)."""
    lv = p["level"]
    if lv == -2:
        return 1, p["m"], p["n"]
    if lv == -1:
        return p["k"], 0, p["n"]
    if lv == 0:
        return -p["k"], p["m"], p["n"]
    if lv == 1:
        return p["a"] - p["b"], 0, p["c"] + p["a"] * p["p"]
    m, n = p["m"], p["n"]   # множимо обидві частини на m·n > 0
    return n - m, 0, p["r"] * m * n + n * p["p"] + m * p["q"]


def solve_linear_ineq(p: dict) -> dict:
    """Лінійна нерівність: звести до A·x op C і поділити на A (якщо A < 0 — змінити знак)."""
    a, b, c = _linear_abc(p)
    op, bound = solve_linear(a, b, c, p["op"])
    ans = ray(op, bound)
    steps = []
    if p["level"] == 2:
        steps.append(f"Множимо обидві частини на {p['m'] * p['n']} (це додатне число, знак не змінюється).")
    steps.append(f"Переносимо доданки з x ліворуч, числа праворуч: {fmt_poly([a, 0])} {OP_TEXT[p['op']]} {fmt_num(c - b)}.")
    if a < 0:
        steps.append(f"Ділимо на від'ємне число {fmt_num(a)} і змінюємо знак: x {OP_TEXT[op]} {fmt_num(bound)}.")
    elif a != 1:
        steps.append(f"Ділимо на {fmt_num(a)}: x {OP_TEXT[op]} {fmt_num(bound)}.")
    steps.append(_answer_line(ans))
    typical = ("Перенести число в іншу частину без зміни знака." if p["level"] == -2 else
               "Не змінити знак нерівності при діленні на від'ємне число.")
    return {"answer": ans, "steps": steps, "typical_error": typical}


def wrong_linear_ineq(p: dict) -> list:
    """Типові помилки: без зміни знака, зі зміною знака межі, з іншим знаком переносу."""
    a, b, c = _linear_abc(p)
    op, bound = solve_linear(a, b, c, p["op"])
    wrong = [ray(FLIP_OP[op], bound), ray(op, -bound)]
    if p["level"] == -2:
        wrong.insert(0, ray(op, F(p["n"] + p["m"])))
    return wrong


# ============================================================================= 7. квадратні нерівності
def make_quadratic_ineq(level: int, rng: random.Random) -> dict:
    """Числа для квадратної нерівності з цілими коренями r1 < r2."""
    if level in (-2, -1):
        n = rng.randint(1, 7)
        return {"level": level, "lead": 1, "r1": -n, "r2": n, "op": "<=" if level == -2 else ">"}
    r1, r2 = _roots(rng, 2)
    lead = -rng.randint(1, 3) if level == 2 else 1
    op = rng.choice([">", ">=", "<", "<="])
    return {"level": level, "lead": lead, "r1": r1, "r2": r2, "op": op}


def text_quadratic_ineq(p: dict) -> str:
    """Умова: x² ≤ n², добуток (x − r1)(x − r2) або розгорнутий тричлен."""
    lv, sign, r1, r2 = p["level"], OP_TEXT[p["op"]], p["r1"], p["r2"]
    if lv in (-2, -1):
        return f"Розв'яжіть нерівність x² {sign} {r2 * r2}."
    if lv == 0:
        return f"Розв'яжіть нерівність {fmt_factor(r1)}{fmt_factor(r2)} {sign} 0."
    lead = p["lead"]
    poly = fmt_poly([lead, -lead * (r1 + r2), lead * r1 * r2])
    return f"Розв'яжіть нерівність {poly} {sign} 0."


def solve_quadratic_ineq(p: dict) -> dict:
    """Квадратна нерівність: корені, напрям гілок, вибір проміжків."""
    r1, r2, lead = p["r1"], p["r2"], p["lead"]
    op = p["op"] if lead > 0 else FLIP_OP[p["op"]]
    ans = sign_method([r1, r2], [], op)
    steps = [f"Корені тричлена: x₁ = {fmt_num(r1)}, x₂ = {fmt_num(r2)}."]
    if lead < 0:
        steps.append(f"Ділимо на {fmt_num(lead)} < 0 і змінюємо знак: (x − x₁)(x − x₂) {OP_TEXT[op]} 0.")
    inside = op in ("<", "<=")
    steps.append("Парабола з гілками вгору від'ємна між коренями і додатна зовні — "
                 f"беремо {'проміжок між коренями' if inside else 'два крайні проміжки'}.")
    steps.append(_answer_line(ans))
    return {"answer": ans, "steps": steps,
            "typical_error": "Взяти «зовнішні» проміжки замість «внутрішнього» (або навпаки)."}


def wrong_quadratic_ineq(p: dict) -> list:
    """Типові помилки: протилежні проміжки, інші дужки, забута зміна знака при a < 0."""
    ans = solve_quadratic_ineq(p)["answer"]
    wrong = [complement(ans), toggle_closed(ans)]
    if p["lead"] < 0:
        wrong.insert(0, sign_method([p["r1"], p["r2"]], [], p["op"]))
    return wrong


# ============================================================================= 8. метод інтервалів
INTERVAL_SHAPES = {-2: (2, 0, None), -1: (1, 1, "strict"), 0: (1, 1, "loose"), 1: (3, 0, None), 2: (2, 1, "loose")}


def make_interval_method(level: int, rng: random.Random) -> dict:
    """Корені чисельника й знаменника (усі різні) і знак нерівності."""
    n_num, n_den, kind = INTERVAL_SHAPES[level]
    roots = rng.sample(range(-6, 7), n_num + n_den)
    ops = {"strict": [">", "<"], "loose": [">=", "<="]}.get(kind, [">", ">=", "<", "<="])
    return {"level": level, "num": sorted(roots[:n_num]), "den": sorted(roots[n_num:]), "op": rng.choice(ops)}


def text_interval_method(p: dict) -> str:
    """Умова: «Розв'яжіть нерівність (x − 1)(x + 2)/(x − 4) ≥ 0»."""
    # множник «x» (корінь 0) пишемо першим: «x(x − 2)», а не «(x − 2)x»
    num = "".join(fmt_factor(r) for r in sorted(p["num"], key=lambda r: r != 0))
    expr = num if not p["den"] else f"{num}/{''.join(fmt_factor(r) for r in p['den'])}"
    return f"Розв'яжіть методом інтервалів нерівність {expr} {OP_TEXT[p['op']]} 0."


def solve_interval_method(p: dict) -> dict:
    """Метод інтервалів: точки на прямій, знаки на проміжках, вибір потрібних."""
    ans = sign_method(p["num"], p["den"], p["op"])
    signs = ", ".join(interval_signs(p["num"] + p["den"]))
    steps = [f"Нулі чисельника: {', '.join(fmt_num(r) for r in p['num'])}."]
    if p["den"]:
        steps.append(f"Нуль знаменника {fmt_num(p['den'][0])} виключаємо завжди: на нуль ділити не можна.")
    steps.append(f"Знаки на проміжках зліва направо: {signs}.")
    want = "+" if p["op"] in (">", ">=") else MINUS
    steps += [f"Беремо проміжки зі знаком «{want}».", _answer_line(ans)]
    return {"answer": ans, "steps": steps,
            "typical_error": "Включити у відповідь нуль знаменника або переплутати знаки на проміжках."}


def wrong_interval_method(p: dict) -> list:
    """Типові помилки: нуль знаменника включено, протилежні знаки, інші дужки."""
    ans = sign_method(p["num"], p["den"], p["op"])
    wrong = [complement(ans), toggle_closed(ans), mirror(ans)]
    if p["den"]:
        wrong.insert(0, sign_method(p["num"] + p["den"], [], p["op"]))
    return wrong


# ============================================================================= 9. системи нерівностей
def _lin(a: int, b: int, op: str, bound: int) -> dict:
    """Лінійна нерівність a·x + b op c, у якої після розв'язання x (op або протилежний) bound."""
    return {"kind": "lin", "a": a, "b": b, "c": a * bound + b, "op": op if a > 0 else FLIP_OP[op]}


def make_ineq_system(level: int, rng: random.Random) -> dict:
    """Будуємо систему «від відповіді»: спершу межі L < R, потім нерівності з такими розв'язками."""
    left, right = rng.choice([">", ">="]), rng.choice(["<", "<="])
    if level == 2:
        n = rng.randint(2, 6)
        low = rng.randint(-n, n - 1)
        return {"level": level, "ask": "set", "ineqs": [{"kind": "sq", "n": n, "op": "<="}, _lin(1, 0, left, low)]}
    low = rng.randint(-6, 3)
    high = low + rng.randint(2, 6)
    if level == -2:
        ineqs = [_lin(1, 0, left, low), _lin(1, 0, right, high)]
    elif level == -1:
        ineqs = [_lin(rng.randint(2, 5), 0, left, low), _lin(1, 0, right, high)]
    elif level == 0:
        ineqs = [_lin(rng.randint(2, 5), _nz(rng, -6, 6), left, low), _lin(rng.randint(2, 4), _nz(rng, -6, 6), right, high)]
    else:
        ineqs = [_lin(-rng.randint(2, 5), _nz(rng, -6, 6), left, low), _lin(rng.randint(2, 4), _nz(rng, -6, 6), right, high)]
    return {"level": level, "ask": "count" if level == 1 else "set", "ineqs": ineqs}


def _ineq_text(q: dict) -> str:
    """Одна нерівність системи як текст."""
    if q["kind"] == "sq":
        return f"x² {OP_TEXT[q['op']]} {q['n'] ** 2}"
    left = fmt_poly([q["a"], q["b"]])
    return f"{left} {OP_TEXT[q['op']]} {fmt_num(q['c'])}"


def _ineq_set(q: dict) -> list:
    """Розв'язок однієї нерівності системи."""
    if q["kind"] == "sq":
        return sign_method([-q["n"], q["n"]], [], q["op"])
    return ray(*solve_linear(q["a"], q["b"], q["c"], q["op"]))


def text_ineq_system(p: dict) -> str:
    """Умова: система з двох нерівностей; на рівні 1 питаємо кількість цілих розв'язків."""
    body = "{ " + ";  ".join(_ineq_text(q) for q in p["ineqs"]) + " }"
    if p["ask"] == "count":
        return f"Скільки цілих розв'язків має система нерівностей {body}?"
    return f"Розв'яжіть систему нерівностей {body}."


def solve_ineq_system(p: dict) -> dict:
    """Система: розв'язуємо кожну нерівність окремо і беремо ПЕРЕТИН."""
    parts = [_ineq_set(q) for q in p["ineqs"]]
    common = intersect(parts[0], parts[1])
    steps = [f"Перша нерівність: {fmt_intervals(parts[0])}.", f"Друга нерівність: {fmt_intervals(parts[1])}.",
             f"Система — це «і», тому беремо спільну частину: {fmt_intervals(common)}."]
    ans = F(count_integers(common)) if p["ask"] == "count" else common
    if p["ask"] == "count":
        steps.append("Рахуємо цілі числа в цьому проміжку, уважно дивлячись на дужки на кінцях.")
    steps.append(_answer_line(ans))
    return {"answer": ans, "steps": steps,
            "typical_error": "Об'єднати розв'язки замість перетину або неправильно врахувати кінці проміжку."}


def wrong_ineq_system(p: dict) -> list:
    """Типові помилки: об'єднання замість перетину, інші дужки, неточний підрахунок."""
    parts = [_ineq_set(q) for q in p["ineqs"]]
    common = intersect(parts[0], parts[1])
    if p["ask"] == "count":
        n = count_integers(common)
        closed = count_integers([(a, b, True, True) for a, b, _ac, _bc in common])
        return [F(v) for v in (n + 1, n - 1, closed, n + 2, n + 3) if v >= 0]
    return [union(parts[0], parts[1]), toggle_closed(common), complement(common)]


# ============================================================================= реєстр
TEMPLATES = {
    "domain": (make_domain, text_domain, solve_domain, wrong_domain),
    "range": (make_range, text_range, solve_range, wrong_range),
    "monotonic": (make_monotonic, text_monotonic, solve_monotonic, wrong_monotonic),
    "linear_func": (make_linear_func, text_linear_func, solve_linear_func, wrong_linear_func),
    "quadratic_func": (make_quadratic_func, text_quadratic_func, solve_quadratic_func, wrong_quadratic_func),
    "linear_ineq": (make_linear_ineq, text_linear_ineq, solve_linear_ineq, wrong_linear_ineq),
    "quadratic_ineq": (make_quadratic_ineq, text_quadratic_ineq, solve_quadratic_ineq, wrong_quadratic_ineq),
    "interval_method": (make_interval_method, text_interval_method, solve_interval_method, wrong_interval_method),
    "ineq_system": (make_ineq_system, text_ineq_system, solve_ineq_system, wrong_ineq_system),
}
