"""Тести банку завдань.

Головна перевірка: правильна відповідь, яку рахує програма з params, справді розв'язує умову.
Для цього кожну умову перевіряємо «в лоб» — підставляємо сотні чисел x у вихідну нерівність чи функцію,
незалежно від коду розв'язувача (методу інтервалів, формули вершини тощо).
"""
import random
from fractions import Fraction as F

import pytest

import config
from core import items
from core.item_templates import TEMPLATES
from core.math_utils import INF, contains, fmt_num, fmt_point, fmt_poly

GRID = [F(i, 12) for i in range(-12 * 12, 12 * 12 + 1)]   # x від −12 до 12 з кроком 1/12
OPS = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}
LEVELS = [int(b) for b in config.DIFFICULTY_LEVELS]


# ----------------------------------------------------------------------------- допоміжне
def _points(answer: list) -> list:
    """Сітка + усі скінченні межі відповіді та точки поруч із ними (там найчастіше помилки)."""
    extra = []
    for a, b, _ac, _bc in answer:
        for v in (a, b):
            if v not in (INF, -INF):
                extra += [F(v), F(v) - F(1, 1000), F(v) + F(1, 1000)]
    return GRID + extra


def _same_set(answer: list, condition) -> bool:
    """Відповідь (множина) збігається з умовою в кожній перевіреній точці."""
    return all(contains(answer, x) == condition(x) for x in _points(answer))


def _poly(coefs: list, x: F) -> F:
    """Значення многочлена в точці x (коефіцієнти від старшого)."""
    value = F(0)
    for c in coefs:
        value = value * x + c
    return value


def _quad(p: dict) -> list:
    """Коефіцієнти параболи з params так, як вона записана в умові."""
    if "h" in p:
        a = p.get("a", 1)
        return [a, -2 * a * p["h"], a * p["h"] ** 2 + p["k"]]
    return [p["a"], p["b"], p["c"]]


# ----------------------------------------------------------------------------- незалежні перевірки умов
def _domain_condition(p: dict):
    """Де функція з умови визначена (перевіряємо прямо за формулою)."""
    lv = p["level"]
    if lv == -2:
        return lambda x: x != p["a"]
    if lv == -1:
        return lambda x: x - p["a"] >= 0
    if lv == 0:
        return lambda x: p["m"] - p["k"] * x >= 0
    if lv == 1:
        return lambda x: x - p["a"] >= 0 and x != p["c"]
    return lambda x: _poly([1, -(p["r1"] + p["r2"]), p["r1"] * p["r2"]], x) >= 0


def _linear_ineq_condition(p: dict):
    """Вихідна лінійна нерівність, як вона записана в умові."""
    lv, cmp = p["level"], OPS[p["op"]]
    if lv == -2:
        return lambda x: cmp(x + p["m"], p["n"])
    if lv == -1:
        return lambda x: cmp(p["k"] * x, p["n"])
    if lv == 0:
        return lambda x: cmp(p["m"] - p["k"] * x, p["n"])
    if lv == 1:
        return lambda x: cmp(p["a"] * (x - p["p"]), p["b"] * x + p["c"])
    return lambda x: cmp((x - p["p"]) / p["m"] - (x + p["q"]) / p["n"], p["r"])


def _interval_condition(p: dict):
    """Добуток/частка множників з умови; у нулях знаменника вираз не визначений → не входить."""
    def holds(x: F) -> bool:
        den = F(1)
        for r in p["den"]:
            den *= x - r
        if den == 0:
            return False
        num = F(1)
        for r in p["num"]:
            num *= x - r
        return OPS[p["op"]](num / den, 0)
    return holds


def _system_condition(p: dict):
    """Обидві нерівності системи виконуються одночасно."""
    def one(q: dict, x: F) -> bool:
        if q["kind"] == "sq":
            return OPS[q["op"]](x * x, q["n"] ** 2)
        return OPS[q["op"]](q["a"] * x + q["b"], q["c"])
    return lambda x: all(one(q, x) for q in p["ineqs"])


def _check_set_answer(subtopic: str, p: dict, answer: list) -> bool:
    """Перевірка відповідей-множин для нерівностей і області визначення."""
    if subtopic == "domain":
        return _same_set(answer, _domain_condition(p))
    if subtopic == "linear_ineq":
        return _same_set(answer, _linear_ineq_condition(p))
    if subtopic == "quadratic_ineq":
        lead, r1, r2 = p["lead"], p["r1"], p["r2"]
        return _same_set(answer, lambda x: OPS[p["op"]](_poly([lead, -lead * (r1 + r2), lead * r1 * r2], x), 0))
    if subtopic == "interval_method":
        return _same_set(answer, _interval_condition(p))
    return _same_set(answer, _system_condition(p))


def _check_function_answer(subtopic: str, p: dict, answer: object) -> bool:
    """Перевірка відповідей про функції: перебором по сітці, а не формулою з розв'язувача."""
    if subtopic == "range":
        values = [_poly(_quad(p), x) for x in GRID]
        if _quad(p)[0] > 0:
            return answer == [(min(values), INF, True, False)]
        return answer == [(-INF, max(values), False, True)]
    if subtopic == "monotonic":
        return _check_monotonic(p, answer)
    if subtopic == "linear_func":
        return _check_linear_func(p, answer)
    return _check_quadratic_func(p, answer)


def _check_monotonic(p: dict, answer: object) -> bool:
    """На знайденому промені функція справді зростає/спадає, а за вершиною — навпаки."""
    if p["level"] == -2:
        return answer.startswith("зростає") == (p["k"] > 0)
    coefs = _quad(p)
    inside = [x for x in GRID if contains(answer, x)]
    outside = [x for x in GRID if not contains(answer, x)]
    trend = 1 if p["ask"] == "up" else -1
    ok_inside = all(trend * (_poly(coefs, b) - _poly(coefs, a)) > 0 for a, b in zip(inside, inside[1:]))
    ok_outside = all(trend * (_poly(coefs, b) - _poly(coefs, a)) < 0 for a, b in zip(outside, outside[1:]))
    return ok_inside and ok_outside


def _check_linear_func(p: dict, answer: object) -> bool:
    """Значення, нуль, кутовий коефіцієнт, точка перетину, паралельна пряма — підстановкою."""
    lv = p["level"]
    if lv == -2:
        return answer == p["k"] * p["x0"] + p["b"]
    if lv == -1:
        return p["k"] * answer + p["b"] == 0
    if lv == 0:
        y1, y2 = (p["k"] * x + p["b"] for x in (p["x1"], p["x2"]))
        return y2 - y1 == answer * (p["x2"] - p["x1"])
    if lv == 1:
        x, y = p["x0"], p["y0"]     # обидві прямі проходять через (x0; y0) і не паралельні → це перетин
        return p["k1"] != p["k2"] and answer == fmt_point(x, y)
    return answer == f"y = {fmt_poly([p['k'], p['y0'] - p['k'] * p['x0']])}" and p["c"] != p["y0"] - p["k"] * p["x0"]


def _check_quadratic_func(p: dict, answer: object) -> bool:
    """Вершина — найнижча точка на сітці; нулі — де многочлен = 0; дотик — рівно один нуль."""
    lv = p["level"]
    if lv in (-2, 0):
        coefs = _quad(p) if lv == 0 else [1, -2 * p["h"], p["h"] ** 2 + p["k"]]
        x_best = min(GRID, key=lambda x: _poly(coefs, x))
        return answer == fmt_point(x_best, _poly(coefs, x_best))
    if lv == -1:
        return answer == _poly([1, p["b"], p["c"]], F(p["x0"]))
    if lv == 1:
        coefs = [1, -(p["r1"] + p["r2"]), p["r1"] * p["r2"]]
        zeros = [x for x in GRID if _poly(coefs, x) == 0]
        return len(zeros) == 2 and answer == f"{fmt_num(zeros[0])} і {fmt_num(zeros[1])}"
    zeros = [x for x in GRID if _poly([1, p["p"], answer], x) == 0]
    return len(zeros) == 1


def _check(subtopic: str, params: dict) -> bool:
    """Незалежна перевірка відповіді шаблону для одного завдання."""
    answer = TEMPLATES[subtopic][2](params)["answer"]
    if subtopic == "ineq_system" and params["ask"] == "count":
        cond = _system_condition(params)
        return answer == sum(1 for x in range(-30, 31) if cond(F(x)))
    if subtopic in ("domain", "linear_ineq", "quadratic_ineq", "interval_method", "ineq_system"):
        return _check_set_answer(subtopic, params, answer)
    return _check_function_answer(subtopic, params, answer)


# ----------------------------------------------------------------------------- тести
@pytest.fixture(scope="module")
def bank() -> list:
    """Банк з головним seed — такий самий, як у data/items.json."""
    return items.build_bank()


def test_bank_has_60_items_with_unique_ids_and_questions(bank: list) -> None:
    """60 завдань, без повторів id і тексту умови."""
    assert len(bank) == config.BANK_SIZE == 60
    assert len({it["id"] for it in bank}) == 60
    assert len({it["question"] for it in bank}) == 60


def test_bank_covers_every_subtopic_and_level(bank: list) -> None:
    """Кожна з 9 підтем є на кожному з 5 рівнів складності."""
    pairs = {(it["subtopic"], it["difficulty"]) for it in bank}
    assert pairs == {(s, b) for s in config.SUBTOPICS for b in config.DIFFICULTY_LEVELS}


def test_item_fields(bank: list) -> None:
    """Обов'язкові поля, 4 різні варіанти, номер правильного 0–3, b від −2 до 2."""
    for it in bank:
        for field in ("id", "topic", "difficulty", "question", "options", "answer", "explanation", "params"):
            assert field in it
        assert len(it["options"]) == 4 and len(set(it["options"])) == 4
        assert 0 <= it["answer"] < 4
        assert -2 <= it["difficulty"] <= 2


def test_solution_from_params_matches_correct_option(bank: list) -> None:
    """Розв'язок, обчислений з params, збігається з правильним варіантом; умова теж відтворюється з params."""
    for it in bank:
        assert items.solve(it)["answer"] == it["options"][it["answer"]], it["id"]
        assert TEMPLATES[it["template"]][1](it["params"]) == it["question"]


def test_correct_answer_really_solves_the_task(bank: list) -> None:
    """Незалежна перевірка підстановкою для кожного завдання банку."""
    wrong = [it["id"] for it in bank if not _check(it["subtopic"], it["params"])]
    assert wrong == []


@pytest.mark.parametrize("subtopic", list(config.SUBTOPICS))
def test_generators_are_correct_on_many_random_numbers(subtopic: str) -> None:
    """Стрес-тест генераторів: 40 випадкових завдань на кожен рівень — усі правильні, 4 варіанти."""
    rng = random.Random(123)
    for level in LEVELS:
        for n in range(40):
            item = items.make_item(subtopic, level, rng, f"t{n}")
            assert _check(subtopic, item["params"]), (level, item["question"])
            assert len(set(item["options"])) == 4


def test_solution_has_steps_and_typical_error(bank: list) -> None:
    """Розбір: кілька кроків, останній — «Відповідь: …»; є типова помилка."""
    for it in bank:
        sol = items.solve(it)
        assert len(sol["steps"]) >= 2 and sol["steps"][-1].startswith("Відповідь")
        assert sol["typical_error"]


def test_same_seed_gives_same_bank_and_file_is_current(bank: list, tmp_path) -> None:
    """Той самий seed → той самий банк; data/items.json збігається з генерацією; JSON читається назад."""
    assert items.build_bank() == bank
    assert items.build_bank(seed=7) != bank
    assert items.load_bank(config.ITEMS_PATH) == bank, "перегенеруйте: python -m core.items"
    path = items.save_bank(bank, tmp_path / "bank.json")
    assert items.load_bank(path) == bank


def test_is_correct(bank: list) -> None:
    """is_correct приймає лише номер правильного варіанта."""
    it = bank[0]
    assert items.is_correct(it, it["answer"])
    assert not items.is_correct(it, (it["answer"] + 1) % 4)
