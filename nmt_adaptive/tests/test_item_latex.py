"""Умова для показу (етап 11): формула з тих самих чисел, що й текст умови; система — з фігурною дужкою."""
import random
import re

import config
from core.item_latex import question_view
from core.items import load_bank, make_item


def _numbers(text: str) -> list:
    """Усі числа тексту чи LaTeX (дроби \\frac{a}{b} → a/b; показник ² і ^{2} не рахуємо)."""
    text = text.replace("−", "-").replace("^{2}", "").replace("²", "")
    text = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", text)
    return sorted(re.findall(r"\d+", text))


def test_formula_has_same_numbers_as_question() -> None:
    """Для банку і 2250 згенерованих завдань числа в «тексті + формулі» ті самі, що в умові банку."""
    items = load_bank() + [make_item(s, lv, random.Random(i), "g") for i in range(50)
                           for s in config.SUBTOPICS for lv in (-2, -1, 0, 1, 2)]
    for item in items:
        view = question_view(item)
        shown = f"{view['lead']} {view['formula']} {view['tail']}"
        assert _numbers(item["question"]) == _numbers(shown), item["question"]


def test_system_uses_cases_brace() -> None:
    """Система нерівностей — \\begin{cases} … \\\\ … \\end{cases}, а не «{ …; … }»."""
    item = make_item("ineq_system", 0, random.Random(0), "s")
    formula = question_view(item)["formula"]
    assert formula.startswith(r"\begin{cases}") and r"\\" in formula and ";" not in formula


def test_fractions_and_roots_are_typeset() -> None:
    """Дроби — \\frac, корені — \\sqrt{…} (без «/» і «√»)."""
    for template, level in (("interval_method", 2), ("domain", 1), ("linear_ineq", 2), ("domain", 2)):
        formula = question_view(make_item(template, level, random.Random(3), "f"))["formula"]
        assert "/" not in formula and "√" not in formula
        assert r"\frac" in formula or r"\sqrt" in formula
