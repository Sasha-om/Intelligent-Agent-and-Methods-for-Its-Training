"""Покроковий розбір (етап 11): відповідь як у варіанті, перевірка на прикладі правдива, LaTeX і SVG коректні."""
import random
import re
import xml.etree.ElementTree as ET

import pytest

import config
from core import latex as L
from core.items import load_bank, make_item
from core.walk_common import condition_lines, sample_points
from core.walkthrough import CONDITIONS, solution_set, walkthrough
from core.item_templates import TEMPLATES

LEVELS = (-2, -1, 0, 1, 2)


def _items(n_per_pair: int = 12) -> list:
    """Банк + згенеровані завдання всіх підтем і рівнів."""
    generated = [make_item(s, lv, random.Random(1000 * i + lv), f"t{i}") for i in range(n_per_pair)
                 for s in config.SUBTOPICS for lv in LEVELS]
    return load_bank() + generated


ITEMS = _items()


def _balanced(tex: str) -> bool:
    """Фігурні дужки LaTeX збалансовані (екрановані \\{ \\} не рахуємо)."""
    depth = 0
    for ch in re.sub(r"\\[{}]", "", tex):
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            return False
    return depth == 0


def test_answer_matches_correct_option() -> None:
    """Розбір закінчується тією самою відповіддю, що й правильний варіант."""
    for item in ITEMS:
        w = walkthrough(item)
        assert w["answer"] == item["options"][item["answer"]], item["question"]
        assert w["steps"][-1]["title"] == "Відповідь"


def test_steps_are_numbered_and_explained() -> None:
    """Щонайменше 3 кроки; «Крок 1, 2, …» підряд; у кожного кроку є текст, формула або малюнок."""
    for item in ITEMS:
        steps = walkthrough(item)["steps"]
        numbered = [s["title"] for s in steps if s["title"].startswith("Крок")]
        assert len(steps) >= 3 and numbered
        assert [int(t.split()[1].rstrip(".")) for t in numbered] == list(range(1, len(numbered) + 1))
        assert all(s["text"] or s["math"] or s["figure"] for s in steps)


def test_latex_and_markdown_are_well_formed() -> None:
    """Формули з парними дужками; у тексті парна кількість «$»; без «None» і незаміненого Python-форматування."""
    for item in ITEMS:
        for s in walkthrough(item)["steps"]:
            for line in s["math"]:
                assert _balanced(line) and "None" not in line, line
            assert s["text"].count("$") % 2 == 0, s["text"]
            assert "{L." not in s["text"] and "{p[" not in s["text"]


def test_figures_are_valid_svg() -> None:
    """Кожен малюнок — коректний SVG."""
    for item in ITEMS[:300]:
        for s in walkthrough(item)["steps"]:
            if s["figure"]:
                assert ET.fromstring(s["figure"]).tag.endswith("svg")


def test_check_on_example_is_true() -> None:
    """Перевірка на прикладі: число з відповіді задовольняє умову, число поза відповіддю — ні."""
    checked = 0
    for item in ITEMS:
        template, params = item["template"], item["params"]
        if template not in CONDITIONS:
            continue
        answer = TEMPLATES[template][2](params)["answer"]
        inside, outside = sample_points(solution_set(template, params, answer), tuple(params.get("den", ())))
        assert condition_lines(CONDITIONS[template](params), inside)[1], item["question"]
        if outside is not None:
            assert not condition_lines(CONDITIONS[template](params), outside)[1], item["question"]
        checked += 1
    assert checked > 300


def test_check_step_present_for_inequalities() -> None:
    """У нерівностях і області визначення є крок «Перевірка на прикладі»; у задачах на функції — теж (свій)."""
    for item in ITEMS:
        titles = [s["title"] for s in walkthrough(item)["steps"]]
        if item["template"] in CONDITIONS or item["template"] in ("linear_func", "quadratic_func"):
            has_check = "Перевірка на прикладі" in titles
            simple = item["template"] in ("linear_func", "quadratic_func") and item["params"]["level"] in (-2, -1, 0)
            assert has_check or simple, (item["template"], item["params"]["level"])


@pytest.mark.parametrize("text, expected", [
    ("(−∞; 2] ∪ [5; +∞)", r"$(-\infty ;\, 2] \cup  [5;\, +\infty )$"),
    ("−3/2", r"$-\dfrac{3}{2}$"),
    ("−2 і 3", "$-2$ і $3$"),
    ("y = 2x − 3", "$y = 2x - 3$"),
    ("нулів немає", "нулів немає"),
    ("зростає при x > 0 і спадає при x < 0", "зростає при $x > 0$ і спадає при $x < 0$"),
    ("{4}", r"$\{4\}$"),
])
def test_option_markdown(text, expected) -> None:
    """Варіанти відповідей: математика — у $…$, слова — текстом."""
    assert L.markdown(text) == expected


def test_latex_numbers() -> None:
    """Числа, проміжки, многочлени."""
    from fractions import Fraction
    from core.math_utils import INF
    assert L.num(Fraction(-7, 2)) == r"-\frac{7}{2}" and L.paren(-3) == "(-3)" and L.paren(3) == "3"
    assert L.intervals([(-INF, Fraction(2), False, True)]) == r"(-\infty;\,2]" and L.intervals([]) == r"\varnothing"
    assert L.poly([1, -5, 6]) == "x^{2} - 5x + 6" and L.poly([-1, 0]) == "-x" and L.factor(0) == "x"
