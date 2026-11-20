"""Тести допоміжної математики: текст чисел, метод інтервалів, перетин, доповнення."""
from fractions import Fraction as F

from core.math_utils import (INF, MINUS, complement, contains, count_integers, fmt_factor, fmt_intervals,
                             fmt_num, fmt_poly, interval_signs, intersect, mirror, ray, sign_method,
                             solve_linear, union)


def test_numbers_are_printed_exactly() -> None:
    """Дроби не округлюються: 3/2 лишається 3/2."""
    assert fmt_num(3) == "3"
    assert fmt_num(F(-3, 2)) == f"{MINUS}3/2"
    assert fmt_num(F(4, 2)) == "2"


def test_polynomials_and_factors() -> None:
    """x² − 5x + 6, −x + 2, (x + 3)."""
    assert fmt_poly([1, -5, 6]) == f"x² {MINUS} 5x + 6"
    assert fmt_poly([-1, 2]) == f"{MINUS}x + 2"
    assert fmt_poly([0, 0]) == "0"
    assert fmt_factor(-3) == "(x + 3)"
    assert fmt_factor(0) == "x"


def test_sign_method_quadratic_outside() -> None:
    """(x − 2)(x − 5) ≥ 0 → (−∞; 2] ∪ [5; +∞)."""
    ans = sign_method([2, 5], [], ">=")
    assert fmt_intervals(ans) == f"({MINUS}∞; 2] ∪ [5; +∞)"


def test_sign_method_excludes_denominator() -> None:
    """(x − 1)/(x − 4) ≤ 0 → [1; 4): нуль знаменника не входить навіть при «≤»."""
    assert fmt_intervals(sign_method([1], [4], "<=")) == "[1; 4)"


def test_interval_signs_alternate() -> None:
    """Для трьох різних коренів знаки чергуються: −, +, −, + (крайній правий завжди +)."""
    assert interval_signs([1, 2, 3]) == [MINUS, "+", MINUS, "+"]


def test_linear_flip_on_negative() -> None:
    """−2x + 1 ≥ 5 → x ≤ −2 (знак змінюється)."""
    op, bound = solve_linear(-2, 1, 5, ">=")
    assert op == "<=" and bound == -2


def test_intersect_keeps_strongest_brackets() -> None:
    """[1; 4] ∩ (1; 3] = (1; 3]; [1; 4] ∩ [4; 6) = {4}; [1; 2) ∩ (2; 3] = ∅."""
    assert fmt_intervals(intersect([(1, 4, True, True)], [(1, 3, False, True)])) == "(1; 3]"
    assert fmt_intervals(intersect([(1, 4, True, True)], [(4, 6, True, False)])) == "{4}"
    assert intersect([(1, 2, True, False)], [(2, 3, False, True)]) == []


def test_complement_and_union() -> None:
    """Доповнення [2; 5) — це (−∞; 2) ∪ [5; +∞); об'єднання двох променів."""
    assert fmt_intervals(complement([(2, 5, True, False)])) == f"({MINUS}∞; 2) ∪ [5; +∞)"
    assert union(ray(">", 3), ray("<", 1)) == [(-INF, 1, False, False), (3, INF, False, False)]
    assert complement([(-INF, INF, False, False)]) == []


def test_mirror_and_contains() -> None:
    """Дзеркало [2; +∞) — це (−∞; −2]; contains враховує дужки."""
    assert fmt_intervals(mirror(ray(">=", 2))) == f"({MINUS}∞; {MINUS}2]"
    assert contains([(1, 3, True, False)], 1) and not contains([(1, 3, True, False)], 3)


def test_count_integers() -> None:
    """(−1; 3] містить 0, 1, 2, 3 — чотири цілих; промінь — нескінченно багато."""
    assert count_integers([(-1, 3, False, True)]) == 4
    assert count_integers([(F(1, 2), F(5, 2), True, True)]) == 2
    assert count_integers(ray(">", 0)) is None
