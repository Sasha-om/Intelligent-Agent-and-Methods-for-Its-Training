"""Покроковий розбір завдання для учня: кроки з поясненнями, формулами, малюнками і перевіркою на прикладі.

walkthrough(item) → {"steps": [крок, …], "answer": текст правильної відповіді, "typical_error": текст}.
Кроки нумеруються «Крок 1. …»; «Перевірка на прикладі» і «Відповідь» — без номера.
Відповідь береться з solve() шаблону — тією самою функцією, що будує правильний варіант, тож розбір і варіант
не можуть розійтися; tests/test_walkthrough.py перевіряє, що перевірка на прикладі справді виконується.
"""
from core.item_templates import TEMPLATES, _ineq_set
from core.items import answer_text
from core.math_utils import intersect
from core.walk_common import answer_step, check_step
from core.walk_functions import (domain_conditions, walk_domain, walk_linear_func, walk_monotonic,
                                 walk_quadratic_func, walk_range)
from core.walk_inequalities import (interval_conditions, linear_ineq_conditions, quadratic_ineq_conditions,
                                    system_conditions, walk_ineq_system, walk_interval_method, walk_linear_ineq,
                                    walk_quadratic_ineq)

WALKS = {"domain": walk_domain, "range": walk_range, "monotonic": walk_monotonic, "linear_func": walk_linear_func,
         "quadratic_func": walk_quadratic_func, "linear_ineq": walk_linear_ineq,
         "quadratic_ineq": walk_quadratic_ineq, "interval_method": walk_interval_method,
         "ineq_system": walk_ineq_system}
CONDITIONS = {"domain": domain_conditions, "linear_ineq": linear_ineq_conditions,
              "quadratic_ineq": quadratic_ineq_conditions, "interval_method": interval_conditions,
              "ineq_system": system_conditions}
UNNUMBERED = ("Перевірка на прикладі", "Відповідь")


def solution_set(template: str, params: dict, answer: object) -> list | None:
    """Множина розв'язків для перевірки на прикладі (у системі «скільки цілих» — спільний проміжок)."""
    if template == "ineq_system":
        parts = [_ineq_set(q) for q in params["ineqs"]]
        return intersect(parts[0], parts[1])
    return answer if isinstance(answer, list) else None


def _check(template: str, params: dict, answer: object) -> dict | None:
    """Крок «Перевірка на прикладі» для задач, де відповідь — множина x (нерівності, область визначення)."""
    if template not in CONDITIONS:
        return None
    exclude = tuple(params.get("den", ()))
    return check_step(CONDITIONS[template](params), solution_set(template, params, answer), exclude)


def number(steps: list) -> list:
    """«Крок 1. …», «Крок 2. …»; перевірка і відповідь — без номера."""
    k = 0
    for s in steps:
        if s["title"] not in UNNUMBERED:
            k += 1
            s["title"] = f"Крок {k}. {s['title']}"
    return steps


def walkthrough(item: dict) -> dict:
    """Повний покроковий розбір завдання з банку чи згенерованого."""
    template, params = item["template"], item["params"]
    solution = TEMPLATES[template][2](params)
    steps = WALKS[template](params)
    check = _check(template, params, solution["answer"])
    if check is not None:
        steps.append(check)
    steps.append(answer_step(solution["answer"]))
    return {"steps": number(steps), "answer": answer_text(solution["answer"]),
            "typical_error": solution["typical_error"]}
