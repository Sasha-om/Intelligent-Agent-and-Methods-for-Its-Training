"""Банк завдань: створення завдань із шаблонів, 4 варіанти відповіді, збереження в JSON і розв'язок.

Навіщо: агентові потрібні завдання різної складності з різних підтем. Ми не копіюємо реальні
завдання НМТ, а генеруємо свої з шаблонів (core/item_templates.py) з випадковими «гарними» числами.

Поля завдання (словник):
    id, topic, topic_name, subtopic, subtopic_name, difficulty (b від −2 до 2), template, params,
    question, options (4 тексти), answer (номер правильного варіанта 0–3), explanation (один рядок).
Покроковий розв'язок і типова помилка НЕ зберігаються — їх завжди рахує solve(item) з params.

Запуск `python -m core.items` (з папки nmt_adaptive) перегенеровує data/items.json.
"""
import json
import random
from fractions import Fraction
from pathlib import Path

import config
from core.item_templates import TEMPLATES
from core.math_utils import complement, fmt_intervals, fmt_num, mirror, shift, toggle_closed

MAX_ATTEMPTS = 50   # скільки разів пробуємо згенерувати завдання з новим текстом, якщо такий уже є


# ----------------------------------------------------------------------------- відповіді як текст
def answer_text(answer: object) -> str:
    """Об'єкт відповіді → текст варіанта: проміжки, число або готовий рядок."""
    if isinstance(answer, list):
        return fmt_intervals(answer)
    if isinstance(answer, str):
        return answer
    return fmt_num(answer)


def _filler_answers(answer: object, brackets_matter: bool) -> list:
    """Запасні неправильні відповіді, якщо типових помилок не вистачило на 3 варіанти."""
    if isinstance(answer, list):
        extra = [toggle_closed(answer), complement(answer), mirror(answer)] if brackets_matter else []
        return extra + [shift(answer, 1), shift(answer, -1), shift(answer, 2), shift(answer, -2)]
    if isinstance(answer, Fraction):
        return [answer + 1, answer - 1, -answer, answer + 2, 2 * answer, answer - 2]
    return []


def make_options(solution: dict, wrong: list, rng: random.Random) -> tuple:
    """Правильна відповідь + 3 різні неправильні, перемішані. Повертає (варіанти, номер правильного)."""
    correct = answer_text(solution["answer"])
    candidates = wrong + _filler_answers(solution["answer"], solution.get("brackets_matter", True))
    chosen = []
    for cand in candidates:
        text = answer_text(cand)
        if text != correct and text not in chosen:
            chosen.append(text)
        if len(chosen) == config.N_OPTIONS - 1:
            break
    if len(chosen) < config.N_OPTIONS - 1:
        raise ValueError(f"не вистачає неправильних варіантів для відповіді {correct!r}")
    options = chosen + [correct]
    rng.shuffle(options)
    return options, options.index(correct)


# ----------------------------------------------------------------------------- одне завдання
def make_item(subtopic: str, level: int, rng: random.Random, item_id: str) -> dict:
    """Нове завдання підтеми subtopic рівня level (−2…2) з випадковими числами від rng."""
    make, text, solve_fn, wrong_fn = TEMPLATES[subtopic]
    params = make(level, rng)
    solution = solve_fn(params)
    options, answer = make_options(solution, wrong_fn(params), rng)
    group, name = config.SUBTOPICS[subtopic]
    return {"id": item_id, "topic": group, "topic_name": config.TOPIC_GROUPS[group],
            "subtopic": subtopic, "subtopic_name": name, "difficulty": float(level),
            "template": subtopic, "params": params, "question": text(params),
            "options": options, "answer": answer, "explanation": solution["steps"][0]}


def solve(item: dict) -> dict:
    """Розв'язок завдання з його params: {"answer": текст, "steps": [...], "typical_error": текст}.

    Рахується тим самим кодом шаблону, що й умова, тому відповідь завжди відповідає числам в умові.
    """
    _make, _text, solve_fn, _wrong = TEMPLATES[item["template"]]
    solution = solve_fn(item["params"])
    return {"answer": answer_text(solution["answer"]), "steps": solution["steps"],
            "typical_error": solution["typical_error"]}


def is_correct(item: dict, option_index: int) -> bool:
    """Чи обрав учень правильний варіант."""
    return option_index == item["answer"]


# ----------------------------------------------------------------------------- банк
def _plan(size: int) -> list:
    """Порядок (підтема, рівень): кожна пара підтема × рівень трапляється, рівні розподілені рівномірно."""
    subtopics = list(config.SUBTOPICS)
    levels = [int(b) for b in config.DIFFICULTY_LEVELS]
    n = len(subtopics)
    return [(subtopics[i % n], levels[(i // n + i % n) % len(levels)]) for i in range(size)]


def build_bank(seed: int = config.SEED, size: int = config.BANK_SIZE) -> list:
    """Генерує банк із size завдань без повторів тексту умови. Той самий seed → той самий банк."""
    rng = random.Random(seed)
    bank, seen = [], set()
    for number, (subtopic, level) in enumerate(_plan(size), start=1):
        for _attempt in range(MAX_ATTEMPTS):
            item = make_item(subtopic, level, rng, f"it{number:03d}")
            if item["question"] not in seen:
                break
        seen.add(item["question"])
        bank.append(item)
    return bank


def save_bank(bank: list, path: Path = config.ITEMS_PATH) -> Path:
    """Зберігає банк у JSON (UTF-8, з відступами, щоб файл можна було читати очима)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bank, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_bank(path: Path = config.ITEMS_PATH) -> list:
    """Читає банк завдань із JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summary(bank: list) -> dict:
    """Скільки завдань у кожній підтемі й на кожному рівні — для перевірки рівномірності."""
    by_subtopic, by_level = {}, {}
    for item in bank:
        by_subtopic[item["subtopic"]] = by_subtopic.get(item["subtopic"], 0) + 1
        by_level[item["difficulty"]] = by_level.get(item["difficulty"], 0) + 1
    return {"total": len(bank), "by_subtopic": by_subtopic, "by_level": dict(sorted(by_level.items()))}


def main() -> None:
    """Перегенерувати data/items.json і показати короткий підсумок."""
    bank = build_bank()
    path = save_bank(bank)
    info = summary(bank)
    print(f"Збережено {info['total']} завдань → {path}")
    print("За рівнями складності:", info["by_level"])
    for sub, count in info["by_subtopic"].items():
        print(f"  {config.SUBTOPICS[sub][1]}: {count}")
    example = bank[0]
    print("\nПриклад:", example["question"])
    for i, option in enumerate(example["options"]):
        print(f"  {'ABCD'[i]}) {option}{'   ← правильно' if i == example['answer'] else ''}")


if __name__ == "__main__":
    main()
