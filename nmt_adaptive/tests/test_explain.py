"""Тести пояснень рішень агента (app/explain.py): лише факти, узгоджені з дією, простими словами."""
import random

import config
from app.explain import FALLBACK_TEXT, headline, items_word, state_view, why_text
from app.session import LiveStudent
from core.env import AdaptiveEnv, encode_state


def _env(answers: list, energy: int | None = None, since_break: int | None = None) -> AdaptiveEnv:
    """Середовище з «історією»: останні відповіді (True/False), енергія, завдань без перерви."""
    env = AdaptiveEnv(LiveStudent({"energy": energy} if energy else {}), random.Random(0))
    env.reset()
    env.recent.extend(answers)
    env.items_done = len(answers)
    env.since_break = len(answers) if since_break is None else since_break
    return env


ITEM = {"subtopic": "domain", "subtopic_name": "Область визначення", "difficulty": 1.0}


def test_items_word() -> None:
    """Відмінювання слова «завдання»."""
    assert [items_word(n) for n in (1, 2, 4, 5, 11, 12, 21, 22, 25)] == [
        "завдання", "завдання", "завдання", "завдань", "завдань", "завдань", "завдання", "завдання", "завдань"]


def test_headline_names_topic_and_level() -> None:
    """Заголовок: дія + тема + складність словами."""
    text = headline("harder", _env([True]), ITEM)
    assert text.startswith(config.ACTION_TEXT_UA["harder"]) and "«Область визначення»" in text and "складність — висока" in text


def test_headline_review_is_honest() -> None:
    """Повторення: на старті — «Починаємо», тема без відповідей — «якої ще не було», інакше — «найслабша»."""
    assert headline("review_weak_topic", _env([]), ITEM).startswith("Починаємо з теми")
    env = _env([True])
    assert "якої ще не було" in headline("review_weak_topic", env, ITEM)
    env.estimator.update("domain", 0.0, False)
    assert "найслабша" in headline("review_weak_topic", env, ITEM)


def test_harder_praise_only_when_accuracy_high() -> None:
    """«Добре виходить» — лише коли точність справді висока; інакше нейтрально, з тими самими числами."""
    good = why_text("harder", _env([True] * 5), ITEM, {}, False)
    poor = why_text("harder", _env([False, False, True]), ITEM, {}, False)
    assert "добре виходить" in good and "правильно 5 з 5 останніх" in good
    assert "добре виходить" not in poor and "правильно 1 з 3 останніх" in poor


def test_easier_mentions_difficulty_only_when_low() -> None:
    """«Давались непросто» — лише коли точність низька."""
    low = why_text("easier", _env([False, False, False, True]), ITEM, {}, False)
    mid = why_text("easier", _env([True, True, False]), ITEM, {}, False)
    assert "непросто" in low and "непросто" not in mid


def test_review_why_uses_topic_counts() -> None:
    """Повторення: «у цій темі правильно 1 з 3» або «цієї теми ще не було»."""
    env = _env([True, False])
    assert "правильно 1 з 3" in why_text("review_weak_topic", env, ITEM, {"domain": (3, 1)}, False)
    assert "ще не було" in why_text("review_weak_topic", env, ITEM, {}, False)


def test_break_why_and_low_energy() -> None:
    """Перерва: скільки завдань без перерви, енергія (якщо низька)."""
    text = why_text("short_break", _env([True] * 5, energy=2, since_break=6), None, {}, False)
    assert text.startswith("Ти розв'язав 6 завдань без перерви") and "енергія — 2 з 5" in text
    assert "не поспішай" in why_text("same", _env([True], energy=1), ITEM, {}, False)
    assert "не поспішай" not in why_text("same", _env([True], energy=4), ITEM, {}, False)


def test_start_and_fallback() -> None:
    """Без відповідей — пояснення про стартову оцінку; запасна дія — чесно «запасне правило»."""
    assert "ще не відповідав" in why_text("same", _env([]), ITEM, {}, False)
    assert why_text("harder", _env([True] * 5), ITEM, {}, True) == FALLBACK_TEXT


def test_state_view_words() -> None:
    """Стан словами: 4 рядки в порядку знання, втома, точність, самопочуття."""
    view = state_view(encode_state(4, 2, 0, 1))
    assert view == [("Рівень знань (оцінка)", "високий"), ("Втома", "помітна"),
                    ("Точність останніх відповідей", "низька"), ("Самопочуття", "середнє або не вказано")]


def test_every_action_has_text() -> None:
    """Для кожної дії й кожного рівня точності пояснення не порожнє."""
    for answers in ([], [False] * 5, [True, False], [True] * 5):
        for name in config.ACTIONS:
            item = None if name == "short_break" else ITEM
            env = _env(answers, since_break=max(1, len(answers)))
            assert why_text(name, env, item, {}, False).strip() and headline(name, env, item).strip()
