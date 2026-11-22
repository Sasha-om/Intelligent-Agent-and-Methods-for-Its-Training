"""Пояснення рішень агента для учня — простими словами і лише з фактів (без streamlit).

Агент обирає дію за Q-таблицею, вивченою в симуляторі; власних «причин» він не знає.
Тому пояснення складається з трьох частин:
    заголовок — що буде далі (тема і складність);
    «чому»    — факти, які бачив агент і які узгоджуються з дією («правильно 4 з 5 останніх»);
    «що бачив агент» — ті самі 4 частини стану, з яких агент обирав (рівень знань, втома, точність, самопочуття).
"""
import config
from core.env import AdaptiveEnv, decode_state, fatigue_level

FALLBACK_TEXT = "Такої ситуації агент у навчанні не бачив, тому діє за простим запасним правилом."


def items_word(n: int) -> str:
    """«1 завдання», «3 завдання», «5 завдань», «11 завдань», «21 завдання»."""
    if n % 10 == 0 or n % 10 >= 5 or 11 <= n % 100 <= 14:
        return "завдань"
    return "завдання"


def recent_text(env: AdaptiveEnv) -> str:
    """«правильно 4 з 5 останніх» або «» (ще немає відповідей)."""
    recent = list(env.recent)
    return f"правильно {sum(recent)} з {len(recent)} останніх" if recent else ""


def slowed(env: AdaptiveEnv) -> bool:
    """Чи стали відповіді повільнішими (додатковий рівень утоми з часу відповіді)."""
    return fatigue_level(env.since_break, env.times) > fatigue_level(env.since_break, [])


def headline(name: str, env: AdaptiveEnv, item: dict | None) -> str:
    """Заголовок: що буде далі. Для «повторення» — чесно: старт / нова тема / справді слабка тема."""
    if item is None:
        return config.ACTION_TEXT_UA[name]
    topic = f"«{item['subtopic_name']}», складність — {config.LEVEL_NAMES_UA[int(item['difficulty'])]}"
    if name != "review_weak_topic":
        return f"{config.ACTION_TEXT_UA[name]} Тема {topic}."
    if env.items_done == 0:
        return f"Починаємо з теми {topic}."
    if env.estimator.answers(item["subtopic"]) == 0:
        return f"Пробуємо тему, якої ще не було: {topic}."
    return f"{config.ACTION_TEXT_UA[name]} Тема {topic}."


def _level_why(name: str, env: AdaptiveEnv) -> str:
    """«Чому» для легше / те саме / складніше — лише з точності останніх відповідей."""
    acc, level = recent_text(env), env.state_parts()["accuracy"]
    if not acc:
        return "Ти ще не відповідав, тож агент бере рівень, що за стартовою оцінкою тобі підходить."
    if name == "harder":
        return f"У тебе добре виходить ({acc}) — пробуємо трохи вище." if level == 2 else \
            f"Зараз {acc}. Агент пробує трохи складніше, щоб перевірити твою межу."
    if name == "easier":
        return f"Останні завдання давались непросто ({acc}) — трохи легше, щоб закріпити основу." if level == 0 \
            else f"Зараз {acc}. Трохи легше завдання — щоб закріпити впевненість."
    return f"Зараз {acc} — рівень тобі підходить, продовжуємо." if level >= 1 else \
        f"Зараз {acc}. Лишаємось на тому самому рівні, без різких змін."


def _review_why(env: AdaptiveEnv, item: dict, topic_counts: dict) -> str:
    """«Чому» для повторення теми: скільки правильних саме в цій темі."""
    if env.items_done == 0:
        return "Ти ще не відповідав, тож усі теми поки рівні — почнемо з цієї."
    done, right = topic_counts.get(item["subtopic"], (0, 0))
    if done == 0:
        return "Цієї теми ще не було — перевіримо, як вона тобі дається."
    return f"У цій темі поки правильно {right} з {done} — варто повторити."


def _break_why(env: AdaptiveEnv) -> str:
    """«Чому» для перерви: скільки завдань без перерви, уповільнення, енергія."""
    facts = [f"ти розв'язав {env.since_break} {items_word(env.since_break)} без перерви"]
    if slowed(env):
        facts.append("відповіді стали повільнішими")
    if env.self_report is not None and env.self_report <= config.LOW_ENERGY:
        facts.append(f"енергія — {env.self_report} з 5")
    text = "; ".join(facts)
    return f"{text[0].upper()}{text[1:]} — час коротко відпочити, щоб не втомитися."


def why_text(name: str, env: AdaptiveEnv, item: dict | None, topic_counts: dict, fallback: bool) -> str:
    """«Чому так» — факти, узгоджені з дією; запасна дія пояснюється як просте правило."""
    if fallback:
        return FALLBACK_TEXT
    if name == "short_break":
        return _break_why(env)
    text = _review_why(env, item, topic_counts) if name == "review_weak_topic" else _level_why(name, env)
    if env.self_report is not None and env.self_report <= config.LOW_ENERGY:
        text += f" Енергія {env.self_report} з 5 — не поспішай."
    return text


def state_view(state: int) -> list:
    """Стан, з якого обирав агент, словами: [(«Втома», «трохи»), …]."""
    parts = dict(zip(("ability", "fatigue", "accuracy", "self_report"), decode_state(state)))
    return [(config.STATE_NAMES_UA[key][0], config.STATE_NAMES_UA[key][1][value]) for key, value in parts.items()]
