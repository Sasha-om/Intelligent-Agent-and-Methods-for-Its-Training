"""Сторінка вчителя: лише агреговані дані пілоту, без кодів і таблиць «по учнях».

Запуск (з папки nmt_adaptive):  streamlit run app/app_teacher.py
Якщо учнів менше MIN_GROUP_FOR_STATS — агрегатів немає (щоб за числами не можна було впізнати людину).
"""
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # щоб працювали import config / app
    sys.path.insert(0, str(ROOT))

import pandas as pd         # noqa: E402
import streamlit as st      # noqa: E402

import config               # noqa: E402
from app import storage     # noqa: E402


TODAY_NOTE = ("Сьогоднішні сесії з'являться тут завтра — щоб за зміною чисел не можна було впізнати учня."
              if config.TEACHER_TODAY_HIDDEN else "")


def percent(value: float | None) -> str:
    """0,734 → «73 %»; немає даних → «—»."""
    return "—" if value is None else f"{round(100 * value)} %"


def show_numbers(agg: dict) -> None:
    """Головні числа: учні, сесії, середня кількість завдань, частка достроково завершених."""
    cols = st.columns(4)
    cols[0].metric("Учнів", agg["students"])
    cols[1].metric("Сесій", agg["sessions"])
    cols[2].metric("Завдань за сесію", f"{agg['mean_items']:.1f}".replace(".", ","))
    cols[3].metric("Завершили достроково", percent(agg["dropped_share"]))
    gain = agg["mean_theta_gain"]
    gain_text = "—" if gain is None else f"{gain:+.2f}".replace(".", ",")
    st.write(f"Середній приріст оцінки знань θ̂ за сесію: **{gain_text}** · "
             f"середній час відповіді: **{agg['mean_time_sec']:.0f} с** · "
             f"запасних рішень агента (стан, якого не було в навчанні): **{percent(agg['fallback_share'])}**")


def show_topics(agg: dict) -> None:
    """Частка правильних за підтемами і «слабкі теми класу»."""
    names = {key: name for key, (_group, name) in config.SUBTOPICS.items()}
    accuracy = agg["accuracy_by_subtopic"]
    if not accuracy:
        st.info(f"Жодну підтему ще не розв'язували {config.MIN_GROUP_FOR_STATS} різних учнів.")
        return
    st.subheader("Частка правильних відповідей за підтемами")
    order = sorted(accuracy, key=accuracy.get)
    table = pd.DataFrame({"Підтема": [names[k] for k in order], "Правильних": [100 * accuracy[k] for k in order]})
    bar = st.column_config.ProgressColumn("Правильних", format="%.0f %%", min_value=0, max_value=100)
    st.dataframe(table, hide_index=True, column_config={"Правильних": bar})
    st.subheader("Слабкі теми класу")
    for key in agg["weak_topics"]:
        st.write(f"- {names[key]} — {percent(accuracy[key])} правильних")


def main() -> None:
    """Точка входу сторінки вчителя."""
    st.set_page_config(page_title="НМТ: вчитель", page_icon="📊")
    st.title("📊 Пілот: загальна картина класу")
    st.caption("Дані пілоту. Це не оцінки учнів.")
    with closing(storage.connect()) as conn:
        agg = storage.class_aggregates(conn)
    if not agg:
        st.info(f"Замало даних: агрегати з'являться, коли завдання розв'яжуть щонайменше "
                f"{config.MIN_GROUP_FOR_STATS} учні. {TODAY_NOTE}")
        return
    st.write(f"Сесії до {agg['until']} (не включно). {TODAY_NOTE}")
    show_numbers(agg)
    show_topics(agg)


main()
