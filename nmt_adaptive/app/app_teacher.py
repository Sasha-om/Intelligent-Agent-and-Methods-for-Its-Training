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
from app import pilot_storage, storage   # noqa: E402
from app.netinfo import class_url   # noqa: E402

FINISHED_NOTE = ("Сесія учня з'являється тут, щойно він її завершить (усі завдання, «Завершити зараз» або «Вийти»)."
                 if config.TEACHER_UNFINISHED_HIDDEN else "")


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


def num(value: float | None, digits: int = 2, sign: bool = False) -> str:
    """Число з десятковою комою; None → «—»."""
    if value is None:
        return "—"
    return f"{value:{'+' if sign else ''}.{digits}f}".replace(".", ",")


TEST_N = len(config.PILOT_TEST_SLOTS)   # завдань у вхідному / вихідному тесті

PILOT_ROWS = (   # рядки таблиці порівняння: (назва, як показати числа режиму)
    ("Сесій у таблиці", lambda a: str(a["chosen"])),
    ("Учнів", lambda a: str(a["students"])),
    ("Завершили вихідний тест", lambda a: str(a["finished"])),
    (f"Вхідний тест, середнє з {TEST_N}", lambda a: num(a["pre"])),
    ("Вхідний тест — лише ті, хто завершив", lambda a: num(a["pre_done"])),
    (f"Вихідний тест, середнє з {TEST_N}", lambda a: num(a["post"])),
    ("Приріст, балів [95 % ДІ]", lambda a: f"{num(a['gain'], sign=True)} [{num(a['gain_low'], sign=True)}; "
                                           f"{num(a['gain_high'], sign=True)}]"),
    ("Нормалізована зміна (−1…1)", lambda a: num(a["norm_gain"], sign=True)),
    ("Кинули до кінця", lambda a: percent(a["dropped"])),
    ("Хвилин на сесію", lambda a: num(a["minutes"], 0)),
)


RESULT_ROWS = ("Вхідний тест — лише ті, хто завершив", f"Вихідний тест, середнє з {TEST_N}", "Приріст, балів [95 % ДІ]",
               "Нормалізована зміна (−1…1)")


def hidden(label: str, data: dict) -> bool:
    """Рядок результатів, коли вихідний тест завершили < 3 учні (інакше це бали однієї-двох людей)."""
    return data.get("results_hidden", False) and label in RESULT_ROWS


def pilot_table(arms: dict) -> pd.DataFrame:
    """Таблиця «показник × режим»; режим з < 3 учнів — лише «обрали: N», решта — «замало даних»."""
    columns = {}
    for arm, data in arms.items():
        name = config.PILOT_ARM_NAMES_UA[arm]
        if data.get("too_few"):
            columns[name] = [str(data["chosen"])] + ["замало даних"] * (len(PILOT_ROWS) - 1)
        else:
            columns[name] = [show(data) if not hidden(label, data) else "замало даних" for label, show in PILOT_ROWS]
    return pd.DataFrame(columns, index=[label for label, _ in PILOT_ROWS])


def comparison_line(c: dict) -> str:
    """«Q-агент − CAT: приріст +0,40 [−0,3; +1,1]; з урахуванням вхідного тесту +0,35 […]; g = 0,30»."""
    adj = c["adjusted"]
    adjusted = "—" if adj is None else (f"{num(adj['effect'], sign=True)} [{num(adj['low'], sign=True)}; "
                                        f"{num(adj['high'], sign=True)}] — "
                                        f"{'значуща різниця' if adj['significant'] else 'різниця незначуща'}")
    return (f"**Q-агент − {config.PILOT_ARM_NAMES_UA[c['other']]}** (завершили: {c['n_q']} і {c['n_other']}): "
            f"приріст {num(c['diff'], sign=True)} [{num(c['low'], sign=True)}; {num(c['high'], sign=True)}] бала; "
            f"з урахуванням вхідного тесту {adjusted}; розмір ефекту g = {num(c['hedges_g'])}")


def show_pilot() -> None:
    """Порівняння режимів: таблиця, графік «вхід → вихід», різниці з довірчими інтервалами."""
    with closing(storage.connect()) as conn:
        agg, counts = pilot_storage.arm_aggregates(conn), pilot_storage.choice_counts(conn)
    st.subheader("Порівняння режимів: вхідний → вихідний тест")
    st.write("Сьогодні обрали: " + " · ".join(f"{config.PILOT_ARM_NAMES_UA[a]} — **{n}**" for a, n in counts.items()))
    st.caption("Таблиця: перша сесія кожного учня; оновлюється пачками — коли в режимі завершать ще 3 учні "
               "(щоб за зміною чисел не було видно балів одного учня). Різниці режимів — серед тих, хто завершив; "
               "ДІ 97,5 % (поправка Бонферроні на 2 порівняння). Режим обирали самі учні, групи невеликі. "
               "Симуляція до уроку: різниця на вихідному тесті більше залежить від утоми, ніж від знань. "
               "Це не оцінки учнів.")
    st.dataframe(pilot_table(agg["arms"]))
    shown = {config.PILOT_ARM_NAMES_UA[a]: d for a, d in agg["arms"].items() if not d.get("too_few") and d["post"]}
    if shown:
        st.line_chart(pd.DataFrame({name: [d["pre_done"], d["post"]] for name, d in shown.items()},
                                   index=["1. вхідний тест", "2. вихідний тест"]))
    for c in agg["compare"]:
        st.markdown(comparison_line(c))


def show_class_status() -> None:
    """Адреса тесту для учнів, скільки зараз проходять тест, кнопка «Оновити»."""
    with closing(storage.connect()) as conn:
        active = storage.in_progress_today(conn)
    left, right = st.columns([3, 1])
    left.write(f"Адреса тесту для учнів (режим класу): **{class_url()}** · зараз проходять тест: **{active}**")
    if right.button("🔄 Оновити", key="refresh"):
        st.rerun()


def show_training() -> None:
    """Тренування: загальна картина класу (або «замало даних»)."""
    with closing(storage.connect()) as conn:
        agg = storage.class_aggregates(conn)
    if not agg:
        st.info(f"Замало даних: результати з'являться, коли тест завершать щонайменше "
                f"{config.MIN_GROUP_FOR_STATS} учні. {FINISHED_NOTE}")
        return
    st.caption(FINISHED_NOTE)
    show_numbers(agg)
    show_topics(agg)


def annul() -> None:
    """Дія кнопки «Анулювати»: архівна копія + очищення бази; галочки знімаються (щоб не натиснути вдруге)."""
    with closing(storage.connect()) as conn:
        archive = storage.annul_all(conn)
    st.session_state.update(annulled=str(archive) if archive else "", annul_sure=False, annul_busy=False)


def show_annul() -> None:
    """«Анулювати результати»: копія бази — в архів, таблиця вчителя починається з нуля (перед справжнім уроком)."""
    with st.expander("⚠️ Анулювати результати"):
        st.write("Усі сесії, тести й коди учнів зникнуть зі сторінки вчителя, і нові уроки рахуватимуться з нуля. "
                 "Копія бази збережеться в папці `data/students/archive/` на цьому комп'ютері (не в git) — "
                 "її можна відкрити звітом `python -m app.pilot_report --db <файл>`.")
        with closing(storage.connect()) as conn:
            active = storage.in_progress_today(conn)
        sure = st.checkbox("Так, анулювати всі результати", key="annul_sure")
        busy = active > 0 and not st.checkbox(f"Зараз проходять тест: {active}. Їхні сесії теж будуть анульовані — "
                                              "я знаю, що учні вже закінчили", key="annul_busy")
        st.button("Анулювати", key="annul_btn", type="primary", disabled=not sure or busy, on_click=annul)
    if "annulled" in st.session_state:
        done = st.session_state.pop("annulled")
        st.success(f"Результати анульовано. Копія: {done}" if done else "База вже порожня — анулювати нічого.")


def main() -> None:
    """Точка входу сторінки вчителя."""
    st.set_page_config(page_title="НМТ: вчитель", page_icon="📊")
    st.title("📊 Пілот: загальна картина класу")
    st.caption("Дані пілоту. Це не оцінки учнів.")
    show_class_status()
    if config.PILOT_COMPARE:
        show_pilot()
        st.divider()
        st.subheader("Тренування: загальна картина")
    show_training()
    st.divider()
    show_annul()


main()
