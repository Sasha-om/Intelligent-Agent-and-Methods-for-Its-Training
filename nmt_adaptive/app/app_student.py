"""Сторінка учня: згода → код → чек-ін → завдання → розбір → рішення агента → перерва → підсумок.

Запуск (з папки nmt_adaptive):  streamlit run app/app_student.py
Streamlit перезапускає цей файл на кожен клік, тому весь стан — у st.session_state,
а логіка — у app/session.py і app/storage.py (їх тестує pytest без браузера).
"""
import json
import random
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # щоб працювали import config / core / app
    sys.path.insert(0, str(ROOT))

import numpy as np          # noqa: E402
import pandas as pd         # noqa: E402
import streamlit as st      # noqa: E402

import config               # noqa: E402
from app import storage     # noqa: E402
from app.session import LiveSession   # noqa: E402
from core.agent import QAgent         # noqa: E402
from core.items import load_bank      # noqa: E402

CONSENT_TEXT = """
**Що це.** Тренажер з математики (функції та нерівності) для підготовки до НМТ. Наступне завдання обирає
програма-агент. Це пілот наукової роботи МАН: ми перевіряємо, чи зручна платформа і чи доречні рішення агента.

**Що записуємо:** випадковий код (не ім'я), твої відповіді, час відповіді, рішення агента, а також — лише якщо
захочеш — настрій, енергію і сон за шкалою 1–5.

**Чого НЕ записуємо:** імені, прізвища, пошти, класу, дати народження, медичних даних.

**Де дані:** лише в файлі на цьому комп'ютері. В інтернет і в git вони не потрапляють.

**Твої права:** можна зупинитися будь-коли. У меню «Мої дані» можна завантажити або видалити все, що записано.
"""
CONSENT_BOX = ("Мені пояснили, як це працює, і мої батьки (або школа) дали згоду на участь."
               if config.REQUIRE_GUARDIAN_CONSENT else "Мені пояснили, як це працює, і я погоджуюсь.")
NO_ANSWER = "не скажу"
SCALE = [NO_ANSWER, *config.CHECKIN_SCALE]


@st.cache_resource
def get_bank() -> list:
    """Банк завдань (лише читання, спільний для всіх учнів)."""
    return load_bank()


def db() -> closing:
    """З'єднання з базою на один перезапуск сторінки: with db() as conn: ..."""
    return closing(storage.connect())


def go(step: str) -> None:
    """Перехід на інший екран."""
    st.session_state.step = step


# ----------------------------------------------------------------------------- 1. згода і 2. код
def page_consent() -> None:
    """Згода: без галочки далі не пускає і нічого не записує."""
    st.markdown(CONSENT_TEXT)
    agreed = st.checkbox(CONSENT_BOX, key="consent_box")
    if st.button("Далі", key="consent_next", disabled=not agreed, type="primary"):
        st.session_state.consent = True
        go("code")
        st.rerun()


def page_code() -> None:
    """Код учня: новий (видаємо й просимо записати) або вже наявний."""
    left, right = st.columns(2)
    with left:
        st.subheader("Я тут уперше")
        if st.button("Отримати код", key="get_code", type="primary"):
            with db() as conn:
                code = storage.new_student_code(conn, random.SystemRandom())
                storage.register_consent(conn, code, guardian_consent=config.REQUIRE_GUARDIAN_CONSENT)
            st.session_state.update(code=code, new_code=True)
            go("checkin")
            st.rerun()
    with right:
        st.subheader("Я вже маю код")
        typed = st.text_input("Твій код", key="code_input", max_chars=config.STUDENT_CODE_LENGTH)
        if st.button("Увійти", key="login"):
            login(typed.strip().upper())


def login(code: str) -> None:
    """Вхід за кодом: код має бути в базі; згоду оновлюємо (галочку щойно поставили знову)."""
    with db() as conn:
        if not storage.student_exists(conn, code):
            st.error("Такого коду немає. Перевір літери або отримай новий код.")
            return
        storage.register_consent(conn, code, guardian_consent=config.REQUIRE_GUARDIAN_CONSENT)
    st.session_state.update(code=code, new_code=False)
    go("checkin")
    st.rerun()


# ----------------------------------------------------------------------------- 3. чек-ін
def page_checkin() -> None:
    """Добровільний чек-ін 1–5. «Пропустити» — агент працює із середньою самооцінкою."""
    if st.session_state.get("new_code"):
        st.success(f"Твій код: **{st.session_state.code}** — запиши його, щоб наступного разу продовжити.")
    st.write("Як ти зараз? (1 — погано, 5 — чудово). Це необов'язково.")
    answers = {key: st.select_slider(label, SCALE, value=NO_ANSWER, key=f"checkin_{key}")
               for key, label in config.CHECKIN_QUESTIONS.items()}
    start, skip = st.columns(2)
    if start.button("Почати", key="checkin_start", type="primary"):
        start_session({k: (None if v == NO_ANSWER else v) for k, v in answers.items()})
    if skip.button("Пропустити", key="checkin_skip"):
        start_session({})


def start_session(checkin: dict) -> None:
    """Нова сесія: запис у базу, свій екземпляр замороженого агента, перше рішення агента."""
    with db() as conn:
        st.session_state.session_id = storage.start_session(conn, st.session_state.code, checkin)
    st.session_state.update(closed=False, new_code=False)
    agent = QAgent.load(config.POLICY_PATH, np.random.default_rng(0))   # ε = 0, без навчання
    st.session_state.live = LiveSession(agent, get_bank(), checkin, seed=random.SystemRandom().randrange(config.SESSION_SEED_MAX))
    next_step()


def next_step() -> None:
    """Агент обирає дію: завдання (секундомір стартує) або перерву (рішення про перерву теж записуємо)."""
    decision = st.session_state.live.decide()
    if decision["action"] == "short_break":
        with db() as conn:
            st.session_state.break_id = storage.save_break(conn, st.session_state.session_id, decision)
    go("break" if decision["action"] == "short_break" else "task")
    st.rerun()


# ----------------------------------------------------------------------------- 4–6. завдання, розбір, рішення агента
def show_decision() -> None:
    """Рішення агента: що далі (жирним), чому (факти) і, за бажанням, стан, з якого він обирав."""
    decision = st.session_state.live.decision
    st.info(f"**{decision['text']}**\n\n{decision['reason']}", icon="🤖")
    with st.expander("Як агент це вирішив?"):
        if decision["fallback"]:
            st.write("Такої ситуації агент у навчанні не бачив, тому діє просте запасне правило. Він бачив:")
        else:
            st.write("Агент бачить 4 речі і для кожної ситуації має вивчену найкращу дію "
                     f"(навчався на {config.TRAIN_EPISODES:,} сесіях з віртуальними учнями):".replace(",", " "))
        st.markdown("\n".join(f"- {label}: **{value}**" for label, value in decision["state_view"]))


def page_task() -> None:
    """Завдання з 4 варіантами. Час іде від показу завдання до натискання «Відповісти»."""
    live = st.session_state.live
    item = live.item
    show_decision()
    st.caption(f"Завдання {live.env.items_done + 1} з {config.SESSION_ITEMS} · {item['subtopic_name']}")
    st.markdown(f"#### {item['question']}")
    labels = [f"{'АБВГ'[i]}) {text}" for i, text in enumerate(item["options"])]
    choice = st.radio("Обери відповідь:", range(len(labels)), format_func=labels.__getitem__, index=None,
                      key=f"answer_{live.env.items_done}")
    if st.button("Відповісти", key="answer_btn", disabled=choice is None, type="primary"):
        out = live.answer(choice)
        save_answer(out["record"])
        if live.done:            # 20-те завдання: сесію закриваємо одразу, навіть якщо учень не натисне «Підсумок»
            close_session(dropped=False)
        st.session_state.feedback = out["feedback"]
        go("feedback")
        st.rerun()
    stop_button()


def save_answer(record: dict) -> None:
    """Запис відповіді; якщо база зайнята чи недоступна — повідомлення, але сесія не ламається."""
    try:
        with db() as conn:
            storage.save_answer(conn, st.session_state.session_id, record)
    except sqlite3.Error:
        st.session_state.save_error = True


def page_feedback() -> None:
    """Розбір після відповіді: правильно — один рядок; помилка — кроки і типова помилка."""
    fb = st.session_state.feedback
    if st.session_state.pop("save_error", False):
        st.error("Цю відповідь не вдалося записати (база зайнята). Можна продовжувати.")
    if fb["correct"]:
        st.success(f"✅ Правильно! {fb['text']}")
    else:
        st.error(f"❌ Помилка. Твоя відповідь: {fb['chosen']}. Правильна: **{fb['answer']}**")
        st.markdown("**Розв'язок:**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(fb["steps"], 1)))
        st.warning(f"Типова помилка: {fb['typical_error']}")
    if st.session_state.live.done:
        if st.button("Підсумок", key="to_summary", type="primary"):
            finish(dropped=False)
    else:
        if st.button("Далі", key="feedback_next", type="primary"):
            next_step()
        stop_button()


def page_break() -> None:
    """Перерва, яку запропонував агент. Повернутися можна раніше."""
    show_decision()
    st.subheader(f"☕ Перерва {config.BREAK_MINUTES} хвилини")
    st.write("Встань, розімнися, попий води. Коли будеш готовий — натисни «Продовжити».")
    energy = st.select_slider("Як енергія зараз? (необов'язково)", SCALE, value=NO_ANSWER, key="break_energy")
    if st.button("Продовжити", key="break_continue", type="primary"):
        report = None if energy == NO_ANSWER else energy
        st.session_state.live.take_break(report)
        with db() as conn:
            storage.break_taken(conn, st.session_state.break_id, report)
        next_step()
    stop_button()


def stop_button() -> None:
    """Учень може зупинитися будь-коли (сесія рахується достроково завершеною)."""
    if st.button("Завершити зараз", key="stop"):
        finish(dropped=True)


def close_session(dropped: bool) -> None:
    """Записує кінець сесії в базу (лише один раз)."""
    if st.session_state.get("closed"):
        return
    with db() as conn:
        storage.end_session(conn, st.session_state.session_id, dropped, st.session_state.live.stats())
    st.session_state.closed = True


def finish(dropped: bool) -> None:
    """Кінець сесії: запис у базу (якщо ще не записано) і підсумок."""
    close_session(dropped)
    go("summary")
    st.rerun()


# ----------------------------------------------------------------------------- 8. підсумок
def page_summary() -> None:
    """Підсумок сесії і прогрес за днями."""
    s = st.session_state.live.summary()
    st.subheader("Підсумок")
    st.metric("Правильних відповідей", f"{s['right']} з {s['items']}")
    if s["good"]:
        st.success("Вдалося: " + ", ".join(s["names"][t] for t in s["good"]))
    if s["weak"]:
        st.warning("Варто повторити: " + ", ".join(s["names"][t] for t in s["weak"]))
    with db() as conn:
        history = storage.student_history(conn, st.session_state.code)
    if len(history) > 1:      # прогрес має сенс між днями (дні без відповідей не показуємо)
        st.write("Частка правильних відповідей за днями:")
        st.line_chart(pd.DataFrame({"частка правильних": [h["accuracy"] for h in history]},
                                   index=[h["day"] for h in history]))
    again, leave = st.columns(2)
    if again.button("Нова сесія", key="again", type="primary"):
        go("checkin")
        st.rerun()
    if leave.button("Вийти (наступний учень)", key="logout_summary"):
        logout()


# ----------------------------------------------------------------------------- 9. мої дані
def sidebar_my_data() -> None:
    """Меню «Мої дані»: експорт (JSON) і видалення з підтвердженням."""
    code = st.session_state.get("code")
    if not code:
        return
    st.sidebar.header("Мої дані")
    shown = code if st.sidebar.toggle("Показати мій код", key="show_code") else "•" * len(code)
    st.sidebar.write(f"Код: **{shown}**")
    if st.sidebar.button("Вийти (наступний учень)", key="logout"):
        logout()
    with db() as conn:
        data = storage.student_data(conn, code)
    st.sidebar.download_button("Експорт моїх даних", json.dumps(data, ensure_ascii=False, indent=2),
                               file_name=f"nmt_{code}.json", mime="application/json", key="export")
    st.sidebar.caption("На спільному комп'ютері видали файл із «Завантажень» після перегляду.")
    sure = st.sidebar.checkbox("Так, видалити все назавжди", key="delete_sure")
    if st.sidebar.button("Видалити мої дані", key="delete", disabled=not sure):
        with db() as conn:
            storage.delete_student(conn, code)
        st.session_state.clear()
        st.session_state.deleted = True
        st.rerun()


def logout() -> None:
    """Вихід: незавершена сесія закривається як дострокова, стан сторінки очищується (для спільного ПК)."""
    if st.session_state.get("live") is not None and not st.session_state.get("closed"):
        close_session(dropped=True)
    st.session_state.clear()
    st.rerun()


SCREENS = {"consent": page_consent, "code": page_code, "checkin": page_checkin, "task": page_task,
           "feedback": page_feedback, "break": page_break, "summary": page_summary}


def main() -> None:
    """Точка входу сторінки учня."""
    st.set_page_config(page_title="Тренажер НМТ", page_icon="📐")
    st.title("📐 Тренажер НМТ: функції та нерівності")
    st.session_state.setdefault("step", "consent")
    if st.session_state.pop("deleted", False):
        st.success("Усі твої дані видалено.")
    sidebar_my_data()
    SCREENS[st.session_state.step]()


main()
