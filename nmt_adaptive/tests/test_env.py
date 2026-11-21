"""Тести середовища: стан (135), вибір завдання за дією, кожен доданок нагороди, кінець сесії, абляції."""
import random

import pytest

import config
from core.ability import target_level_for
from core.env import (AdaptiveEnv, accuracy_level, ability_level, decode_state, encode_state, fatigue_level,
                      self_level)
from sim.student_sim import Response, VirtualStudent

A = {name: i for i, name in enumerate(config.ACTIONS)}


class FakeStudent:
    """Учень «за сценарієм»: відповідає з наперед заданого списку; запам'ятовує, що в нього питали."""

    def __init__(self, answers: list, report: int | None = 3, energy: int | None = 3) -> None:
        """answers — список (правильно, час, відмова); report — самооцінка після перерви; energy — з чек-іну."""
        self.answers = list(answers)
        self.report, self.energy = report, energy
        self.asked, self.breaks = [], 0
        self.theta = {s: 0.0 for s in config.SUBTOPICS}

    def checkin(self) -> dict | None:
        """Чек-ін: None — учень пропустив."""
        return None if self.energy is None else {"mood": 3, "energy": self.energy, "sleep": 3}

    def answer(self, subtopic: str, b: float) -> Response:
        """Наступна відповідь зі сценарію; при «правильно» справжня θ підтеми трохи росте."""
        self.asked.append((subtopic, b))
        correct, time_sec, dropped = self.answers.pop(0)
        if correct:
            self.theta[subtopic] += 0.1
        return Response(correct, time_sec, dropped)

    def take_break(self) -> None:
        """Рахує перерви."""
        self.breaks += 1

    def self_report(self) -> int | None:
        """Самооцінка після перерви."""
        return self.report

    def true_theta(self) -> dict:
        """Справжні знання (для бонусу gain_source='true')."""
        return dict(self.theta)


def _ok(n: int, time_sec: float = 60.0) -> list:
    """n правильних відповідей без відмови."""
    return [(True, time_sec, False)] * n


def _env(student: FakeStudent, **kw: object) -> AdaptiveEnv:
    """Середовище з фіксованим seed."""
    return AdaptiveEnv(student, random.Random(0), **kw)


# ----------------------------------------------------------------------------- стан
def test_135_states_and_inverse() -> None:
    """encode/decode — взаємно обернені, номери 0…134 без пропусків."""
    codes = {encode_state(a, f, c, s) for a in range(5) for f in range(3) for c in range(3) for s in range(3)}
    assert codes == set(range(config.N_STATES))
    for code in range(config.N_STATES):
        assert encode_state(*decode_state(code)) == code


def test_ability_buckets() -> None:
    """Межі −1,2; −0,4; 0,4; 1,2 → 5 відер (значення на межі — у верхнє відро)."""
    assert [ability_level(t) for t in (-3, -1.2, -0.5, 0.0, 0.4, 1.19, 1.2, 3)] == [0, 1, 1, 2, 3, 3, 4, 4]


def test_fatigue_by_items_and_time_drift() -> None:
    """< 6 завдань без перерви → 0; 6–11 → 1; ≥ 12 → 2; уповільнення в 1,4 раза → +1 (не більше 2)."""
    assert fatigue_level(5, []) == 0 and fatigue_level(6, []) == 1 and fatigue_level(12, []) == 2
    slow = [60, 60, 60, 90, 90, 90]
    assert fatigue_level(3, slow) == 1
    assert fatigue_level(2, slow) == 0                               # після перерви лише 2 відповіді — дрейф не рахуємо
    assert fatigue_level(3, [60, 60, 60, 70, 70, 70]) == 0
    assert fatigue_level(12, slow) == 2
    assert fatigue_level(2, [60, 60, 90, 90]) == 0                   # замало даних для дрейфу


def test_accuracy_and_self_levels() -> None:
    """Точність останніх 5: <0,4 / <0,8 / ≥0,8; без відповідей — середня. Самооцінка 1–2/3/4–5; без неї — середня."""
    assert accuracy_level([]) == 1
    assert accuracy_level([True, False, False, False, False]) == 0
    assert accuracy_level([True, True, False, False, False]) == 1
    assert accuracy_level([True] * 4 + [False]) == 2
    assert [self_level(v) for v in (1, 2, 3, 4, 5, None)] == [0, 0, 1, 2, 2, config.SELF_REPORT_DEFAULT]


def test_reset_state_uses_checkin() -> None:
    """На старті: знання — середнє відро, бадьорий, точність середня, самооцінка з енергії чек-іну."""
    env = _env(FakeStudent(_ok(1), energy=5))
    assert decode_state(env.reset()) == (2, 0, 1, 2)
    env2 = _env(FakeStudent(_ok(1), energy=None))
    assert decode_state(env2.reset())[3] == config.SELF_REPORT_DEFAULT


# ----------------------------------------------------------------------------- дії → завдання
def test_level_is_relative_to_visible_estimate() -> None:
    """easier/same/harder = рівень «під учня» (з видимої θ̂) −1 / 0 / +1, у межах −2…2; підтеми йдуть по колу."""
    st = FakeStudent(_ok(5) + [(False, 60, False)] * 5)
    env = _env(st)
    env.reset()
    for action in ("harder", "same", "easier", "harder", "harder", "easier", "same", "easier", "easier", "easier"):
        subtopic = env.subtopics[env.pointer % 9]
        base = target_level_for(env.estimator.theta_blended(subtopic))
        expected = max(-2, min(2, base + {"easier": -1, "harder": 1}.get(action, 0)))
        env.step(A[action])
        assert st.asked[-1] == (subtopic, expected)
    subs = [s for s, _b in st.asked]
    order = list(config.SUBTOPICS)
    assert all(order.index(b) == (order.index(a) + 1) % 9 for a, b in zip(subs, subs[1:]))


def test_same_state_same_action_same_relative_level() -> None:
    """Виправлення з рев'ю: дія «same» завжди дає завдання з p̂ біля 0,7 — незалежно від історії рівнів."""
    for answers in (_ok(6), [(False, 60, False)] * 6):
        env = _env(FakeStudent(answers))
        env.reset()
        for _ in range(5):
            env.step(A["harder"])
        info = env.step(A["same"])[3]
        assert abs(info["p_hat"] - config.ZPD_TARGET_P) < 0.3


def test_review_weak_topic_picks_weakest_and_does_not_stick() -> None:
    """review бере найслабшу підтему за змішаною θ̂ і не повторює одну підтему більше REVIEW_MAX_REPEAT разів поспіль."""
    st = FakeStudent([(True, 60, False), (False, 60, False)] + [(False, 60, False)] * 6)
    env = _env(st)
    env.reset()
    env.step(A["same"])
    env.step(A["same"])                                              # друга підтема — помилка
    weak = st.asked[1][0]
    assert env.estimator.weakest_blended() == weak
    env.step(A["review_weak_topic"])
    assert st.asked[2][0] == weak
    for _ in range(5):
        env.step(A["review_weak_topic"])
    subs = [s for s, _b in st.asked]
    runs = max(len(list(g)) for _k, g in __import__("itertools").groupby(subs))
    assert runs <= config.REVIEW_MAX_REPEAT


def test_review_only_session_covers_several_subtopics() -> None:
    """Сесія лише з «повторення» не зводиться до однієї підтеми (знахідка рев'ю)."""
    env = AdaptiveEnv(VirtualStudent("weak_enduring", random.Random(1)), random.Random(1), gain_source="true")
    env.reset()
    seen, done = set(), False
    while not done:
        _s, _r, done, info = env.step(A["review_weak_topic"])
        seen.add(info["item"]["subtopic"])
    assert len(seen) >= 3


def test_break_action() -> None:
    """Перерва: учень відпочиває, лічильник «без перерви» = 0, самооцінка оновлюється; не більше 3 перерв."""
    st = FakeStudent(_ok(10), report=1)
    env = _env(st)
    env.reset()
    for _ in range(7):
        env.step(A["same"])
    assert env.state_parts()["fatigue"] == 1
    state, reward, done, info = env.step(A["short_break"])
    assert st.breaks == 1 and env.state_parts()["fatigue"] == 0 and decode_state(state)[3] == 0
    assert reward == 0 and not done and info["item"] is None
    env.step(A["short_break"]); env.step(A["short_break"])
    assert A["short_break"] not in env.valid_actions()
    with pytest.raises(ValueError):
        env.step(A["short_break"])


# ----------------------------------------------------------------------------- нагорода
def test_reward_zpd() -> None:
    """+1 рівно тоді, коли p̂ (до відповіді) у зоні 0,6–0,8: на старті θ̂ = 0 це рівень 0, а не −1."""
    env = _env(FakeStudent(_ok(2)))
    env.reset()
    _s, r_same, _d, info = env.step(A["same"])
    assert info["in_zpd"] and info["reward_parts"]["zpd"] == pytest.approx(config.R_ZPD)
    assert r_same == pytest.approx(sum(info["reward_parts"].values()))
    _s, r_easy, _d, info2 = env.step(A["easier"])
    assert info2["p_hat"] > config.ZPD_HIGH and info2["reward_parts"]["zpd"] == 0


def test_reward_two_wrong_in_a_row() -> None:
    """−0,5 лише за другу (і кожну наступну) помилку поспіль."""
    env = _env(FakeStudent([(False, 60, False)] * 3))
    env.reset()
    parts = [env.step(A["easier"])[3]["reward_parts"]["two_wrong"] for _ in range(3)]
    assert parts == [0, config.R_TWO_WRONG, config.R_TWO_WRONG]


def test_reward_fatigue_without_break() -> None:
    """−0,3, якщо втома в стані 2 і дія — не перерва; перерва цей штраф не отримує."""
    env = _env(FakeStudent(_ok(20)))
    env.reset()
    for _ in range(12):
        env.step(A["same"])
    assert env.state_parts()["fatigue"] == 2
    assert env.step(A["same"])[3]["reward_parts"]["fatigue"] == config.R_FATIGUE_NO_BREAK
    assert env.step(A["short_break"])[3]["reward_parts"]["fatigue"] == 0


def test_dropout_ends_episode() -> None:
    """Відмова: −3 і кінець епізоду."""
    env = _env(FakeStudent([(True, 60, False), (False, 60, True)]), gain_source="estimate")
    env.reset()
    env.step(A["same"])
    _s, _r, done, info = env.step(A["same"])
    assert done and info["dropped"] and info["reward_parts"]["dropout"] == config.R_DROPOUT


def test_session_ends_after_20_items_with_gain_bonus() -> None:
    """Епізод — 20 завдань (перерви не рахуються); бонус за приріст — на кожному кроці, сума = COEF × приріст."""
    st = FakeStudent(_ok(20))
    env = _env(st, gain_source="true")
    env.reset()
    env.step(A["short_break"])
    bonuses = []
    for i in range(20):
        _s, _r, done, info = env.step(A["same"])
        bonuses.append(info["reward_parts"]["gain"])
        assert done == (i == 19)
    gain = sum(st.theta.values()) / len(st.theta)                   # старт 0 → кінець
    assert sum(bonuses) == pytest.approx(config.R_GAIN_COEF * gain)
    assert bonuses[0] == pytest.approx(config.R_GAIN_COEF * 0.1 / 9)   # крок «правильно» → θ підтеми +0,1
    assert env.session_gain() == pytest.approx(gain)
    with pytest.raises(RuntimeError):
        env.step(A["same"])


def test_gain_source_estimate_uses_general_theta() -> None:
    """gain_source='estimate' — приріст загальної θ̂ (так працює і з реальним учнем)."""
    env = _env(FakeStudent([(True, 60, True)]), gain_source="estimate")
    env.reset()
    info = env.step(A["harder"])[3]
    assert info["reward_parts"]["gain"] == pytest.approx(config.R_GAIN_COEF * env.estimator.general())
    with pytest.raises(ValueError):
        AdaptiveEnv(FakeStudent([]), random.Random(0), gain_source="magic")


# ----------------------------------------------------------------------------- абляції, детермінізм, API
def test_ablations_hide_only_their_part() -> None:
    """Без втоми → частина втоми 0 (але штраф рахується); без самооцінки → 1; без перерви → дії немає."""
    env = _env(FakeStudent(_ok(20), energy=5), use_fatigue=False, use_self_report=False, allow_break=False)
    state = env.reset()
    assert decode_state(state)[3] == 1
    for _ in range(12):
        state = env.step(A["same"])[0]
    assert decode_state(state)[1] == 0
    assert env.step(A["same"])[3]["reward_parts"]["fatigue"] == config.R_FATIGUE_NO_BREAK
    assert A["short_break"] not in env.valid_actions()


def test_same_seed_same_trajectory() -> None:
    """Той самий seed учня і середовища → та сама траєкторія (з віртуальним учнем)."""
    def run() -> list:
        """Одна сесія з випадковими діями, що залежать лише від seed."""
        env = AdaptiveEnv(VirtualStudent("weak_tiring", random.Random(3)), random.Random(4), gain_source="true")
        pick, out, done, state = random.Random(5), [], False, env.reset()
        while not done:
            state, reward, done, info = env.step(pick.choice(env.valid_actions()))
            out.append((state, round(reward, 9), info["action_name"]))
        return out
    assert run() == run()


def test_item_factory_and_split_api() -> None:
    """Для застосунку: next_item(дія) → завдання; apply_answer(відповідь) → крок; item_factory будує повне завдання."""
    made = []
    def factory(subtopic: str, level: int, rng: random.Random) -> dict:
        """Фейкове «повне» завдання."""
        made.append((subtopic, level))
        return {"subtopic": subtopic, "difficulty": float(level), "question": "?"}
    env = AdaptiveEnv(FakeStudent([]), random.Random(0), item_factory=factory)
    env.reset()
    item = env.next_item(A["same"])
    assert item["question"] == "?" and made == [(item["subtopic"], 0)]
    state, reward, done, info = env.apply_answer(Response(True, 50.0, False))
    assert info["correct"] and info["reward_parts"]["zpd"] == pytest.approx(config.R_ZPD) and not done


def test_answer_time_is_normalised_by_difficulty() -> None:
    """Для «дрейфу часу» час ділиться на (2 − p̂): важке завдання саме по собі не виглядає як втома."""
    env = _env(FakeStudent([(True, 90.0, False)]))
    env.reset()
    info = env.step(A["harder"])[3]
    assert env.times[-1] == pytest.approx(90.0 / (2 - info["p_hat"]))
