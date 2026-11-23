"""Віртуальний учень для симуляції: приховані знання, втома, відмова, шумна самооцінка.

Навіщо: Q-навчанню потрібні десятки тисяч сесій, а живих учнів для цього немає (і їх не можна
робити «піддослідними» під час випадкових дій агента). Тому агент вчиться на віртуальних учнях.

Важливо: параметри симулятора (config.SIM_*) — ПРИПУЩЕННЯ, а не виміри реальних учнів.
Агент бачить лише те, що бачив би й у живого учня: правильність, час відповіді, самооцінку, чек-ін.
Справжні знання й втома (true_theta, fatigue) — лише для метрик симуляції.

Модель:
    θ_eff = θ − k·F,   P = c + (1 − c)·σ(a·(θ_eff − b))       (та сама формула, що в core/ability.py)
    учень вчиться (θ += швидкість), лише якщо 0,5 ≤ P ≤ 0,85 — «зона найближчого розвитку»
    втома F росте з кожним завданням і падає після перерви
"""
import copy
import random
from dataclasses import dataclass

import config
from core.ability import p_correct


@dataclass(frozen=True)
class Response:
    """Що видно після відповіді: правильність, час (секунди) і чи кинув учень сесію після неї."""
    correct: bool
    time_sec: float
    dropped: bool


def dropout_chance(wrong_streak: int, fatigue: float) -> float:
    """Шанс, що учень кине сесію на цьому кроці: росте від серії помилок і високої втоми.

    Одна помилка ще не серія: штраф рахується з другої помилки поспіль.
    """
    chance = config.SIM_DROPOUT_BASE + config.SIM_DROPOUT_STREAK * max(0, wrong_streak - 1)
    if fatigue > config.SIM_DROPOUT_FATIGUE_LEVEL:
        chance += config.SIM_DROPOUT_FATIGUE
    return min(config.SIM_DROPOUT_MAX, chance)


def _clip(value: float, low: float, high: float) -> float:
    """Обмежити число межами [low; high]."""
    return max(low, min(high, value))


def _scale_1_5(value: float) -> int:
    """Округлити й обмежити до шкали чек-іну 1–5."""
    return int(_clip(round(value), 1, 5))


class VirtualStudent:
    """Віртуальний учень одного з 4 типів (config.STUDENT_TYPES)."""

    def __init__(self, type_key: str, rng: random.Random) -> None:
        """Створює учня: справжні знання по підтемах, швидкість втоми, стартова втома — з rng."""
        if type_key not in config.STUDENT_TYPES:
            raise KeyError(f"Невідомий тип учня {type_key!r}. Доступні: {', '.join(config.STUDENT_TYPES)}")
        kind = config.STUDENT_TYPES[type_key]
        self.type_key = type_key
        self.rng = rng
        self._theta = {s: _clip(rng.gauss(kind["theta_mean"], config.SIM_THETA_SD), config.THETA_MIN, config.THETA_MAX)
                       for s in config.SUBTOPICS}
        self.fatigue_rate = max(config.SIM_MIN_FATIGUE_RATE, rng.gauss(kind["fatigue_rate"], config.SIM_FATIGUE_RATE_SD))
        self._start_fatigue = rng.uniform(0.0, config.SIM_START_FATIGUE_MAX)
        self._fatigue = self._start_fatigue
        self.wrong_streak = 0

    # ------------------------------------------------------------------ те, що бачить агент
    def answer(self, subtopic: str, b: float) -> Response:
        """Учень розв'язує завдання підтеми складності b: відповідь, час, можлива відмова."""
        p = self.true_probability(subtopic, b)
        correct = self.rng.random() < p
        time_sec = self._answer_time(p)
        if config.SIM_LEARN_LOW <= p <= config.SIM_LEARN_HIGH:
            self._theta[subtopic] = min(config.THETA_MAX, self._theta[subtopic] + config.SIM_LEARN_RATE)
        self._fatigue = min(1.0, self._fatigue + self.fatigue_rate)
        self.wrong_streak = 0 if correct else self.wrong_streak + 1
        dropped = self.rng.random() < dropout_chance(self.wrong_streak, self._fatigue)
        return Response(correct=correct, time_sec=time_sec, dropped=dropped)

    def _answer_time(self, p: float) -> float:
        """Час відповіді: довший для важкого завдання (мала P) і для втомленого учня, з шумом."""
        noise = self.rng.lognormvariate(0.0, config.SIM_TIME_NOISE)
        return config.SIM_BASE_TIME * (1.0 + (1.0 - p)) * (1.0 + self._fatigue) * noise

    def take_break(self) -> None:
        """Коротка перерва знімає частину накопиченої втоми."""
        self._fatigue *= 1.0 - config.SIM_BREAK_RECOVERY

    def rest_overnight(self) -> None:
        """«Наступний день»: утома повертається до стартової, серія помилок скидається (знання лишаються)."""
        self._fatigue = self._start_fatigue
        self.wrong_streak = 0

    def self_report(self) -> int:
        """Самооцінка стану 1–5 (5 — бадьорий): шумна версія прихованої втоми."""
        return _scale_1_5(5 - 4 * self._fatigue + self.rng.gauss(0.0, config.SIM_SELF_REPORT_NOISE))

    def checkin(self) -> dict:
        """Чек-ін на початку сесії: настрій, енергія, сон (1–5), кожен зі своїм шумом."""
        noise = config.SIM_SELF_REPORT_NOISE
        came_tired = self._start_fatigue / config.SIM_START_FATIGUE_MAX     # 0 — виспався, 1 — прийшов утомленим
        return {"mood": _scale_1_5(4 - 2 * self._fatigue + self.rng.gauss(0.0, noise)),
                "energy": self.self_report(),
                "sleep": _scale_1_5(5 - 2 * came_tired + self.rng.gauss(0.0, noise))}

    # ------------------------------------------------------------------ лише для метрик симуляції
    def true_theta(self) -> dict:
        """Справжні знання по підтемах (копія). Агент цього НЕ бачить."""
        return dict(self._theta)

    def true_probability(self, subtopic: str, b: float) -> float:
        """Справжня ймовірність правильної відповіді з урахуванням втоми."""
        theta_eff = self._theta[subtopic] - config.SIM_FATIGUE_K * self._fatigue
        return p_correct(theta_eff, b)

    def fatigue(self) -> float:
        """Прихована втома 0…1."""
        return self._fatigue


def make_population(type_key: str, n: int, seed: int) -> list:
    """n різних учнів одного типу; той самий seed → ті самі учні."""
    master = random.Random(seed)
    return [VirtualStudent(type_key, random.Random(master.randrange(2**31))) for _ in range(n)]


def clone(student: VirtualStudent, seed: int) -> VirtualStudent:
    """Точна копія учня (знання, втома, серія) з новим генератором Random(seed).

    Навіщо: на етапі 8 усі стратегії мають працювати з ТИМ САМИМ учнем — чесне парне порівняння.
    Оригінал не змінюється.
    """
    twin = copy.deepcopy(student)
    twin.rng = random.Random(seed)
    return twin
