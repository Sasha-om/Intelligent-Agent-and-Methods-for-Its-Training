"""Оцінка знань учня: θ̂ по кожній підтемі та ймовірність правильної відповіді p̂.

Модель відповіді (IRT з вгадуванням, «3PL»):
    p(θ, b) = c + (1 − c) · σ(a · (θ − b)),   σ(z) = 1 / (1 + e^(−z))
θ — знання учня, b — складність завдання (обидва на одній шкалі від −3 до 3),
a — наскільки різко p зростає зі знаннями, c — шанс вгадати (1 з 4 варіантів).

Оновлення після відповіді — як рейтинг у шахах (Elo):
    θ̂ ← θ̂ + K · (відповідь − p̂),   відповідь = 1 (правильно) або 0 (ні)
Різниця «відповідь − p̂» — це «здивування» системи: чим несподіваніший результат, тим більший крок.

Навіщо: θ̂ дає агентові «рівень знань» у стані, p̂ — нагороду за зону розвитку (0,6–0,8),
а найслабша підтема — дію «повторення слабкої теми».
"""
import math
import random

import config


def sigmoid(z: float) -> float:
    """Логістична функція σ(z) = 1/(1 + e^(−z)). Записана так, щоб не переповнюватися при великих |z|."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def p_correct(theta: float, b: float, a: float = config.IRT_A, c: float = config.GUESS_C) -> float:
    """Ймовірність правильної відповіді учня зі знаннями theta на завдання складності b.

    Нею користуються і оцінювач (з оціненою θ̂), і симулятор (зі справжньою θ, якої агент не бачить).
    """
    return c + (1.0 - c) * sigmoid(a * (theta - b))


def best_level_for(theta: float) -> float:
    """Рівень складності з DIFFICULTY_LEVELS, найближчий до theta (так обирає завдання стратегія CAT)."""
    return min(config.DIFFICULTY_LEVELS, key=lambda level: (abs(level - theta), level))


def target_level_for(theta: float, p: float = config.ZPD_TARGET_P) -> float:
    """Рівень «під учня»: найближчий до складності, на якій учень із theta відповість з імовірністю p.

    Зона розвитку несиметрична (θ − b від −0,09 до 0,67), тому ціль — p ≈ 0,7, а не θ = b.
    З формули 3PL: b* = θ − logit((p − c)/(1 − c)) / a.
    """
    q = (p - config.GUESS_C) / (1.0 - config.GUESS_C)
    return best_level_for(theta - math.log(q / (1.0 - q)) / config.IRT_A)


def _clip(value: float) -> float:
    """Обмежити θ̂ межами шкали, щоб оцінка не «втекла» після довгої серії."""
    return max(config.THETA_MIN, min(config.THETA_MAX, value))


class AbilityEstimator:
    """Оцінки знань θ̂ одного учня по всіх підтемах (окремо для кожної)."""

    def __init__(self, subtopics: tuple = tuple(config.SUBTOPICS), prior: float = config.THETA_PRIOR,
                 k: float = config.ELO_K) -> None:
        """Усі θ̂ = prior (поки нічого не знаємо); k — крок оновлення."""
        self._order = list(subtopics)
        self._theta = {s: float(prior) for s in self._order}
        self._answers = {s: 0 for s in self._order}
        self._prior = float(prior)
        self._general = float(prior)       # загальна θ̂ учня: оновлюється після КОЖНОЇ відповіді
        self._k = k

    def _check(self, subtopic: str) -> None:
        """Зрозуміла помилка для невідомої підтеми (замість тихого створення нової)."""
        if subtopic not in self._theta:
            raise KeyError(f"Невідома підтема {subtopic!r}. Доступні: {', '.join(self._order)}")

    def theta(self, subtopic: str) -> float:
        """Поточна оцінка θ̂ підтеми."""
        self._check(subtopic)
        return self._theta[subtopic]

    def answers(self, subtopic: str) -> int:
        """Скільки відповідей уже було з цієї підтеми."""
        self._check(subtopic)
        return self._answers[subtopic]

    def p_hat(self, subtopic: str, b: float) -> float:
        """Оцінена ймовірність, що учень правильно розв'яже завдання підтеми складності b."""
        return p_correct(self.theta(subtopic), b)

    def update(self, subtopic: str, b: float, correct: bool) -> float:
        """Оновлює θ̂ після відповіді. Повертає p̂ ДО оновлення — саме вона йде в нагороду агента."""
        p_before = self.p_hat(subtopic, b)
        y = 1.0 if correct else 0.0
        self._theta[subtopic] = _clip(self._theta[subtopic] + self._k * (y - p_before))
        self._general = _clip(self._general + self._k * (y - p_correct(self._general, b)))
        self._answers[subtopic] += 1
        return p_before

    def general(self) -> float:
        """Загальна θ̂ учня (по всіх підтемах разом). Іде в «рівень знань» стану агента."""
        return self._general

    def theta_blended(self, subtopic: str, m: int = config.ABILITY_BLEND_M) -> float:
        """Змішана θ̂ підтеми: поки відповідей з неї мало, більше довіряємо загальній θ̂.

        θ = w·θ̂_підтеми + (1 − w)·θ̂_загальна,  w = n / (n + m),  n — відповідей з підтеми.
        Навіщо: за сесію з однієї підтеми лише 2–3 відповіді, і її власна θ̂ майже не відходить від 0.
        """
        n = self.answers(subtopic)
        w = n / (n + m)
        return w * self._theta[subtopic] + (1.0 - w) * self._general

    def p_hat_blended(self, subtopic: str, b: float) -> float:
        """p̂ зі змішаної θ̂ — саме вона йде в нагороду агента за зону розвитку."""
        return p_correct(self.theta_blended(subtopic), b)

    def _answered(self) -> list:
        """Підтеми, з яких уже була хоч одна відповідь (у порядку config)."""
        return [s for s in self._order if self._answers[s] > 0]

    def overall(self) -> float:
        """Загальний рівень: середня θ̂ по підтемах з відповідями; без відповідей — стартове значення."""
        answered = self._answered()
        if not answered:
            return self._prior
        return sum(self._theta[s] for s in answered) / len(answered)

    def weakest_subtopic(self, rng: random.Random | None = None) -> str:
        """Підтема з найменшою θ̂ серед тих, де були відповіді (при рівних — перша за порядком config).

        Якщо відповідей ще не було, слабку тему не видно — беремо випадкову через rng
        (або першу, якщо rng не передано), щоб результат був відтворюваним.
        """
        answered = self._answered()
        if not answered:
            return rng.choice(self._order) if rng is not None else self._order[0]
        return min(answered, key=lambda s: (self._theta[s], self._order.index(s)))

    def weakest_blended(self, rng: random.Random | None = None, exclude: str | None = None) -> str:
        """Найслабша підтема за ЗМІШАНОЮ θ̂ серед УСІХ підтем (без відповідей — загальна θ̂).

        На відміну від weakest_subtopic, не «залипає» на першій підтемі з відповіддю.
        exclude — підтема, яку не можна брати (щоб не повторювати одну тему без кінця);
        рівні значення розв'язує rng (або порядок config, якщо rng немає).
        """
        options = [s for s in self._order if s != exclude]
        lowest = min(self.theta_blended(s) for s in options)
        ties = [s for s in options if math.isclose(self.theta_blended(s), lowest)]
        return rng.choice(ties) if rng is not None else ties[0]

    def snapshot(self) -> dict:
        """Копія всіх θ̂ (для бази даних, підсумку сесії й метрик)."""
        return dict(self._theta)
