"""Агент табличного Q-навчання: таблиця Q (135 станів × 5 дій), ε-жадібний вибір, оновлення, збереження.

Та сама формула, що й у тренажері (Platform for Interaction/qlearn/tabular.py):
    Q(s, a) ← Q(s, a) + α · (r + γ · max Q(s′, a′) − Q(s, a))
    на кінці сесії майбутнього немає: ціль = r.
Відмінності від тренажера:
    - стан — номер 0…134 (опис учня), а не клітинка лабіринту;
    - max Q(s′) — лише серед ДОЗВОЛЕНИХ у s′ дій (перерва може бути вичерпана);
    - уся випадковість через переданий numpy.random.Generator (немає глобального random).
Назви методів (name, reset, choose_action, params) — як у контракті qlearn.Agent; клас не успадковується,
бо той контракт прив'язаний до сітки (env.nb, greedy_path).
"""
import json
from pathlib import Path

import numpy as np

import config
from core.env import BREAK, decode_state

FORMAT = "nmt-q-v1"
SAME = config.ACTIONS.index("same")
REQUIRED_KEYS = ("format", "actions", "n_states", "params", "episodes", "q")


class QAgent:
    """Табличний Q-агент для вибору наступного завдання."""

    name = "q_agent"

    def __init__(self, rng: np.random.Generator, alpha: float = config.ALPHA, gamma: float = config.GAMMA,
                 eps_start: float = config.EPS_START, eps_end: float = config.EPS_END,
                 decay_episodes: int | None = None) -> None:
        """rng — генератор випадковості; decay_episodes — за скільки сесій ε спадає до eps_end."""
        self.rng = rng
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        default_decay = int(config.EPS_DECAY_SHARE * config.TRAIN_EPISODES)
        self.decay_episodes = max(1, decay_episodes if decay_episodes is not None else default_decay)
        self.frozen = False
        self.reset()

    def reset(self) -> None:
        """Почати навчання з нуля: Q = 0, лічильники відвідувань = 0, сесій = 0."""
        self.q = np.zeros((config.N_STATES, config.N_ACTIONS))
        self.visits = np.zeros((config.N_STATES, config.N_ACTIONS), dtype=int)
        self.episodes = 0
        self.last_was_fallback = False

    # ------------------------------------------------------------------ дослідження
    def epsilon(self) -> float:
        """Поточна частка випадкових дій: лінійно від eps_start до eps_end, далі стала; у замороженого — 0."""
        if self.frozen:
            return 0.0
        share = self.episodes / self.decay_episodes
        if share >= 1.0:
            return self.eps_end                    # рівно eps_end, без похибки округлення
        return self.eps_start - (self.eps_start - self.eps_end) * share

    def end_episode(self) -> None:
        """Кінець навчальної сесії: +1 до лічильника, тож ε трохи спадає."""
        self.episodes += 1

    # ------------------------------------------------------------------ вибір дії
    @staticmethod
    def _valid(valid: list | None) -> list:
        """Список дозволених дій (усі 5, якщо обмежень немає)."""
        return list(range(config.N_ACTIONS)) if valid is None else list(valid)

    def known(self, state: int) -> bool:
        """Чи бачив агент цей стан під час навчання (хоч одне оновлення)."""
        return bool(self.visits[state].sum() > 0)

    @staticmethod
    def fallback_action(state: int, valid: list) -> int:
        """Запасна дія для стану, якого агент ніколи не бачив: при високій втомі — перерва, інакше — «те саме».

        Навіщо: у такому стані вся строка Q = 0, і «найкраща» дія визначалась би лише правилом рівних (easier),
        а не навчанням (знахідка рев'ю етапу 6).
        """
        if decode_state(state)[1] == config.N_FATIGUE - 1 and BREAK in valid:
            return BREAK
        return SAME if SAME in valid else valid[0]

    def greedy_action(self, state: int, valid: list | None = None) -> int:
        """Найкраща дія серед дозволених; при рівних Q — менший номер (детерміновано й легко пояснити).

        Заморожений агент у невідомому стані бере запасну дію і ставить прапорець last_was_fallback
        (застосунок записує це в журнал пілоту).
        """
        options = self._valid(valid)
        self.last_was_fallback = self.frozen and not self.known(state)
        if self.last_was_fallback:
            return self.fallback_action(state, options)
        return options[int(np.argmax(self.q[state, options]))]

    def choose_action(self, state: int, valid: list | None = None) -> int:
        """ε-жадібний вибір: з імовірністю ε — випадкова дозволена дія, інакше — найкраща.

        Заморожений агент не бере випадкових чисел узагалі (не зсуває спільний генератор).
        """
        options = self._valid(valid)
        if not self.frozen and self.rng.random() < self.epsilon():
            return options[int(self.rng.integers(len(options)))]
        return self.greedy_action(state, options)

    # ------------------------------------------------------------------ навчання
    def update(self, s: int, a: int, r: float, s_next: int, done: bool, valid_next: list | None = None) -> float:
        """Одне оновлення Q-навчання. Повертає TD-помилку (ціль − Q(s,a) до оновлення).

        Заморожений агент нічого не змінює і повертає 0.
        """
        if self.frozen:
            return 0.0
        future = 0.0 if done else float(np.max(self.q[s_next, self._valid(valid_next)]))
        td_error = r + self.gamma * future - self.q[s, a]
        self.q[s, a] += self.alpha * td_error
        self.visits[s, a] += 1
        return float(td_error)

    def freeze(self) -> None:
        """Заморожена політика (для застосунку з живими учнями): ε = 0, таблиця більше не змінюється."""
        self.frozen = True

    def params(self) -> dict:
        """Параметри агента — для журналів, CSV і звітів."""
        return {"alpha": self.alpha, "gamma": self.gamma, "eps_start": self.eps_start, "eps_end": self.eps_end,
                "decay_episodes": self.decay_episodes, "epsilon": self.epsilon(), "episodes": self.episodes,
                "frozen": self.frozen}

    # ------------------------------------------------------------------ збереження
    def save(self, path: Path) -> Path:
        """Зберігає таблицю й параметри в JSON (UTF-8). Числа — з повною точністю."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"format": FORMAT, "actions": list(config.ACTIONS), "n_states": config.N_STATES,
                "params": self.params(), "episodes": self.episodes, "q": self.q.tolist(),
                "visits": self.visits.tolist()}
        path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path, rng: np.random.Generator) -> "QAgent":
        """Завантажує збережену політику — одразу ЗАМОРОЖЕНОЮ. Перевіряє формат, дії, розміри й числа."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        q, visits = _checked_tables(data, path)
        p = data["params"]
        agent = cls(rng, p["alpha"], p["gamma"], p["eps_start"], p["eps_end"], p["decay_episodes"])
        agent.q, agent.visits, agent.episodes = q, visits, data["episodes"]
        agent.freeze()
        return agent


def _checked_tables(data: object, path: Path) -> tuple:
    """Перевіряє вміст файлу політики; повертає (q, visits) або кидає ValueError українською."""
    problem = f"Файл політики {path} не підходить: очікується формат {FORMAT}, дії {list(config.ACTIONS)}, " \
              f"таблиця {config.N_STATES}×{config.N_ACTIONS} зі скінченних чисел."
    shape = (config.N_STATES, config.N_ACTIONS)
    if not isinstance(data, dict) or any(k not in data for k in REQUIRED_KEYS):
        raise ValueError(problem)
    if data["format"] != FORMAT or data["actions"] != list(config.ACTIONS) or not isinstance(data["episodes"], int):
        raise ValueError(problem)
    try:
        q = np.array(data["q"], dtype=float)
        visits = np.array(data.get("visits", np.ones(shape)), dtype=int)
    except (ValueError, TypeError) as error:
        raise ValueError(problem) from error
    if q.shape != shape or visits.shape != shape or not np.isfinite(q).all():
        raise ValueError(problem)
    return q, visits
