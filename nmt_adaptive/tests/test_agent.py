"""Тести агента Q-навчання: формула оновлення, маска дій, ε-спад, заморожений режим, JSON, навчання на іграшці."""
import json

import numpy as np
import pytest

import config
from core.agent import QAgent
from core.env import encode_state

ALPHA, GAMMA = config.ALPHA, config.GAMMA


def _agent(seed: int = 0, **kw: object) -> QAgent:
    """Агент з фіксованим seed і коротким спадом ε (10 сесій) для тестів."""
    kw.setdefault("decay_episodes", 10)
    kw.setdefault("alpha_mode", "const")                             # ручні формули — зі сталим α
    return QAgent(np.random.default_rng(seed), **kw)


# ----------------------------------------------------------------------------- формула оновлення
def test_update_terminal_by_hand() -> None:
    """Кінець сесії: ціль = r (без майбутнього). Q = 0, r = 1 → Q = α; TD-помилка = 1."""
    agent = _agent()
    td = agent.update(5, 2, 1.0, 7, done=True)
    assert td == pytest.approx(1.0)
    assert agent.q[5, 2] == pytest.approx(ALPHA)
    assert np.count_nonzero(agent.q) == 1


def test_update_non_terminal_by_hand() -> None:
    """Не кінець: ціль = r + γ·max Q(s′). max Q(s′) = 2 → Q = α·(1 + γ·2)."""
    agent = _agent()
    agent.q[7] = [0.5, 2.0, -1.0, 0.0, 0.3]
    agent.update(5, 2, 1.0, 7, done=False)
    assert agent.q[5, 2] == pytest.approx(ALPHA * (1.0 + GAMMA * 2.0))


def test_update_uses_only_valid_next_actions() -> None:
    """max Q(s′) — лише серед дозволених у s′: недоступна перерва з великим Q не впливає."""
    agent = _agent()
    agent.q[7] = [0.5, 1.0, -1.0, 0.0, 10.0]                         # 4 = short_break
    agent.update(5, 0, 0.0, 7, done=False, valid_next=[0, 1, 2, 3])
    assert agent.q[5, 0] == pytest.approx(ALPHA * GAMMA * 1.0)


def test_repeated_updates_converge_to_target() -> None:
    """Багато однакових оновлень → Q(s,a) прямує до цілі r (для кінця сесії)."""
    agent = _agent()
    for _ in range(300):
        agent.update(0, 0, 2.5, 1, done=True)
    assert agent.q[0, 0] == pytest.approx(2.5, abs=1e-6)


# ----------------------------------------------------------------------------- вибір дії
def test_greedy_choice_and_ties() -> None:
    """Жадібний вибір — найбільше Q; рівні значення → менший номер дії; маска враховується."""
    agent = _agent()
    agent.q[3] = [1.0, 4.0, 4.0, 0.0, 9.0]
    assert agent.greedy_action(3) == 4
    assert agent.greedy_action(3, valid=[0, 1, 2, 3]) == 1
    assert agent.greedy_action(10) == 0


def test_choose_action_respects_mask_while_exploring() -> None:
    """Навіть під час випадкового дослідження (ε = 1) агент обирає лише дозволені дії."""
    agent = _agent()
    picks = {agent.choose_action(0, valid=[1, 3]) for _ in range(200)}
    assert picks == {1, 3}


def test_epsilon_decays_linearly_then_stays() -> None:
    """ε: 1,0 на старті → лінійно → EPS_END після decay_episodes → не нижче."""
    agent = _agent(decay_episodes=10)
    assert agent.epsilon() == pytest.approx(config.EPS_START)
    for _ in range(5):
        agent.end_episode()
    assert agent.epsilon() == pytest.approx(config.EPS_START - (config.EPS_START - config.EPS_END) * 0.5)
    for _ in range(50):
        agent.end_episode()
    assert agent.epsilon() == config.EPS_END                          # рівно, без похибки округлення
    assert agent.episodes == 55


def test_default_decay_follows_config() -> None:
    """За замовчуванням ε спадає протягом EPS_DECAY_SHARE × TRAIN_EPISODES сесій."""
    agent = QAgent(np.random.default_rng(0))
    assert agent.decay_episodes == int(config.EPS_DECAY_SHARE * config.TRAIN_EPISODES)
    assert agent.alpha_mode == config.ALPHA_MODE


def test_same_seed_same_choices() -> None:
    """Той самий seed → та сама послідовність дій (відтворюваність)."""
    a, b = _agent(seed=3), _agent(seed=3)
    seq_a = [a.choose_action(s % 135) for s in range(100)]
    seq_b = [b.choose_action(s % 135) for s in range(100)]
    assert seq_a == seq_b and len(set(seq_a)) == 5


# ----------------------------------------------------------------------------- заморожений режим і JSON
def test_freeze_stops_learning_and_exploring() -> None:
    """Заморожений агент: ε = 0, update нічого не змінює й повертає 0, вибір завжди жадібний."""
    agent = _agent()
    agent.q[2] = [0.0, 0.0, 3.0, 0.0, 0.0]
    agent.visits[2, 2] = 1                                           # стан відомий агентові
    agent.freeze()
    before = agent.q.copy()
    assert agent.update(2, 0, 100.0, 3, done=True) == 0.0
    assert np.array_equal(agent.q, before)
    assert agent.epsilon() == 0.0
    assert {agent.choose_action(2) for _ in range(50)} == {2}


def test_save_and_load_roundtrip(tmp_path) -> None:
    """Збереження й завантаження: таблиця «до біта», параметри, назви дій; завантажений — заморожений."""
    agent = _agent(seed=1)
    agent.q[:] = np.random.default_rng(9).normal(size=agent.q.shape)
    for _ in range(4):
        agent.end_episode()
    path = agent.save(tmp_path / "policy.json")
    loaded = QAgent.load(path, np.random.default_rng(0))
    assert np.array_equal(loaded.q, agent.q)
    assert loaded.frozen and loaded.episodes == 4
    assert loaded.params()["alpha"] == agent.params()["alpha"]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["actions"] == list(config.ACTIONS) and data["n_states"] == config.N_STATES


def _good_file(tmp_path) -> dict:
    """Вміст правильного файлу політики (як словник), щоб псувати його в тестах."""
    path = _agent().save(tmp_path / "good.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("spoil", ["other_actions", "short_row", "nan", "no_params", "list", "episodes_text"])
def test_load_rejects_bad_files(tmp_path, spoil: str) -> None:
    """Зіпсований файл → зрозуміла помилка українською (а не KeyError чи помилка numpy англійською)."""
    data = _good_file(tmp_path)
    if spoil == "other_actions":
        data["actions"] = ["a", "b"]
    elif spoil == "short_row":
        data["q"][0] = [0.0, 0.0]
    elif spoil == "nan":
        data["q"][0][0] = float("nan")
    elif spoil == "no_params":
        del data["params"]
    elif spoil == "list":
        data = [data]
    else:
        data["episodes"] = "x"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="не підходить"):
        QAgent.load(path, np.random.default_rng(0))


def test_visits_are_counted_and_saved(tmp_path) -> None:
    """Кожне оновлення рахує відвідування (s, a); вони зберігаються й завантажуються."""
    agent = _agent()
    agent.update(3, 1, 1.0, 4, done=False)
    agent.update(3, 1, 1.0, 4, done=True)
    assert agent.visits[3, 1] == 2 and agent.known(3) and not agent.known(4)
    loaded = QAgent.load(agent.save(tmp_path / "p.json"), np.random.default_rng(0))
    assert np.array_equal(loaded.visits, agent.visits)


def test_frozen_agent_uses_fallback_in_unknown_states() -> None:
    """Невідомий стан у замороженого агента: втома 2 → перерва (якщо дозволена), інакше «те саме»; прапорець ставиться."""
    agent = _agent()
    agent.freeze()
    tired = encode_state(4, 2, 0, 2)
    calm = encode_state(0, 1, 2, 0)
    break_, same = config.ACTIONS.index("short_break"), config.ACTIONS.index("same")
    assert agent.choose_action(tired) == break_ and agent.last_was_fallback
    assert agent.choose_action(tired, valid=[0, 1, 2, 3]) == same
    assert agent.choose_action(calm) == same
    agent.visits[calm, 2] = 1
    agent.q[calm, 2] = 1.0
    assert agent.choose_action(calm) == 2 and not agent.last_was_fallback


def test_frozen_agent_does_not_consume_random_numbers() -> None:
    """Заморожений агент не бере чисел із генератора — спільна випадковість застосунку не зсувається."""
    rng = np.random.default_rng(0)
    agent = QAgent(rng)
    agent.freeze()
    for state in range(50):
        agent.choose_action(state)
    assert rng.random() == np.random.default_rng(0).random()


def test_params_and_name() -> None:
    """name і params() — як у контракті агента тренажера (для журналів і порівнянь)."""
    agent = _agent()
    params = agent.params()
    assert agent.name == "q_agent"
    assert {"alpha", "gamma", "eps_start", "eps_end", "decay_episodes", "epsilon", "episodes"} <= set(params)


def test_reset_clears_table() -> None:
    """reset: Q = 0 і лічильник сесій = 0 (нове навчання з нуля)."""
    agent = _agent()
    agent.update(0, 0, 1.0, 1, done=True)
    agent.end_episode()
    agent.reset()
    assert not agent.q.any() and agent.episodes == 0


# ----------------------------------------------------------------------------- навчання на іграшці
def test_learns_best_action_on_toy_problem() -> None:
    """Іграшка: 2 стани; у стані 0 дія 2 дає +1 і веде в стан 1, інші — 0 і кінець; у стані 1 дія 4 дає +2.

    Оптимально: 0 —(2)→ 1 —(4)→ кінець, тобто Q(0,2) = 1 + γ·2. Агент має це знайти.
    """
    agent = _agent(seed=4, decay_episodes=300)
    for _ in range(600):
        state, done = 0, False
        while not done:
            action = agent.choose_action(state)
            if state == 0 and action == 2:
                reward, nxt, done = 1.0, 1, False
            elif state == 1 and action == 4:
                reward, nxt, done = 2.0, 1, True
            else:
                reward, nxt, done = 0.0, state, True
            agent.update(state, action, reward, nxt, done)
            state = nxt
        agent.end_episode()
    assert agent.greedy_action(0) == 2 and agent.greedy_action(1) == 4
    assert agent.q[0, 2] == pytest.approx(1.0 + GAMMA * 2.0, abs=0.05)


def test_visit_based_step_size() -> None:
    """Режим "visits": α = max(ALPHA_MIN, 1/(1+N)^ALPHA_POWER); перше оновлення пари — повний крок (Q = ціль)."""
    agent = _agent(alpha_mode="visits")
    agent.update(0, 1, 2.0, 1, done=True)
    assert agent.q[0, 1] == pytest.approx(2.0)
    expected = (1 + 1) ** -config.ALPHA_POWER
    assert agent.step_size(0, 1) == pytest.approx(expected)
    agent.update(0, 1, 0.0, 1, done=True)
    assert agent.q[0, 1] == pytest.approx(2.0 - expected * 2.0)
    agent.visits[0, 1] = 10**6
    assert agent.step_size(0, 1) == config.ALPHA_MIN
    with pytest.raises(ValueError):
        _agent(alpha_mode="magic")


def test_alpha_mode_saved_and_loaded(tmp_path) -> None:
    """Режим кроку зберігається в JSON і відновлюється."""
    agent = _agent(alpha_mode="visits")
    loaded = QAgent.load(agent.save(tmp_path / "p.json"), np.random.default_rng(0))
    assert loaded.alpha_mode == "visits"
