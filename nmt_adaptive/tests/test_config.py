"""Тести config.py: числа узгоджені між собою і відповідають плану.

Навіщо: якщо хтось змінить, наприклад, кількість відер знань, а розмір стану забуде —
тест одразу покаже помилку, ще до запуску навчання.
"""
import config


def test_state_space_has_135_states() -> None:
    """Стан = знання (5) × втома (3) × точність (3) × самооцінка (3) = 135."""
    assert config.N_ABILITY == 5
    assert config.N_FATIGUE == 3
    assert config.N_ACCURACY == 3
    assert config.N_SELF_REPORT == 3
    assert config.N_STATES == 135


def test_five_actions_in_fixed_order() -> None:
    """Порядок дій важливий: номер дії — це номер стовпця Q-таблиці."""
    assert config.ACTIONS == ("easier", "same", "harder", "review_weak_topic", "short_break")
    assert config.N_ACTIONS == 5
    assert set(config.ACTION_TEXT_UA) == set(config.ACTIONS)


def test_edges_are_sorted() -> None:
    """Межі відер мають іти за зростанням, інакше дискретизація зламається."""
    for edges in (config.ABILITY_EDGES, config.ACCURACY_EDGES):
        assert list(edges) == sorted(edges)
    assert config.FATIGUE_ITEMS_MID < config.FATIGUE_ITEMS_HIGH


def test_difficulty_levels_cover_minus2_to_2() -> None:
    """5 рівнів складності від -2 до 2 (вимога завдання)."""
    assert config.DIFFICULTY_LEVELS == (-2.0, -1.0, 0.0, 1.0, 2.0)


def test_subtopics_cover_both_topics() -> None:
    """Не менше 6 підтем, і кожна належить до «Функцій» або «Нерівностей»."""
    assert len(config.SUBTOPICS) >= 6
    groups = {group for group, _name in config.SUBTOPICS.values()}
    assert groups == set(config.TOPIC_GROUPS)


def test_zone_bounds_are_valid_probabilities() -> None:
    """Межі зони розвитку — ймовірності, нижня менша за верхню."""
    assert 0 < config.ZPD_LOW < config.ZPD_HIGH < 1
    assert 0 < config.SIM_LEARN_LOW < config.SIM_LEARN_HIGH < 1
    assert 0 <= config.GUESS_C < 1


def test_reward_signs_match_plan() -> None:
    """Зона розвитку — плюс, помилки, втома й відмова — мінус."""
    assert config.R_ZPD == 1.0
    assert config.R_TWO_WRONG == -0.5
    assert config.R_FATIGUE_NO_BREAK == -0.3
    assert config.R_DROPOUT == -3.0
    assert config.R_GAIN_COEF > 0


def test_q_learning_hyperparameters_in_range() -> None:
    """α, γ, ε — у допустимих межах; навчання 20–50 тисяч сесій (план)."""
    assert 0 < config.ALPHA <= 1
    assert 0 <= config.GAMMA < 1
    assert 0 <= config.EPS_END < config.EPS_START <= 1
    assert 20_000 <= config.TRAIN_EPISODES <= 50_000


def test_four_student_types() -> None:
    """Чотири типи учнів: сильний/слабкий × витривалий/стомлюваний."""
    types = config.STUDENT_TYPES
    assert len(types) == 4
    strong = [t for t in types.values() if t["theta_mean"] > 0]
    tiring = [t for t in types.values() if t["fatigue_rate"] > 0.05]
    assert len(strong) == 2 and len(tiring) == 2


def test_evaluation_size_matches_plan() -> None:
    """Не менше 500 учнів на тип і 5 різних seed."""
    assert config.EVAL_STUDENTS_PER_TYPE >= 500
    assert len(config.EVAL_SEEDS) == 5
    assert len(set(config.EVAL_SEEDS)) == 5


def test_student_data_lives_in_ignored_folder() -> None:
    """База учнів лежить у data/students/ — ця папка в .gitignore."""
    assert config.DB_PATH.parent == config.STUDENTS_DIR
    assert config.STUDENTS_DIR.name == "students"
    assert config.EXPORTS_DIR.name == "exports"
