"""Симуляція пілоту (етап 10) — ДО уроку: що покаже вхідний і вихідний тест для трьох режимів.

Сценарій той самий, що на уроці: 6 завдань на вході → 14 тренувальних (Q-агент / CAT / фіксований) → 6 на виході.
Віртуальні учні — ті самі (clone) для всіх режимів. Питання:
  1) який приріст на тесті очікувати і наскільки він схожий на справжній приріст знань
     (і що було б, якби вихідний тест писали наступного дня — без утоми);
  2) потужність: якщо в групі n учнів, як часто різниця «Q − інший режим» виходить значущою.

Запуск:  python -m sim.pilot_sim            (≈ 1–2 хв)
         python -m sim.pilot_sim --quick    (швидка перевірка)
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np

import config
from core.stats import ancova, mean_ci
from core.agent import QAgent
from core.baselines import CATPolicy
from core.env import BREAK, AdaptiveEnv
from core.pilot_design import fixed_slots, exam_slots
from sim.evaluate import ENV_SEED_SHIFT
from sim.legacy import write_csv
from sim.student_sim import clone, make_population

TYPES = list(config.STUDENT_TYPES)
SAME = config.ACTIONS.index("same")
OTHERS = ("cat", "fixed")


def run_test(student: object) -> tuple:
    """Вхідний або вихідний тест: (бал, відповіді [(підтема, рівень, правильно?)], кинув?)."""
    answers = []
    for subtopic, level in exam_slots():
        response = student.answer(subtopic, float(level))
        answers.append((subtopic, level, response.correct))
        if response.dropped:
            return sum(ok for *_, ok in answers), answers, True
    return sum(ok for *_, ok in answers), answers, False


def choose(arm: str, env: AdaptiveEnv, state: int, agent: QAgent, cat: CATPolicy) -> tuple:
    """Крок режиму: (дія, завдання або None)."""
    if arm == "q_agent":
        action = agent.choose_action(state, env.valid_actions())
        return action, None if action == BREAK else env.next_item(action)
    if arm == "cat":
        action = cat.choose_action(state, env.valid_actions())
        return action, env.next_item(action)
    subtopic, level = fixed_slots()[env.items_done]
    return SAME, env.next_given_item({"subtopic": subtopic, "difficulty": float(level)})


def train(arm: str, student: object, pre_answers: list, seed: int, agent: QAgent) -> bool:
    """Тренування в режимі arm; True — учень кинув сесію."""
    env = AdaptiveEnv(student, random.Random(seed + ENV_SEED_SHIFT), gain_source="true",
                      session_items=config.PILOT_TRAIN_ITEMS)
    env.reset()
    state, done, info = env.warm_start(pre_answers), False, {"dropped": False}
    cat = CATPolicy()
    cat.reset(env)
    while not done:
        action, item = choose(arm, env, state, agent, cat)
        if item is None:
            state, _, done, info = env.step(BREAK)
        else:
            state, _, done, info = env.apply_answer(student.answer(item["subtopic"], item["difficulty"]))
    return bool(info["dropped"])


def mean_theta(student: object) -> float:
    """Середні справжні знання учня."""
    values = student.true_theta().values()
    return sum(values) / len(values)


def run_lesson(arm: str, student: object, seed: int, agent: QAgent) -> dict:
    """Один «урок» віртуального учня: вхід → тренування → вихід. post = None, якщо кинув."""
    start = mean_theta(student)
    pre, pre_answers, dropped = run_test(student)
    if not dropped:
        dropped = train(arm, student, pre_answers, seed, agent)
    post, post_next_day = None, None
    if not dropped:
        twin = clone(student, seed + 1)       # та сама людина, але вихідний тест — наступного дня
        twin.rest_overnight()
        post_next_day = run_test(twin)[0]
        post, _, dropped = run_test(student)
        post = None if dropped else post
    return {"режим": arm, "тип_учня": student.type_key, "вхід": pre, "вихід": post,
            "вихід_наступного_дня": post_next_day, "справжній_приріст": mean_theta(student) - start,
            "кинув": dropped}


def simulate(students: int, seed: int, agent: QAgent) -> list:
    """students учнів (порівну 4 типи); кожен — клон того самого учня в кожному режимі."""
    rows = []
    per_type = max(1, students // len(TYPES))
    for t_index, type_key in enumerate(TYPES):
        for i, student in enumerate(make_population(type_key, per_type, seed * 100 + t_index)):
            lesson_seed = seed * 10**6 + t_index * 10**4 + i
            for arm in config.PILOT_ARMS:
                rows.append({**run_lesson(arm, clone(student, lesson_seed), lesson_seed, agent), "учень": lesson_seed})
    return rows


def arm_rows(rows: list, rng: np.random.Generator) -> list:
    """Очікувані числа кожного режиму: вхід, вихід, приріст на тесті [ДІ], справжній приріст, відмови."""
    out = []
    for arm in config.PILOT_ARMS:
        mine = [r for r in rows if r["режим"] == arm]
        done = [r for r in mine if r["вихід"] is not None]
        gain = mean_ci([r["вихід"] - r["вхід"] for r in done], rng)
        true_gain = np.array([r["справжній_приріст"] for r in done])
        test_gain = np.array([r["вихід"] - r["вхід"] for r in done], dtype=float)
        rested = [r["вихід_наступного_дня"] - r["вхід"] for r in mine if r["вихід_наступного_дня"] is not None]
        out.append({"режим": config.PILOT_ARM_NAMES_UA[arm], "учнів": len(mine),
                    "вхід": np.mean([r["вхід"] for r in mine]), "вихід": np.mean([r["вихід"] for r in done]),
                    "приріст_тест": gain[0], "приріст_ДІ_низ": gain[1], "приріст_ДІ_верх": gain[2],
                    "приріст_якщо_вихід_наступного_дня": float(np.mean(rested)),
                    "справжній_приріст_θ": float(true_gain.mean()), "кинули": 1 - len(done) / len(mine),
                    "кореляція_тест_і_справжній": float(np.corrcoef(test_gain, true_gain)[0, 1])})
    return out


def one_trial(pools: dict, n: int, other: str, rng: np.random.Generator) -> tuple:
    """Один «урок» з n учнями в кожному з двох режимів: (ефект ANCOVA або None, значущий на користь Q?)."""
    picked = {arm: [pools[arm][k] for k in rng.integers(0, len(pools[arm]), n)] for arm in ("q_agent", other)}
    done = [(r, arm == "q_agent") for arm, rs in picked.items() for r in rs if r["вихід"] is not None]
    res = ancova([r["вхід"] for r, _ in done], [r["вихід"] for r, _ in done], [q for _, q in done],
                 alpha=config.PILOT_ALPHA)
    if res is None:
        return None, False
    return res["effect"], res["significant"] and res["effect"] > 0


def power_rows(rows: list, repeats: int, rng: np.random.Generator) -> list:
    """Таблиця потужності: для кожного n і кожного «Q − інший» — частка значущих «уроків» і середній ефект."""
    pools = {arm: [r for r in rows if r["режим"] == arm] for arm in config.PILOT_ARMS}
    out = []
    for other in OTHERS:
        for n in config.PILOT_SIM_GROUP_SIZES:
            trials = [one_trial(pools, n, other, rng) for _ in range(repeats)]
            effects = [e for e, _ in trials if e is not None]
            out.append({"порівняння": f"Q-агент − {config.PILOT_ARM_NAMES_UA[other]}", "учнів_у_групі": n,
                        "потужність": sum(s for _, s in trials) / repeats,
                        "середній_ефект_балів": float(np.mean(effects)) if effects else None})
    return out


def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Аргументи командного рядка."""
    parser = argparse.ArgumentParser(description="Симуляція пілоту: вхідний/вихідний тест, 3 режими, потужність.")
    parser.add_argument("--students", type=int, default=config.PILOT_SIM_STUDENTS, help="учнів на режим")
    parser.add_argument("--repeats", type=int, default=config.PILOT_SIM_REPEATS, help="повторів уроку для потужності")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="мала швидка перевірка (200 учнів, 100 повторів)")
    parser.add_argument("--out", default=str(config.RESULTS_DIR))
    return parser.parse_args(argv)


def main(argv: list | None = None) -> None:
    """Запускає симуляцію пілоту і записує results/симуляція_пілоту_*.csv/png."""
    args = parse_args(argv)
    students, repeats = config.PILOT_SIM_QUICK if args.quick else (args.students, args.repeats)
    start, rng = time.time(), np.random.default_rng(args.seed)
    agent = QAgent.load(config.POLICY_PATH, np.random.default_rng(0))
    rows = simulate(students, args.seed, agent)
    arms, power = arm_rows(rows, rng), power_rows(rows, repeats, rng)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "симуляція_пілоту_режими.csv", arms, list(arms[0]))
    write_csv(out / "симуляція_пілоту_потужність.csv", power, list(power[0]))
    save_power_chart(power, out / "симуляція_пілоту_потужність.png")
    print_tables(arms, power)
    print(f"Готово за {time.time() - start:.0f} с → {out}")


def save_power_chart(power: list, path: object) -> None:
    """Графік: потужність залежно від кількості учнів у групі (лінія на кожне порівняння)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name in dict.fromkeys(r["порівняння"] for r in power):
        mine = [r for r in power if r["порівняння"] == name]
        ax.plot([r["учнів_у_групі"] for r in mine], [100 * r["потужність"] for r in mine], marker="o", label=name)
    ax.axhline(80, color="grey", linestyle="--", linewidth=1, label="80 % — звичайна вимога")
    ax.set_xlabel("учнів у кожній групі")
    ax.set_ylabel("ймовірність побачити значущу перевагу Q, %")
    ax.set_title("Симуляція пілоту: потужність (вхід 6 → 14 → вихід 6)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def print_tables(arms: list, power: list) -> None:
    """Короткі таблиці в термінал."""
    print("\nРежим                    вхід  вихід  приріст на тесті [95% ДІ]  справжній Δθ  кинули  кор.  наст.день")
    for r in arms:
        print(f"{r['режим']:<24} {r['вхід']:.2f}  {r['вихід']:.2f}   {r['приріст_тест']:+.3f} "
              f"[{r['приріст_ДІ_низ']:+.3f}; {r['приріст_ДІ_верх']:+.3f}]   {r['справжній_приріст_θ']:+.3f}"
              f"      {100 * r['кинули']:.0f} %   {r['кореляція_тест_і_справжній']:.2f}"
              f"   {r['приріст_якщо_вихід_наступного_дня']:+.2f}")
    print("\nПотужність (частка «уроків», де Q значуще кращий; α = 0,025 з поправкою):")
    for r in power:
        print(f"  {r['порівняння']:<36} n = {r['учнів_у_групі']:>2}: {100 * r['потужність']:5.1f} %"
              f"   ефект ≈ {r['середній_ефект_балів']:+.2f} бала")


if __name__ == "__main__":
    main()
