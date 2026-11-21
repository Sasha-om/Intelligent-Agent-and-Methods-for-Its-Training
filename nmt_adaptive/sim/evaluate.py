"""Порівняння стратегій на віртуальних учнях (СИМУЛЯЦІЯ): довірчі інтервали, парний bootstrap, абляції, стійкість.

Запуск з папки nmt_adaptive:
    python -m sim.evaluate            # 500 учнів × 4 типи × 5 seed (кілька хвилин)
    python -m sim.evaluate --quick    # 50 учнів × 4 типи × 1 seed — швидка перевірка

Чесність порівняння: усі стратегії працюють з ТИМИ САМИМИ учнями (clone) і тими самими seed середовища.
Головна метрика — приріст СПРАВЖНІХ знань віртуального учня (не нагорода, яку ми самі придумали).
"""
import argparse
import contextlib
import copy
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import FuncFormatter
from statistics import NormalDist

import config
from core.baselines import make_policies
from core.env import BREAK, AdaptiveEnv
from sim import legacy
from sim.student_sim import clone, make_population

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = {"gain": "приріст знань", "zpd_true": "частка в зоні розвитку", "dropped": "відмови",
           "frustration": "фрустрація", "est_error": "похибка оцінки знань", "reward": "нагорода"}
NAMES_UA = {"random": "випадкова", "staircase": "драбинка (Staircase)", "cat": "CAT", "rule_6": "правило: перерва/6",
            "rule_5": "правило: перерва/5", "cat_break_6": "CAT + перерва/6",
            "staircase_break_6": "драбинка + перерва/6", "q_agent": "Q-агент", "q_no_fatigue": "Q без втоми",
            "q_no_self_report": "Q без самооцінки", "q_no_break": "Q без перерви"}
TYPES = list(config.STUDENT_TYPES)
ALL = "усі"
MAIN = ("random", "staircase", "cat", "staircase_break_6", "cat_break_6", "rule_6", "rule_5", "q_agent")
ENV_SEED_SHIFT = 7_919_000    # seed середовища ≠ seed учня (щоб їхні випадкові потоки не збігались)
BOOTSTRAP_CHUNK = 200         # bootstrap блоками по 200 повторів — менше пам'яті


# ----------------------------------------------------------------------------- одна сесія
def run_session(policy: object, student: object, env_seed: int) -> dict:
    """Одна сесія стратегії з учнем; метрики рахуються зі СПРАВЖНІХ параметрів віртуального учня."""
    env = AdaptiveEnv(student, random.Random(env_seed + ENV_SEED_SHIFT), gain_source="true",
                      **getattr(policy, "env_kwargs", {}))
    state, done, total, items, in_zone, frustration = env.reset(), False, 0.0, 0, 0, 0
    policy.reset(env)
    while not done:
        action = policy.choose_action(state, env.valid_actions())
        if action == BREAK:
            state, reward, done, info = env.step(action)
        else:
            item = env.next_item(action)
            in_zone += config.ZPD_LOW <= student.true_probability(item["subtopic"], item["difficulty"]) <= config.ZPD_HIGH
            state, reward, done, info = env.apply_answer(student.answer(item["subtopic"], item["difficulty"]))
            items += 1
            frustration += info["reward_parts"]["two_wrong"] != 0
        policy.observe(info)
        total += reward
    truth = student.true_theta()
    error = np.mean([abs(env.estimator.theta_blended(s) - truth[s]) for s in truth])
    return {"gain": env.session_gain(), "zpd_true": in_zone / max(items, 1), "dropped": float(info["dropped"]),
            "frustration": frustration / max(items, 1), "est_error": float(error), "reward": total}


def evaluate(policies: list, students: int, seeds: list, progress: bool = False) -> list:
    """Кожна стратегія × кожен тип × students учнів × кожен seed. Учні й seed середовища — спільні для всіх."""
    rows = []
    for seed in seeds:
        if progress:
            print(f"  seed {seed}: {len(policies)} стратегій × {len(TYPES)} типи × {students} учнів …", flush=True)
        for t_index, type_key in enumerate(TYPES):
            population = make_population(type_key, students, seed * 100 + t_index)
            for i, student in enumerate(population):
                env_seed = seed * 10**6 + t_index * 10**4 + i
                for policy in policies:
                    metrics = run_session(policy, clone(student, env_seed), env_seed)
                    rows.append({"seed": seed, "type": type_key, "student": i, "policy": policy.name, **metrics})
    return rows


# ----------------------------------------------------------------------------- статистика
def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int = config.BOOTSTRAP_SAMPLES,
                 level: float = config.CI_LEVEL) -> tuple:
    """Середнє і довірчий інтервал (bootstrap): тисячі разів перевибираємо значення з поверненням."""
    values = np.asarray(values, dtype=float)
    means = np.concatenate([values[rng.integers(0, len(values), size=(min(BOOTSTRAP_CHUNK, samples - start),
                                                                          len(values)))].mean(axis=1)
                            for start in range(0, samples, BOOTSTRAP_CHUNK)])
    tail = (1 - level) / 2 * 100
    return float(values.mean()), float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def _select(rows: list, policy: str, type_key: str, metric: str) -> np.ndarray:
    """Значення метрики стратегії для типу учнів (або всіх), у ФІКСОВАНОМУ порядку учнів (для парності)."""
    chosen = [r for r in rows if r["policy"] == policy and (type_key == ALL or r["type"] == type_key)]
    chosen.sort(key=lambda r: (r["seed"], r["type"], r["student"]))
    return np.array([r[metric] for r in chosen], dtype=float)


def summary_rows(rows: list, policies: list, rng: np.random.Generator) -> list:
    """Таблиця «стратегія × тип учнів»: для кожної метрики — середнє і 95 % ДІ."""
    out = []
    for name in policies:
        for type_key in TYPES + [ALL]:
            row = {"стратегія": NAMES_UA.get(name, name), "тип_учня": _type_ua(type_key)}
            for metric, title in METRICS.items():
                mean, low, high = bootstrap_ci(_select(rows, name, type_key, metric), rng)
                row.update({title: mean, f"{title}_ДІ_низ": low, f"{title}_ДІ_верх": high})
            out.append(row)
    return out


def _type_ua(type_key: str) -> str:
    """Назва типу учня українською."""
    return ALL if type_key == ALL else config.STUDENT_TYPES[type_key]["name"]


def paired_test(rows: list, a: str, b: str, metric: str, type_key: str, rng: np.random.Generator) -> dict:
    """Парний bootstrap: різниця «a мінус b» на ТИХ САМИХ учнях; значуща, якщо 95 % ДІ не містить 0."""
    diff = _select(rows, a, type_key, metric) - _select(rows, b, type_key, metric)
    mean, low, high = bootstrap_ci(diff, rng)
    return {"порівняння": f"{NAMES_UA.get(a, a)} − {NAMES_UA.get(b, b)}", "тип_учня": _type_ua(type_key),
            "метрика": METRICS[metric], "різниця": mean, "ДІ_низ": low, "ДІ_верх": high,
            "значуща": bool(low > 0 or high < 0)}


def test_rows(rows: list, rng: np.random.Generator, others: list, target: str = "q_agent") -> list:
    """Q-агент проти кожної іншої стратегії: приріст, відмови, зона — для кожного типу і разом.

    Тестів багато, тому додаємо поправку Бонферроні: «значуща_з_поправкою», якщо |z| > z для рівня 0,05/кількість.
    """
    tests = [paired_test(rows, target, other, metric, type_key, rng)
             for other in others for metric in ("gain", "dropped", "zpd_true") for type_key in TYPES + [ALL]]
    return add_bonferroni(tests)


def add_bonferroni(tests: list) -> list:
    """z = різниця / стандартна похибка (з ширини 95 % ДІ); поріг Бонферроні для m тестів."""
    z_ci = NormalDist().inv_cdf(0.5 + config.CI_LEVEL / 2)
    z_crit = NormalDist().inv_cdf(1 - (1 - config.CI_LEVEL) / 2 / max(len(tests), 1))
    for t in tests:
        se = (t["ДІ_верх"] - t["ДІ_низ"]) / (2 * z_ci)
        t["z"] = t["різниця"] / se if se > 0 else 0.0
        t["значуща_з_поправкою"] = bool(abs(t["z"]) > z_crit)
    return tests


# ----------------------------------------------------------------------------- стійкість до припущень
@contextlib.contextmanager
def simulator_override(changes: dict) -> "contextlib.AbstractContextManager":
    """Тимчасово змінює параметри симулятора в config (і повертає їх назад навіть при помилці)."""
    saved = {key: copy.deepcopy(getattr(config, key)) for key in changes if key != "fatigue_scale"}
    saved_types = copy.deepcopy(config.STUDENT_TYPES)
    try:
        for key, value in changes.items():
            if key == "fatigue_scale":
                for kind in config.STUDENT_TYPES.values():
                    kind["fatigue_rate"] *= value
            else:
                setattr(config, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(config, key, value)
        config.STUDENT_TYPES.clear()
        config.STUDENT_TYPES.update(saved_types)


def robustness_rows(policies: list, students: int, seed: int, rng: np.random.Generator) -> list:
    """Те саме порівняння за інших припущень симулятора (агента НЕ перенавчаємо)."""
    out = []
    for label, changes in config.ROBUSTNESS.items():
        with simulator_override(changes):
            rows = evaluate(policies, students, [seed])
        for policy in policies:
            gain = bootstrap_ci(_select(rows, policy.name, ALL, "gain"), rng)
            drop = bootstrap_ci(_select(rows, policy.name, ALL, "dropped"), rng)
            out.append({"припущення": label, "стратегія": NAMES_UA[policy.name], "приріст_знань": gain[0],
                        "приріст_ДІ_низ": gain[1], "приріст_ДІ_верх": gain[2], "відмови": drop[0]})
        best = max((p.name for p in policies if p.name != "q_agent"),
                   key=lambda n: _select(rows, n, ALL, "gain").mean())
        out.append({"припущення": label, "стратегія": f"тест: Q-агент − {NAMES_UA[best]}",
                    **_diff_fields(paired_test(rows, "q_agent", best, "gain", ALL, rng))})
    return out


def _diff_fields(test: dict) -> dict:
    """Поля парного тесту для таблиці стійкості."""
    return {"приріст_знань": test["різниця"], "приріст_ДІ_низ": test["ДІ_низ"], "приріст_ДІ_верх": test["ДІ_верх"],
            "відмови": None, "значуща": test["значуща"]}


# ----------------------------------------------------------------------------- мінливість навчання
def training_variability_rows(train_seeds: list, students: int, eval_seed: int, rng: np.random.Generator) -> list:
    """Навчає Q-агента з кількома seed і порівнює КОЖНУ політику з правилами на тих самих учнях.

    Навіщо: основне порівняння оцінює одну політику (seed 42); тут видно, наскільки результат залежить від навчання.
    """
    from core.baselines import QPolicy, RuleBreakPolicy
    from sim.train import train
    out = []
    for seed in train_seeds:
        agent, _history = train(episodes=config.TRAIN_EPISODES, seed=seed)
        q = QPolicy(agent, "q_agent")
        rows = evaluate([q, RuleBreakPolicy(6), RuleBreakPolicy(5)], students, [eval_seed])
        for rule in ("rule_6", "rule_5"):
            test = paired_test(rows, "q_agent", rule, "gain", ALL, rng)
            out.append({"seed_навчання": seed, "приріст_Q": float(_select(rows, "q_agent", ALL, "gain").mean()),
                        "відмови_Q": float(_select(rows, "q_agent", ALL, "dropped").mean()),
                        "порівняння": test["порівняння"], "різниця_приросту": test["різниця"],
                        "ДІ_низ": test["ДІ_низ"], "ДІ_верх": test["ДІ_верх"], "значуща": test["значуща"]})
        print(f"  seed навчання {seed}: приріст Q {out[-1]['приріст_Q']:+.4f}", flush=True)
    return out


# ----------------------------------------------------------------------------- графіки
def bar_chart(summary: list, names: list, metric: str, title: str, path: Path) -> Path:
    """Стовпчики з 95 % ДІ: стратегії × типи учнів (і «усі»)."""
    groups = [_type_ua(t) for t in TYPES + [ALL]]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        rows = [r for r in summary if r["стратегія"] == NAMES_UA[name]]
        means = [r[METRICS[metric]] for r in rows]
        errors = [[m - r[f"{METRICS[metric]}_ДІ_низ"] for m, r in zip(means, rows)],
                  [r[f"{METRICS[metric]}_ДІ_верх"] - m for m, r in zip(means, rows)]]
        xs = [g + (k - (len(names) - 1) / 2) * width for g in range(len(groups))]
        ax.bar(xs, means, width, yerr=errors, capsize=2, label=NAMES_UA[name])
    ax.set_xticks(range(len(groups)), groups)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.3f}".replace(".", ",")))
    ax.set_ylabel(METRICS[metric])
    ax.set_title(f"{title} (симуляція; 95 % ДІ)")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_chart_with_csv(summary: list, names: list, metric: str, title: str, stem: str, out: Path) -> list:
    """PNG і CSV з тими самими числами (для тексту роботи)."""
    title_ua = METRICS[metric]
    rows = [{"стратегія": r["стратегія"], "тип_учня": r["тип_учня"], title_ua: r[title_ua],
             "ДІ_низ": r[f"{title_ua}_ДІ_низ"], "ДІ_верх": r[f"{title_ua}_ДІ_верх"]}
            for r in summary if r["стратегія"] in {NAMES_UA[n] for n in names}]
    csv_path = legacy.write_csv(out / f"{stem}.csv", rows, list(rows[0]))
    return [csv_path, bar_chart(summary, names, metric, title, out / f"{stem}.png")]


# ----------------------------------------------------------------------------- командний рядок
def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Параметри командного рядка."""
    parser = argparse.ArgumentParser(description="Порівняння стратегій на віртуальних учнях (симуляція)")
    parser.add_argument("--students", type=int, default=config.EVAL_STUDENTS_PER_TYPE)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(config.EVAL_SEEDS))
    parser.add_argument("--quick", action="store_true", help="50 учнів × 1 seed — швидка перевірка")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--no-robustness", action="store_true")
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--training-seeds", type=int, nargs="*", default=[],
                        help="додатково: навчити Q-агента з цими seed і порівняти з правилами (≈ 1 хв на seed)")
    args = parser.parse_args(argv)
    if args.quick:
        args.students, args.seeds = 50, [0]
    return args


def print_table(summary: list, tests: list, names: list) -> None:
    """Таблиця на екрані: усі учні разом; потім Q-агент проти кожної бази (приріст знань)."""
    print(f"{'стратегія':24s} {'приріст знань':>24s} {'у зоні':>8s} {'відмови':>8s} {'фрустр.':>8s}")
    for row in [r for r in summary if r["тип_учня"] == ALL and r["стратегія"] in {NAMES_UA[n] for n in names}]:
        gain = METRICS["gain"]
        print(f"{row['стратегія']:24s} {row[gain]:+.4f} [{row[gain + '_ДІ_низ']:+.4f}; {row[gain + '_ДІ_верх']:+.4f}]"
              f" {row[METRICS['zpd_true']]:8.2f} {row[METRICS['dropped']]:8.3f} {row[METRICS['frustration']]:8.3f}")
    print("\nПарний bootstrap, усі учні, приріст знань:")
    for t in [t for t in tests if t["тип_учня"] == ALL and t["метрика"] == METRICS["gain"]]:
        print(f"  {t['порівняння']:42s} {t['різниця']:+.4f} [{t['ДІ_низ']:+.4f}; {t['ДІ_верх']:+.4f}]"
              f"  значуща: {'так' if t['значуща'] else 'ні'}")


def write_outputs(rows: list, policies: list, args: argparse.Namespace, rng: np.random.Generator) -> list:
    """Усі CSV і PNG основного порівняння та абляцій."""
    names = [p.name for p in policies]
    summary = summary_rows(rows, names, rng)
    tests = test_rows(rows, rng, [n for n in MAIN if n in names and n != "q_agent"])
    out, files = args.out, []
    files.append(legacy.write_csv(out / "симуляція_порівняння_стратегій.csv", summary, list(summary[0])))
    files.append(legacy.write_csv(out / "симуляція_bootstrap_тест.csv", tests, list(tests[0])))
    main = [n for n in MAIN if n in names]
    for metric, stem, title in (("gain", "симуляція_приріст_знань", "Приріст справжніх знань за сесію"),
                                ("zpd_true", "симуляція_зона_розвитку", "Частка завдань у зоні розвитку"),
                                ("dropped", "симуляція_відмови", "Частка учнів, що кинули сесію")):
        files += save_chart_with_csv(summary, main, metric, title, stem, out)
    ablation = ["q_agent"] + [n for n in names if n.startswith("q_no")]
    if len(ablation) > 1:
        files += save_chart_with_csv(summary, ablation, "gain", "Абляції Q-агента: приріст знань",
                                     "симуляція_абляції", out)
        ab_tests = test_rows(rows, rng, ablation[1:])
        files.append(legacy.write_csv(out / "симуляція_абляції_тест.csv", ab_tests, list(ab_tests[0])))
    print_table(summary, tests, main + [n for n in ablation if n != "q_agent"])
    return files


def main(argv: list | None = None) -> None:
    """Повне порівняння: основне, абляції, стійкість; таблиця на екрані, PNG і CSV у results/."""
    args = parse_args(argv)
    rng = np.random.default_rng(config.SEED)
    policies = make_policies(np.random.default_rng(config.SEED), ablations=not args.no_ablations)
    print(f"Порівняння стратегій (СИМУЛЯЦІЯ): {args.students} учнів × {len(TYPES)} типи × seed {args.seeds}")
    start = time.time()
    rows = evaluate(policies, args.students, args.seeds, progress=True)
    files = write_outputs(rows, policies, args, rng)
    if not args.no_robustness:
        main_policies = [p for p in policies if p.name in MAIN]
        print(f"\nСтійкість до припущень симулятора ({len(config.ROBUSTNESS)} варіанти, seed {args.seeds[0]}) …", flush=True)
        robust = robustness_rows(main_policies, args.students, args.seeds[0], rng)
        fields = list(dict.fromkeys(key for row in robust for key in row))    # усі стовпці, у порядку появи
        files.append(legacy.write_csv(args.out / "симуляція_стійкість.csv", robust, fields))
    if args.training_seeds:
        print(f"\nМінливість навчання: seed {args.training_seeds} …", flush=True)
        var = training_variability_rows(args.training_seeds, args.students, args.seeds[0], rng)
        files.append(legacy.write_csv(args.out / "симуляція_мінливість_навчання.csv", var, list(var[0])))
    print(f"\nГотово за {time.time() - start:.0f} с. Файли:")
    for file in files:
        print(f"  → {file}")


if __name__ == "__main__":
    main()
