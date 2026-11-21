"""Навчання Q-агента на віртуальних учнях (СИМУЛЯЦІЯ) і збереження замороженої політики.

Запуск з папки nmt_adaptive:
    python -m sim.train                         # 30 000 сесій, seed 42 → data/policy.json + графіки в results/
    python -m sim.train --ablation no_break     # агент без дії «перерва» → data/policy_no_break.json
    python -m sim.train --episodes 2000 --no-plot

Що відбувається: у кожній сесії — новий віртуальний учень (4 типи по черзі), агент обирає дії,
отримує нагороду і оновлює таблицю Q. ε спадає від 1,0 до 0,05 за перші 80 % сесій.
Бонус за приріст знань рахується зі СПРАВЖНІХ знань віртуального учня (gain_source="true").
"""
import argparse
import random
import time
from pathlib import Path

import matplotlib
import numpy as np

import config
from core.agent import QAgent
from core.env import AdaptiveEnv, decode_state
from sim import legacy
from sim.student_sim import VirtualStudent

matplotlib.use("Agg")                       # без вікна: лише файли PNG
import matplotlib.pyplot as plt  # noqa: E402

ABLATIONS = {"none": {}, "no_fatigue": {"use_fatigue": False},
             "no_self_report": {"use_self_report": False}, "no_break": {"allow_break": False}}
NO_BREAK = [i for i, a in enumerate(config.ACTIONS) if a != "short_break"]
ACTION_UA = {"easier": "легше", "same": "те саме", "harder": "складніше",
             "review_weak_topic": "повторення", "short_break": "перерва"}
CURVE_FILE = "симуляція_крива_навчання"
POLICY_FILE = "симуляція_політика_агента"


# ----------------------------------------------------------------------------- одна сесія
def make_env(type_key: str, rng: random.Random, ablation: str = "none") -> AdaptiveEnv:
    """Новий віртуальний учень типу type_key і середовище для нього (seed беруться з rng)."""
    student = VirtualStudent(type_key, random.Random(rng.randrange(2**31)))
    return AdaptiveEnv(student, random.Random(rng.randrange(2**31)), gain_source="true", **ABLATIONS[ablation])


def run_episode(agent: QAgent, env: AdaptiveEnv) -> dict:
    """Одна сесія: агент обирає дії й (якщо не заморожений) вчиться. Повертає метрики сесії."""
    state, done, total, items, in_zpd, two_wrong = env.reset(), False, 0.0, 0, 0, 0
    while not done:
        action = agent.choose_action(state, env.valid_actions())
        next_state, reward, done, info = env.step(action)
        agent.update(state, action, reward, next_state, done, env.valid_actions())
        state, total = next_state, total + reward
        if info["item"] is not None:
            items += 1
            in_zpd += info["in_zpd"]
            two_wrong += info["reward_parts"]["two_wrong"] != 0
    return {"reward": total, "items": items, "breaks": info["breaks"], "dropped": info["dropped"],
            "zpd_share": in_zpd / max(items, 1), "true_gain": env.session_gain(), "two_wrong": two_wrong}


# ----------------------------------------------------------------------------- навчання
def train(episodes: int = config.TRAIN_EPISODES, seed: int = config.SEED, ablation: str = "none",
          progress: bool = False) -> tuple:
    """Навчає агента на episodes сесіях. Повертає (агент, історія — список метрик кожної сесії)."""
    if ablation not in ABLATIONS:
        raise ValueError(f"Невідома абляція {ablation!r}. Доступні: {', '.join(ABLATIONS)}")
    rng = random.Random(seed)
    agent = QAgent(np.random.default_rng(seed), decay_episodes=int(config.EPS_DECAY_SHARE * episodes))
    types, history = list(config.STUDENT_TYPES), []
    for episode in range(episodes):
        student_type = types[episode % len(types)]
        row = run_episode(agent, make_env(student_type, rng, ablation))
        row.update(episode=episode + 1, student_type=student_type, epsilon=agent.epsilon())
        history.append(row)
        agent.end_episode()
        if progress and (episode + 1) % max(1, episodes // 10) == 0:
            recent = history[-config.REWARD_CURVE_WINDOW:]
            print(f"  сесія {episode + 1:6d} / {episodes}   ε = {agent.epsilon():.2f}   "
                  f"нагорода (ковзне) = {sum(r['reward'] for r in recent) / len(recent):+.2f}")
    agent.freeze()
    return agent, history


def policy_path(ablation: str = "none") -> Path:
    """Куди зберегти політику: data/policy.json або data/policy_<абляція>.json."""
    return config.POLICY_PATH if ablation == "none" else config.DATA_DIR / f"policy_{ablation}.json"


# ----------------------------------------------------------------------------- результати
def curve_rows(history: list, every: int = 1) -> list:
    """Рядки CSV кривої навчання (українські назви стовпців); every — брати кожну every-ту сесію.

    Ковзне середнє рахується за ВСІМА сесіями, а потім проріджується (числа ті самі, що на графіку).
    """
    smooth = legacy.moving_average([r["reward"] for r in history], config.REWARD_CURVE_WINDOW)
    return [{"сесія": r["episode"], "тип_учня": config.STUDENT_TYPES[r["student_type"]]["name"],
             "нагорода": r["reward"], "нагорода_ковзне": s, "епсилон": r["epsilon"], "відмова": r["dropped"],
             "частка_у_зоні": r["zpd_share"], "перерв": r["breaks"], "приріст_справжньої_тета": r["true_gain"]}
            for r, s in zip(history, smooth) if r["episode"] % every == 0 or every == 1]


def policy_rows(agent: QAgent) -> list:
    """Для кожного зі 135 станів: розшифровка, найкраща дія, 5 значень Q, кількість відвідувань."""
    rows = []
    for state in range(config.N_STATES):
        ability, fatigue, accuracy, self_rep = decode_state(state)
        action = config.ACTIONS[agent.greedy_action(state)]
        no_break = config.ACTIONS[agent.greedy_action(state, NO_BREAK)]
        visits = int(agent.visits[state].sum())
        row = {"стан": state, "знання": ability, "втома": fatigue, "точність": accuracy, "самооцінка": self_rep,
               "найкраща_дія": ACTION_UA[action], "найкраща_без_перерви": ACTION_UA[no_break],
               "запасна_дія": agent.last_was_fallback, "відвідувань": visits, "мало_даних": visits < config.LOW_VISITS}
        row.update({f"Q_{ACTION_UA[a]}": float(agent.q[state, i]) for i, a in enumerate(config.ACTIONS)})
        rows.append(row)
    return rows


def save_curve(history: list, folder: Path, suffix: str = "") -> tuple:
    """Крива навчання: CSV + PNG (нагорода за сесію, ковзне середнє, ε на другій осі)."""
    rows = curve_rows(history)
    thin = curve_rows(history, config.CURVE_CSV_EVERY)
    csv_path = legacy.write_csv(Path(folder) / f"{CURVE_FILE}{suffix}.csv", thin, list(thin[0]))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = [r["сесія"] for r in rows]
    ax.plot(x, [r["нагорода"] for r in rows], color="#c9d6e8", linewidth=0.5, label="нагорода за сесію")
    ax.plot(x, [r["нагорода_ковзне"] for r in rows], color="#1f4e8c", linewidth=2,
            label=f"ковзне середнє ({config.REWARD_CURVE_WINDOW} сесій)")
    ax.set_xlabel("номер навчальної сесії")
    ax.set_ylabel("сумарна нагорода за сесію")
    ax.set_title(f"Крива навчання Q-агента (симуляція{suffix.replace('_', ', ')})")
    eps_ax = ax.twinx()
    eps_ax.plot(x, [r["епсилон"] for r in rows], color="#d9822b", linestyle="--", label="ε (частка випадкових дій)")
    eps_ax.set_ylabel("ε")
    lines = ax.get_legend_handles_labels()[0] + eps_ax.get_legend_handles_labels()[0]
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=8, framealpha=0.9)
    return csv_path, _save_png(fig, Path(folder) / f"{CURVE_FILE}{suffix}.png")


def _save_png(fig: plt.Figure, path: Path) -> Path:
    """Зберегти рисунок у PNG (150 dpi) і закрити його."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_policy(agent: QAgent, folder: Path, suffix: str = "") -> tuple:
    """Політика: CSV усіх станів + PNG «знання × втома → найкраща дія» (точність і самооцінка — середні)."""
    rows = policy_rows(agent)
    csv_path = legacy.write_csv(Path(folder) / f"{POLICY_FILE}{suffix}.csv", rows, list(rows[0]))
    grid = np.zeros((config.N_FATIGUE, config.N_ABILITY), dtype=int)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for row in rows:
        if row["точність"] == 1 and row["самооцінка"] == 1:
            action = list(ACTION_UA.values()).index(row["найкраща_дія"])
            grid[row["втома"], row["знання"]] = action
            mark = "*" if row["запасна_дія"] else ("?" if row["мало_даних"] else "")
            text = f"{row['найкраща_дія']}{mark}\n(інакше: {row['найкраща_без_перерви']})"
            ax.text(row["знання"], row["втома"], text, ha="center", va="center", fontsize=7)
    ax.imshow(grid, cmap="Pastel1", vmin=0, vmax=8, aspect="auto")
    ax.set_xticks(range(config.N_ABILITY), ["дуже низькі", "низькі", "середні", "високі", "дуже високі"])
    ax.set_yticks(range(config.N_FATIGUE), ["бадьорий", "трохи втомлений", "дуже втомлений"])
    ax.set_xlabel("рівень знань (оцінка θ̂)")
    ax.set_title(f"Найкраща дія агента (симуляція{suffix.replace('_', ', ')}); точність і самооцінка — середні\n"
                 f"«інакше» — найкраща дія, коли перерва недоступна; * — стан не траплявся; ? — менше {config.LOW_VISITS} разів",
                 fontsize=9)
    return csv_path, _save_png(fig, Path(folder) / f"{POLICY_FILE}{suffix}.png")


# ----------------------------------------------------------------------------- командний рядок
def parse_args(argv: list | None = None) -> argparse.Namespace:
    """Параметри командного рядка."""
    parser = argparse.ArgumentParser(description="Навчання Q-агента на віртуальних учнях (симуляція)")
    parser.add_argument("--episodes", type=int, default=config.TRAIN_EPISODES)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="none")
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args(argv)


def summary(history: list) -> str:
    """Короткий підсумок: нагорода на початку й наприкінці навчання, відмови."""
    window = min(config.REWARD_CURVE_WINDOW, len(history))
    first, last = history[:window], history[-window:]
    mean = lambda rows, key: sum(float(r[key]) for r in rows) / len(rows)  # noqa: E731
    return (f"нагорода: перші {window} сесій {mean(first, 'reward'):+.2f} → останні {window} {mean(last, 'reward'):+.2f}; "
            f"відмови {mean(first, 'dropped'):.2f} → {mean(last, 'dropped'):.2f}")


def main(argv: list | None = None) -> None:
    """Навчити агента, зберегти політику й графіки, показати підсумок."""
    args = parse_args(argv)
    print(f"Навчання Q-агента (СИМУЛЯЦІЯ): {args.episodes} сесій, seed {args.seed}, абляція: {args.ablation}")
    start = time.time()
    agent, history = train(args.episodes, args.seed, args.ablation, progress=True)
    meta = {"seed": args.seed, "ablation": args.ablation, "episodes": args.episodes, "gain_source": "true"}
    path = agent.save(args.policy or policy_path(args.ablation), meta)
    print(f"Готово за {time.time() - start:.0f} с. Політику збережено → {path}")
    print(summary(history))
    if not args.no_plot:
        suffix = "" if args.ablation == "none" else f"_{args.ablation}"
        for file in (*save_curve(history, args.out, suffix), *save_policy(agent, args.out, suffix)):
            print(f"  → {file}")
    unknown = sum(1 for s in range(config.N_STATES) if not agent.known(s))
    print(f"Станів, яких агент не бачив: {unknown} з {config.N_STATES} (там — запасна дія)")
    print("Найкраща дія у відвіданих станах:", action_counts(agent))


def action_counts(agent: QAgent) -> str:
    """Скільки відвіданих станів має кожну дію найкращою (з перервою і без неї) — для підсумку в консолі."""
    rows = [r for r in policy_rows(agent) if r["відвідувань"] > 0]
    count = lambda key: {a: sum(r[key] == a for r in rows) for a in ACTION_UA.values()}  # noqa: E731
    best, no_break = count("найкраща_дія"), count("найкраща_без_перерви")
    return "; ".join(f"{a} {best[a]}" for a in best if best[a]) + \
        " | без перерви: " + "; ".join(f"{a} {no_break[a]}" for a in no_break if no_break[a])


if __name__ == "__main__":
    main()
