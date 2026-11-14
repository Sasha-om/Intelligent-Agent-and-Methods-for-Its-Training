"""Пакетний запуск експериментів без GUI (етап 5, FR-C3).

Запуск — з папки "Platform for Interaction" інтерпретатором з PyTorch:
    & "..\\Pytorch_test\\.venv\\Scripts\\python.exe" -m qlearn.runner --agent both --episodes 600 --seeds 0 1 2 3 4 --map rough

Параметри:
    --agent tabular|dqn|both     який агент (типово обидва)
    --episodes N                 епізодів на один запуск (типово 600)
    --seeds 0 1 2 …              seed-и (типово 0…4); кожна пара «агент × seed» — окремий запуск і файл
    --map default|wall_gap|rough|<файл.json>   карта (вбудована, експорт тренажера або файл запуску)
    --gamma, --epsilon           спільні для обох агентів (типово 0.9 і 0.12)
    --alpha                      швидкість навчання таблиці (типово 0.35)
    --set ключ=значення          параметри DQN (lr, hidden, batch, buffer, warmup, target_sync, features, …)
    --out ПАПКА                  куди писати (типово results/)
    --note ТЕКСТ                 підпис, що потрапить у кожен файл

Результат: файл JSON на кожен запуск + comparison-<час>.csv (рядок на запуск) і summary-<час>.csv (середнє ± σ по seed).
"""
import argparse
import random
import sys
from datetime import datetime

from .comparison import ROW_FIELDS, aggregate, aggregate_fields, compare, write_csv
from .environment import GridEnv
from .factory import dqn_available, make_agent
from .maps import BUILTIN_MAPS, load_map
from .recording import RESULTS_DIR, RunRecorder, save_run


def _value(text):
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _key_value(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError(f"очікувалось ключ=значення, отримано {text!r}")
    key, value = text.split("=", 1)
    return key.strip(), _value(value.strip())


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m qlearn.runner", description="Пакетні експерименти: таблиця vs DQN.")
    p.add_argument("--agent", choices=["tabular", "dqn", "both"], default="both")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--map", default="default", help=f"{', '.join(BUILTIN_MAPS)} або шлях до JSON")
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--epsilon", type=float, default=0.12)
    p.add_argument("--alpha", type=float, default=0.35)
    p.add_argument("--set", type=_key_value, action="append", default=[], metavar="КЛЮЧ=ЗНАЧЕННЯ")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--hold", type=int, default=100)
    p.add_argument("--out", default=None)
    p.add_argument("--note", default="")
    args = p.parse_args(argv)
    args.dqn_params = dict(args.set)
    return args


def run_experiment(mode, env_dict, seed, episodes, alpha=0.35, gamma=0.9, epsilon=0.12, dqn_params=None,
                   eval_every=10, hold=100, note=""):
    """Один відтворюваний запуск: seed фіксується ДО створення агента (ваги мережі теж від нього залежать)."""
    random.seed(seed)
    env = GridEnv(env_dict)
    params = {"gamma": gamma, "epsilon": epsilon}
    params.update({"alpha": alpha} if mode == "tabular" else dict(dqn_params or {}))
    agent = make_agent(mode, env, **params)
    recorder = RunRecorder(env, agent, seed=seed, eval_every=eval_every, hold=hold, note=note)
    return recorder.run(episodes)


def _fmt(v, digits=2):
    if v is None:
        return "—"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def print_summary(groups):
    print("\nПідсумок (середнє ± σ по seed):")
    head = f"{'агент':8} {'карта':6} {'γ':5} {'запусків':8} {'зійшлося':9} {'епізод збіжності':18} {'return_ratio':13} {'cost_ratio':11} {'кроків':10} {'час, с':7}"
    print(head)
    print("-" * len(head))
    for g in groups:
        conv = (f"{_fmt(g['convergence_episode_mean'], 0)} ± {_fmt(g['convergence_episode_std'], 0)}"
                if g["convergence_episode_mean"] is not None else "—")
        print(f"{g['agent']:8} {g['map']:6} {g['gamma']:<5} {g['n_runs']:<8} {g['converged']:9} {conv:18} "
              f"{_fmt(g['return_ratio_mean'], 3):13} {_fmt(g['cost_ratio_mean'], 3):11} "
              f"{_fmt(g['total_env_steps_mean'], 0):10} {_fmt(g['duration_sec_mean'], 1):7}")


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        env_dict = load_map(args.map)
    except ValueError as exc:
        print(f"Помилка: {exc}")
        return 2
    modes = ["tabular", "dqn"] if args.agent == "both" else [args.agent]
    if "dqn" in modes and not dqn_available():
        print("Помилка: для DQN потрібен PyTorch — запустіть через Pytorch_test\\.venv\\Scripts\\python.exe")
        return 2
    out = RESULTS_DIR if args.out is None else args.out
    runs = []
    for mode in modes:
        for seed in args.seeds:
            result = run_experiment(mode, env_dict, seed, args.episodes, alpha=args.alpha, gamma=args.gamma,
                                    epsilon=args.epsilon, dqn_params=args.dqn_params,
                                    eval_every=args.eval_every, hold=args.hold, note=args.note)
            path = save_run(result, out)
            s = result["summary"]
            print(f"{mode:8} seed {seed}: {args.episodes} епізодів, збіжність — {_fmt(s['convergence_episode'])}, "
                  f"return_ratio — {_fmt(s['return_ratio'], 3)}, {s['duration_sec']:.1f} с → {path.name}")
            runs.append(result)
    comparison = compare(runs)
    groups = aggregate(runs)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_runs = write_csv(f"{out}/comparison-{stamp}.csv", comparison["rows"], ROW_FIELDS)
    csv_groups = write_csv(f"{out}/summary-{stamp}.csv", groups, aggregate_fields(groups))
    for w in comparison["warnings"]:
        print(f"Увага: {w}")
    print_summary(groups)
    print(f"\nCSV: {csv_runs}\n     {csv_groups}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
