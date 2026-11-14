"""Запис навчального запуску, метрики та збереження у JSON (схема v1) — для обох агентів.

Як користуватися:
    random.seed(seed)                      # seed задає той, хто запускає (runner, GUI)
    agent = make_agent("dqn", env)
    rec = RunRecorder(env, agent, seed=seed)
    result = rec.run(1000)                 # або rec.on_episode(agent.run_episode()) у своєму циклі
    save_run(result, RESULTS_DIR)

Оцінка жадібного шляху не використовує випадкових чисел, тому запис НЕ змінює навчання
(перевіряє tests/test_recording.py::test_recording_does_not_change_learning).

Головна метрика оптимальності — дисконтована винагорода шляху відносно V* (INS-008);
вартість шляху відносно найдешевшого — довідкова: з γ < 1 це різні речі.
"""
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
REQUIRED_FIELDS = ("schema_version", "run_id", "created", "agent", "seed", "gamma", "params", "env", "settings",
                   "metrics", "greedy_eval", "summary", "greedy_path", "versions", "note")
OPTIMAL_TOL = 1e-6


def convergence_from_evals(evals, hold):
    """Перша оцінка, з якої жадібний шлях оптимальний і лишається таким ≥ hold епізодів; None — не зійшовся."""
    start = None
    for e in evals:
        if not e["optimal"]:
            start = None
            continue
        if start is None:
            start = e["episode"]
        if e["episode"] - start >= hold:
            return start
    return None


class RunRecorder:
    def __init__(self, env, agent, seed=None, eval_every=10, hold=100, last_n=50, note=""):
        self.env, self.agent, self.seed, self.note = env, agent, seed, note
        self.eval_every, self.hold, self.last_n = eval_every, hold, last_n
        self.gamma = agent.gamma
        self.optimal_return = env.optimal_return(self.gamma)
        self.optimal_cost = env.optimal_cost()
        self.rewards, self.steps, self.goals, self.evals = [], [], [], []
        self.started = time.perf_counter()
        self.created = datetime.now(timezone.utc)

    # ------------------------------------------------------------- збір
    def on_episode(self, result):
        """Записати результат епізоду (словник з agent.run_episode())."""
        self.rewards.append(round(float(result["reward"]), 4))
        self.steps.append(int(result["steps"]))
        self.goals.append(bool(result["goal"]))
        if len(self.rewards) % self.eval_every == 0:
            self.evals.append(dict(self.evaluate(), episode=len(self.rewards)))

    def run(self, episodes):
        """Навчити агента episodes епізодів із записом; повертає результат (finish())."""
        for _ in range(episodes):
            self.on_episode(self.agent.run_episode())
        return self.finish()

    def evaluate(self):
        """Оцінка поточного жадібного шляху (без випадковості)."""
        path = self.agent.greedy_path()
        reaches = path[-1] == self.env.finish_i
        ret = self.env.path_return(path, self.gamma)
        optimal = (reaches and self.optimal_return is not None
                   and abs(ret - self.optimal_return) < OPTIMAL_TOL)
        return {"reaches_finish": reaches, "path_return": round(ret, 6),
                "path_cost": self.env.path_cost(path) if reaches else None, "optimal": optimal, "path": path}

    # ---------------------------------------------------------- метрики
    def summary(self):
        n = len(self.rewards)
        last = slice(max(0, n - self.last_n), n)
        final = self.evaluate()
        v_star = self.optimal_return
        reaches = final["reaches_finish"]
        return {
            "episodes": n,
            "first_success_episode": next((k + 1 for k, g in enumerate(self.goals) if g), None),
            "convergence_episode": convergence_from_evals(self.evals, self.hold),
            "final_reaches_finish": reaches,
            "final_optimal": final["optimal"],
            "final_path_return": final["path_return"],
            "return_ratio": (final["path_return"] / v_star
                             if reaches and v_star is not None and v_star > 0 else None),
            "return_gap": (round(v_star - final["path_return"], 6) if reaches and v_star is not None else None),
            "final_path_cost": final["path_cost"],
            "cost_ratio": (final["path_cost"] / self.optimal_cost
                           if reaches and self.optimal_cost else None),
            "mean_reward_last_50": (sum(self.rewards[last]) / len(self.rewards[last]) if n else None),
            "success_rate_last_50": (sum(self.goals[last]) / len(self.goals[last]) if n else None),
            "total_env_steps": sum(self.steps),
            "duration_sec": round(time.perf_counter() - self.started, 3),
        }

    def finish(self):
        """Повний результат запуску (схема v1), готовий до save_run."""
        env, summary = self.env, self.summary()
        path = self.agent.greedy_path()
        torch = sys.modules.get("torch")
        stamp = self.created.astimezone().strftime("%Y%m%d-%H%M%S")
        seed_txt = f"seed{self.seed}" if self.seed is not None else "noseed"
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": f"{stamp}_{self.agent.name}_{env.cols}x{env.rows}_{seed_txt}",
            "created": self.created.isoformat(timespec="seconds"),
            "agent": self.agent.name,
            "seed": self.seed,
            "gamma": self.gamma,
            "params": self.agent.params(),
            "env": {"map": env.to_dict(), "fingerprint": env.fingerprint(),
                    "optimal_cost": self.optimal_cost,
                    "optimal_return": None if self.optimal_return is None else round(self.optimal_return, 6)},
            "settings": {"eval_every": self.eval_every, "hold": self.hold, "last_n": self.last_n},
            "metrics": {"reward_per_episode": list(self.rewards), "steps_per_episode": list(self.steps),
                        "reached_goal": [1 if g else 0 for g in self.goals]},
            "greedy_eval": [{k: v for k, v in e.items() if k != "path"} for e in self.evals],
            "summary": summary,
            "greedy_path": [list(env.cell_of(i)) for i in path],
            "versions": {"python": platform.python_version(),
                         "torch": getattr(torch, "__version__", None) if torch else None},
            "note": self.note,
        }


# ------------------------------------------------------------- файли
def save_run(result, folder=RESULTS_DIR):
    """Записати запуск у folder/<run_id>.json (UTF-8); якщо такий файл є — <run_id>-2.json тощо."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path, k = folder / f"{result['run_id']}.json", 2
    while path.exists():
        path, k = folder / f"{result['run_id']}-{k}.json", k + 1
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def load_run(path):
    """Прочитати й перевірити запуск. ValueError з поясненням українською, якщо файл не підходить."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"файл {path.name} пошкоджений або не є JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"файл {path.name} пошкоджений: очікувався об'єкт JSON")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"файл {path.name}: невідома версія схеми {version!r} (підтримується {SCHEMA_VERSION})")
    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"у файлі {path.name} бракує полів: {', '.join(missing)}")
    return data


def list_runs(folder=RESULTS_DIR):
    """Усі *.json у папці результатів, за іменем (тобто за часом створення)."""
    folder = Path(folder)
    return sorted(folder.glob("*.json")) if folder.is_dir() else []
