"""Спільні допоміжні функції для тестів і скриптів у папці tests/.

- load_trainer(): завантажує q_learning_trainer.py як модуль (без запуску вікна).
  Якщо tkinter недоступний (напр. хмарний sandbox), підставляє заглушку — QModel від GUI не залежить.
- MAPS: набір тестових карт (словники у форматі DEFAULT_ENV тренажера).
- path_cost(), optimal_cost(), q_hash(): обчислення для перевірок.
"""
import hashlib
import heapq
import importlib.util
import math
import random
import sys
import types
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
APP_DIR = TESTS_DIR.parent
TRAINER_PATH = APP_DIR / "q_learning_trainer.py"

_trainer_module = None


def _install_tkinter_stub():
    """Підміняє tkinter заглушкою, якщо справжнього немає. Повертає True, якщо підмінив."""
    try:
        import tkinter  # noqa: F401
        return False
    except Exception:
        pass

    class Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            cls = type(name, (), {"__init__": lambda self, *a, **k: None})
            setattr(self, name, cls)
            return cls

    for name in ("tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox"):
        sys.modules[name] = Stub(name)
    return True


def load_trainer():
    """Модуль q_learning_trainer (один раз на процес)."""
    global _trainer_module
    if _trainer_module is None:
        _install_tkinter_stub()
        spec = importlib.util.spec_from_file_location("q_learning_trainer", TRAINER_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _trainer_module = module
    return _trainer_module


def _random_map(seed, cols, rows):
    """Випадкова, але відтворювана карта з досяжним фінішем (як «Випадково» в редакторі)."""
    rng = random.Random(seed)
    start, finish = (0, 0), (cols - 1, rows - 1)
    for _ in range(200):
        obstacles, rough = [], []
        for r in range(rows):
            for c in range(cols):
                if (c, r) in (start, finish):
                    continue
                x = rng.random()
                if x < 0.22:
                    obstacles.append((c, r))
                elif x < 0.28:
                    rough.append((c, r))
        env = {"cols": cols, "rows": rows, "obstacles": obstacles, "rough": rough,
               "start": start, "finish": finish}
        m = load_trainer().QModel()
        m.load_env(env)
        if m.finish_reachable():
            return env
    raise RuntimeError("не вдалося згенерувати досяжну карту")


def build_maps():
    """Тестові карти. K1 — стандартна; K2 — стіна з проходом; K3 — дорогі ділянки; R* — випадкові фіксовані."""
    maps = {
        "K1_default": {"cols": 7, "rows": 6, "obstacles": [], "rough": [], "start": (0, 0), "finish": (6, 5)},
        "K2_wall_gap": {"cols": 7, "rows": 6, "obstacles": [(3, r) for r in range(6) if r != 4], "rough": [],
                        "start": (0, 0), "finish": (6, 0)},
        "K3_rough": {"cols": 7, "rows": 6, "obstacles": [],
                     "rough": [(c, r) for c in range(1, 6) for r in range(1, 5)],
                     "start": (0, 2), "finish": (6, 2)},
        "S_3x3": {"cols": 3, "rows": 3, "obstacles": [], "rough": [], "start": (0, 0), "finish": (2, 2)},
    }
    maps["R1_10x8"] = _random_map(1, 10, 8)
    maps["R2_15x15"] = _random_map(2, 15, 15)
    return maps


UNREACHABLE = {"cols": 5, "rows": 5, "obstacles": [(2, r) for r in range(5)], "rough": [],
               "start": (0, 0), "finish": (4, 4)}


def path_cost(model, path):
    """Сумарна вартість переходів шляху; None, якщо шлях містить неіснуючий перехід."""
    total = 0
    for a, b in zip(path, path[1:]):
        edge = next((o for o in model.nb[a] if o[0] == b), None)
        if edge is None:
            return None
        total += edge[2]
    return total


def optimal_cost(model):
    """Мінімальна вартість шляху старт → фініш (алгоритм Дейкстри); None, якщо недосяжно."""
    dist = {model.start_i: 0}
    heap = [(0, model.start_i)]
    while heap:
        d, s = heapq.heappop(heap)
        if s == model.finish_i:
            return d
        if d > dist.get(s, math.inf):
            continue
        for to, _key, cost, _label in model.nb[s]:
            nd = d + cost
            if nd < dist.get(to, math.inf):
                dist[to] = nd
                heapq.heappush(heap, (nd, to))
    return None


def q_hash(q):
    """SHA-256 усієї Q-таблиці (числа округлені до 1e-9) — чутливий відбиток для регресії."""
    text = ";".join(f"{v:.9f}" for v in q)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def train(model, episodes, seed):
    """Навчання моделі без анімації з фіксованим seed (так само, як «Швидке навчання»)."""
    random.seed(seed)
    for _ in range(episodes):
        model.run_episode_silent()
    return model


class AgentView:
    """Обгортка «середовище + агент» з тим самим API, що й QModel, для helpers.snapshot/path_cost."""

    def __init__(self, env, agent):
        self.env, self.agent = env, agent

    @property
    def nb(self):
        return self.env.nb

    @property
    def q(self):
        return self.agent.q_snapshot()

    @property
    def start_i(self):
        return self.env.start_i

    @property
    def finish_i(self):
        return self.env.finish_i

    def edge_key(self, a, b):
        return self.env.edge_key(a, b)

    def greedy_path(self):
        return self.agent.greedy_path()


def snapshot(model):
    """Ключові числа для golden-знімка."""
    path = model.greedy_path()
    sum_q = sum(model.q[model.edge_key(a, b)] for a, b in zip(path, path[1:]))
    return {
        "path_len": len(path),
        "path_cost": path_cost(model, path),
        "reaches_finish": path[-1] == model.finish_i,
        "sum_q_along_path": round(sum_q, 9),
        "q_sha256": q_hash(model.q),
    }
