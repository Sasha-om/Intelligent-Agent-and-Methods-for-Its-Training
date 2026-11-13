"""DQN — глибоке Q-навчання: замість таблиці Q(s,a) — нейромережа, що за ознаками стану видає 4 значення Q.

Що робить агент на кожному кроці:
  1. обирає дію ε-жадібно серед ДОЗВОЛЕНИХ (стіни й край сітки маскуються);
  2. кладе перехід (s, a, r, s′, done) у буфер досвіду;
  3. бере з буфера випадковий батч і робить крок градієнтного спуску:
         ціль  y = r + γ·(1 − done)·max_{a′ дозволені} Q_target(s′, a′)
         втрата = Huber(Q(s,a), y)
     Q_target — копія мережі, яку оновлюють лише раз на target_sync кроків (так навчання стабільніше);
  4. done = True лише на фініші; обрив епізоду за лімітом кроків — НЕ фініш.

torch імпортується тільки тут і тільки під час створення агента: без PyTorch табличний режим працює (FR-N8).
Винагороди множаться на reward_scale (типово 0.1), щоб мережа працювала з невеликими числами;
q_snapshot() ділить назад — значення Q показуються в тих самих одиницях, що й у таблиці.
"""
import random

from .agent import Agent
from .environment import GOAL_BONUS
from .features import action_mask, all_features, feature_size
from .replay import ReplayBuffer

DEFAULT_PARAMS = {
    "hidden": 64,          # нейронів у кожному з двох прихованих шарів
    "lr": 1e-3,            # швидкість навчання оптимізатора Adam
    "batch": 32,           # переходів у батчі
    "buffer": 5000,        # місткість буфера досвіду
    "warmup": 64,          # з якої кількості переходів у буфері починати навчання
    "target_sync": 100,    # раз на скільки кроків навчання копіювати мережу в цільову
    "train_every": 1,      # навчатися на кожному N-му кроці середовища
    "clip_grad": 10.0,     # обмеження норми градієнта
    "reward_scale": 0.1,   # множник винагороди всередині мережі
    "features": "compact", # вхід мережі: "compact" (типовий, рішення Р3) або "onehot" (до 30×30)
}

TORCH_HINT = ("Режим «Нейромережа» потребує PyTorch, але його не знайдено. "
              "Запустіть тренажер інтерпретатором Pytorch_test\\.venv\\Scripts\\python.exe (там PyTorch встановлено).")


class DQNUnavailable(RuntimeError):
    """PyTorch не встановлено — DQN недоступний."""


def torch_available():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _torch():
    try:
        import torch
        return torch
    except Exception as exc:
        raise DQNUnavailable(TORCH_HINT) from exc


def masked_argmax(q_row, mask):
    """Номер дії з найбільшим Q серед дозволених (mask — булевий тензор з 4 значень); None, якщо дозволених немає."""
    if not bool(mask.any()):
        return None
    return int(q_row.masked_fill(~mask, float("-inf")).argmax().item())


def build_network(n_in, hidden):
    """Повнозв'язна мережа: вхід → hidden → hidden → 4 значення Q (за порядком DIRS)."""
    nn = _torch().nn
    return nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 4))


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, env, gamma=0.9, epsilon=0.12, **params):
        super().__init__(env)
        self.torch = _torch()
        unknown = set(params) - set(DEFAULT_PARAMS)
        if unknown:
            raise ValueError(f"невідомі параметри DQN: {', '.join(sorted(unknown))}")
        self.gamma, self.epsilon = gamma, epsilon
        self.cfg = dict(DEFAULT_PARAMS, **params)
        feature_size(env, self.cfg["features"])          # перевірка назви ознак
        self.reset()

    # ------------------------------------------------------------- будова
    def reset(self):
        """Нова мережа, порожній буфер. Ваги мережі залежать від random.seed — запуски відтворювані."""
        torch, env, cfg = self.torch, self.env, self.cfg
        torch.manual_seed(random.getrandbits(32))
        self.X = torch.tensor(all_features(env, cfg["features"]), dtype=torch.float32)
        self.M = torch.tensor([action_mask(env, i) for i in range(env.cols * env.rows)], dtype=torch.bool)
        self.net = build_network(self.X.shape[1], cfg["hidden"])
        self.target = build_network(self.X.shape[1], cfg["hidden"])
        self.target.load_state_dict(self.net.state_dict())
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"], foreach=True)  # foreach — швидше на CPU
        self.buffer = ReplayBuffer(cfg["buffer"])
        self.visited = bytearray(env.n_keys)
        self.learn_steps = 0      # кроків градієнтного спуску
        self.env_steps = 0        # кроків у середовищі
        self.last_loss = None
        self._q_cache = None

    def params(self):
        return dict({"gamma": self.gamma, "epsilon": self.epsilon}, **self.cfg)

    # ---------------------------------------------------------------- Q
    def _q_state(self, i):
        """4 значення Q онлайн-мережі для стану i (у масштабі мережі)."""
        with self.torch.no_grad():
            return self.net(self.X[i:i + 1])[0]

    def q_snapshot(self):
        """Плаский список Q для всього інтерфейсу: лише дозволені пари, у одиницях винагороди."""
        if self._q_cache is None:
            env, scale = self.env, self.cfg["reward_scale"]
            with self.torch.no_grad():
                q_all = self.net(self.X).tolist()
            q = [0.0] * env.n_keys
            for i in env.free:
                row = q_all[i]
                for _to, key, _cost, _label in env.nb[i]:
                    q[key] = row[key - 4 * i] / scale
            self._q_cache = q
        return self._q_cache

    # ------------------------------------------------------------- дії
    def choose_action(self, i):
        opts = self.env.nb[i]
        if not opts:
            return None
        if random.random() < self.epsilon:
            return opts[int(random.random() * len(opts))]
        row = self._q_state(i).tolist()
        best = opts[0]
        bq = row[best[1] - 4 * i]
        for o in opts:
            v = row[o[1] - 4 * i]
            if v > bq:
                bq, best = v, o
        return best

    # -------------------------------------------------------- навчання
    def compute_targets(self, rewards, next_states, dones):
        """y = r + γ·(1 − done)·max Q_target(s′) серед дозволених дій s′ (тензори однакової довжини)."""
        torch = self.torch
        with torch.no_grad():
            q_next = self.target(self.X[next_states])
            q_next = q_next.masked_fill(~self.M[next_states], float("-inf"))
            best = q_next.max(dim=1).values
            best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))  # стан без дій
            return rewards + self.gamma * (1.0 - dones) * best

    def optimize(self, batch):
        """Один крок градієнтного спуску на батчі переходів; повертає значення втрати."""
        torch, cfg = self.torch, self.cfg
        s = torch.tensor([t[0] for t in batch], dtype=torch.long)
        a = torch.tensor([t[1] for t in batch], dtype=torch.long)
        r = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        s2 = torch.tensor([t[3] for t in batch], dtype=torch.long)
        d = torch.tensor([1.0 if t[4] else 0.0 for t in batch], dtype=torch.float32)
        q = self.net(self.X[s]).gather(1, a.unsqueeze(1)).squeeze(1)
        y = self.compute_targets(r, s2, d)
        loss = torch.nn.functional.smooth_l1_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg["clip_grad"])
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % cfg["target_sync"] == 0:
            self.target.load_state_dict(self.net.state_dict())
        self._q_cache = None
        self.last_loss = loss.item()
        return self.last_loss

    def _train_maybe(self):
        cfg = self.cfg
        if len(self.buffer) < max(cfg["warmup"], cfg["batch"]) or self.env_steps % cfg["train_every"]:
            return None
        return self.optimize(self.buffer.sample(cfg["batch"]))

    def _remember(self, s, key, reward, to, goal):
        self.buffer.push(s, key - 4 * s, reward * self.cfg["reward_scale"], to, goal)
        self.visited[key] = 1
        self.env_steps += 1

    def learn_step(self, i):
        """Один крок з повним звітом для панелі «Обчислення на поточному кроці»."""
        action = self.choose_action(i)
        if action is None:
            return None
        torch, scale = self.torch, self.cfg["reward_scale"]
        to, key, cost, label = action
        slot = key - 4 * i
        is_goal = to == self.env.finish_i
        reward = -cost + (GOAL_BONUS if is_goal else 0)
        old_q = self._q_state(i)[slot].item() / scale
        target = self.compute_targets(torch.tensor([reward * scale]), torch.tensor([to]),
                                      torch.tensor([1.0 if is_goal else 0.0]))[0].item() / scale
        self._remember(i, key, reward, to, is_goal)
        loss = self._train_maybe()
        new_q = self._q_state(i)[slot].item() / scale
        self._q_cache = None
        return {"to": to, "key": key, "old_q": old_q, "new_q": new_q, "reward": reward,
                "cost": cost, "dir": label, "goal": is_goal, "target": target, "loss": loss}

    def run_episode(self):
        """Повний епізод без анімації (без зайвих обчислень для звіту)."""
        env = self.env
        fin, cap = env.finish_i, env.step_cap()
        s, steps, ep_r = env.start_i, 0, 0.0
        while s != fin and steps < cap:
            action = self.choose_action(s)
            if action is None:
                break
            to, key, cost, _label = action
            is_goal = to == fin
            reward = -cost + (GOAL_BONUS if is_goal else 0)
            self._remember(s, key, reward, to, is_goal)
            self._train_maybe()
            ep_r += reward
            steps += 1
            s = to
        self._q_cache = None
        return {"reward": ep_r, "steps": steps, "goal": s == fin}
