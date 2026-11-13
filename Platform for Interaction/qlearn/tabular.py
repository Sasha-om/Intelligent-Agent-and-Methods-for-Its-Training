"""Табличне Q-навчання. Код перенесено з QModel тренажера БЕЗ зміни поведінки
(перевіряють tests/regression_tabular_golden.py і tests/test_tabular.py).

Оновлення: Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s′,·) − Q(s,a)]; на фініші max Q(s′,·) = 0.
"""
import random

from .agent import Agent
from .environment import GOAL_BONUS


class TabularAgent(Agent):
    name = "tabular"

    def __init__(self, env, alpha=0.35, gamma=0.9, epsilon=0.12):
        super().__init__(env)
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.reset()

    def reset(self):
        """Порожня Q-таблиця під поточний розмір сітки."""
        n = self.env.n_keys
        self.q = [0.0] * n
        self.visited = bytearray(n)  # чи оновлювалось Q для цієї пари (s,a)

    def params(self):
        return {"alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon}

    def q_snapshot(self):
        return self.q

    def choose_action(self, i):
        opts = self.env.nb[i]
        if not opts:
            return None
        if random.random() < self.epsilon:
            return opts[int(random.random() * len(opts))]
        return self.best_of(opts)

    def learn_step(self, i):
        """Один крок з анімацією: вибір дії, винагорода, оновлення Q; словник для панелі обчислень."""
        action = self.choose_action(i)
        if action is None:
            return None
        to, key, cost, label = action
        is_goal = to == self.env.finish_i
        reward = -cost + (GOAL_BONUS if is_goal else 0)
        old_q = self.q[key]
        max_next = 0.0 if is_goal else self.max_q_from(to)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next - old_q)
        self.q[key] = new_q
        self.visited[key] = 1
        return {"to": to, "key": key, "old_q": old_q, "new_q": new_q, "reward": reward,
                "cost": cost, "dir": label, "goal": is_goal}

    def run_episode(self):
        """Повний епізод без анімації. Цикл розгорнуто вручну заради швидкості
        (та сама логіка, що й learn_step — це перевіряє тест еквівалентності)."""
        env = self.env
        q, nb, vis = self.q, env.nb, self.visited
        alpha, gamma, eps = self.alpha, self.gamma, self.epsilon
        fin, cap = env.finish_i, env.step_cap()
        rnd = random.random
        s, steps, ep_r = env.start_i, 0, 0.0
        while s != fin and steps < cap:
            opts = nb[s]
            if not opts:
                break
            if rnd() < eps:
                o = opts[int(rnd() * len(opts))]
            else:
                o = opts[0]
                bq = q[o[1]]
                for p in opts:
                    v = q[p[1]]
                    if v > bq:
                        bq, o = v, p
            to, key, cost = o[0], o[1], o[2]
            if to == fin:
                r = GOAL_BONUS - cost
                mx = 0.0
            else:
                r = -cost
                nopts = nb[to]
                if nopts:
                    mx = q[nopts[0][1]]
                    for p in nopts:
                        v = q[p[1]]
                        if v > mx:
                            mx = v
                else:
                    mx = 0.0
            old = q[key]
            q[key] = old + alpha * (r + gamma * mx - old)
            vis[key] = 1
            ep_r += r
            steps += 1
            s = to
        return {"reward": ep_r, "steps": steps, "goal": s == fin}
