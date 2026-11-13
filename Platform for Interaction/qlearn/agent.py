"""Базовий клас агента — спільний контракт для табличного Q-навчання і DQN.

Інтерфейс (його перевіряє tests/test_agent_contract.py):
    name               — назва режиму ("tabular", "dqn")
    reset()            — скинути навчання
    choose_action(i)   — ε-жадібна дія серед дозволених у стані i (елемент env.nb[i]) або None
    learn_step(i)      — один крок + оновлення; словник to/key/old_q/new_q/reward/cost/dir/goal або None
    run_episode()      — повний епізод без анімації → {"reward", "steps", "goal"}
    q_snapshot()       — плаский список Q довжини env.n_keys (для відображення)
    greedy_path()      — жадібний шлях від старту за q_snapshot()
    params()           — параметри для збереження
"""


class Agent:
    name = "base"

    def __init__(self, env):
        self.env = env

    # ---- методи, які має реалізувати кожен агент
    def reset(self):
        raise NotImplementedError

    def choose_action(self, i):
        raise NotImplementedError

    def learn_step(self, i):
        raise NotImplementedError

    def run_episode(self):
        raise NotImplementedError

    def q_snapshot(self):
        raise NotImplementedError

    def params(self):
        raise NotImplementedError

    # ---- спільні допоміжні методи (працюють через q_snapshot)
    def best_of(self, opts):
        """Дія з найбільшим Q серед opts; при рівних — перша за порядком DIRS."""
        q = self.q_snapshot()
        best = opts[0]
        bq = q[best[1]]
        for o in opts:
            v = q[o[1]]
            if v > bq:
                bq, best = v, o
        return best

    def max_q_from(self, i):
        """Найбільше Q серед дозволених дій стану i (0, якщо дій немає)."""
        opts = self.env.nb[i]
        if not opts:
            return 0.0
        q = self.q_snapshot()
        m = q[opts[0][1]]
        for o in opts:
            v = q[o[1]]
            if v > m:
                m = v
        return m

    def greedy_path(self):
        """Жадібний шлях від старту; зупиняється на фініші, у глухому куті або при першому повторі клітинки."""
        env = self.env
        path = [env.start_i]
        visited = {env.start_i}
        cur = env.start_i
        for _ in range(env.cols * env.rows):
            opts = env.nb[cur]
            if not opts:
                break
            nxt = self.best_of(opts)[0]
            if nxt in visited:
                break
            path.append(nxt)
            if nxt == env.finish_i:
                break
            visited.add(nxt)
            cur = nxt
        return path
