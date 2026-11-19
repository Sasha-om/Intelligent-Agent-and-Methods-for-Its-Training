"""Навчання порціями за часом — для «Швидкого навчання» в інтерфейсі (етап 6).

Вікно tkinter працює в одному потоці: поки йде навчання, воно не перемальовується й не реагує.
Тому навчаємо короткими порціями (~25 мс) і повертаємо керування вікну.

    stepper = EpisodeStepper(agent)
    rewards = stepper.train_for(0.025, max_episodes=10)   # винагороди епізодів, що завершились у цій порції
    stepper.last_results                                 # ті самі епізоди: {"reward", "steps", "goal"} (для запису)

Таблиця швидка: епізод — мікросекунди, тому вона йде цілими епізодами через run_episode() — точно як раніше.
DQN повільний (~1.2 мс на крок, епізод може тривати секунди) — він іде окремими кроками (quick_step),
а незавершений епізод продовжується з того самого місця в наступній порції.
"""
import time


class EpisodeStepper:
    def __init__(self, agent):
        self.agent = agent
        self.episodes_done = 0
        self.last_results = []     # епізоди останньої порції: {"reward", "steps", "goal"}
        self.reset()

    def reset(self):
        """Почати новий епізод зі старту."""
        self.state = self.agent.env.start_i
        self.steps = 0
        self.ep_reward = 0.0

    def train_for(self, seconds, max_episodes):
        """Навчати не довше seconds (з точністю до одного кроку) і не більше max_episodes епізодів."""
        agent, env = self.agent, self.agent.env
        finished = self.last_results = []
        t0 = time.perf_counter()
        while len(finished) < max_episodes and time.perf_counter() - t0 < seconds:
            if self.steps == 0 and getattr(agent, "whole_episodes", True):
                res = agent.run_episode()
                finished.append({"reward": res["reward"], "steps": res["steps"], "goal": res["goal"]})
                self.episodes_done += 1
                continue
            res = agent.quick_step(self.state)
            goal = False
            if res is None:                      # глухий кут — епізод закінчено
                done = True
            else:
                to, reward, goal = res
                self.ep_reward += reward
                self.steps += 1
                self.state = to
                done = goal or self.steps >= env.step_cap()
            if done:
                finished.append({"reward": self.ep_reward, "steps": self.steps, "goal": bool(goal)})
                self.episodes_done += 1
                self.reset()
        return [r["reward"] for r in finished]
