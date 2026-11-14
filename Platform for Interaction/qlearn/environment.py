"""Середовище: сітка клітинок як граф.

Стан — ціле число i = r*cols + c (рядок r, стовпець c). Дії — 4 напрями з DIRS;
номер дії («слот») — її позиція в DIRS. Ключ Q-таблиці для пари (стан, дія) — i*4 + slot.

Для кожної прохідної клітинки один раз будується список дозволених дій nb[i]:
кортежі (куди, ключ_Q, вартість, напрям). Перешкоди й край сітки туди просто не потрапляють.
"""
import hashlib
import heapq
import json
import math
from collections import deque

DIRS = [(0, -1, "ВГОРУ"), (0, 1, "ВНИЗ"), (-1, 0, "ВЛІВО"), (1, 0, "ВПРАВО")]
GOAL_BONUS = 50      # бонус за досягнення фінішу
ROUGH_COST = 3       # вартість переходу на ділянку «3×»
DEFAULT_ENV = {
    "cols": 7, "rows": 6,
    "obstacles": [],
    "rough": [],
    "start": (0, 0), "finish": (6, 5),
}


class GridEnv:
    """Сітка з перешкодами й дорогими ділянками.

    Поля cols, rows, obstacles, rough, start, finish можна змінювати напряму
    (так робить редактор у тренажері); після цього обов'язково викликати rebuild().
    """

    def __init__(self, env=None):
        self.load(env if env is not None else DEFAULT_ENV)

    # ------------------------------------------------------------ побудова
    def load(self, env):
        """Завантажити карту зі словника (формат DEFAULT_ENV) і перебудувати граф."""
        self.cols, self.rows = env["cols"], env["rows"]
        self.obstacles = {tuple(p) for p in env["obstacles"]}
        self.rough = {tuple(p) for p in env["rough"]}
        self.start, self.finish = tuple(env["start"]), tuple(env["finish"])
        self.rebuild()

    def rebuild(self):
        """Перебудова графа сусідів після зміни середовища."""
        cols, rows = self.cols, self.rows
        nb = [()] * (cols * rows)
        free = []
        for r in range(rows):
            for c in range(cols):
                if (c, r) in self.obstacles:
                    continue
                i = r * cols + c
                free.append(i)
                lst = []
                for slot, (dc, dr, label) in enumerate(DIRS):
                    nc, nr = c + dc, r + dr
                    if 0 <= nc < cols and 0 <= nr < rows and (nc, nr) not in self.obstacles:
                        cost = ROUGH_COST if (nc, nr) in self.rough else 1
                        lst.append((nr * cols + nc, i * 4 + slot, cost, label))
                nb[i] = tuple(lst)
        self.nb, self.free = nb, free
        self.start_i, self.finish_i = self.idx(self.start), self.idx(self.finish)

    # ----------------------------------------------------------- координати
    @property
    def n_keys(self):
        """Довжина Q-таблиці: 4 дії на кожну клітинку."""
        return self.cols * self.rows * 4

    def in_bounds(self, c, r):
        return 0 <= c < self.cols and 0 <= r < self.rows

    def idx(self, cell):
        return cell[1] * self.cols + cell[0]

    def cell_of(self, i):
        return i % self.cols, i // self.cols

    def step_cap(self):
        """Ліміт кроків в епізоді росте разом із розміром сітки."""
        return max(100, self.cols * self.rows * 3)

    def edge_key(self, a, b):
        """Ключ Q для переходу зі стану a в стан b (None, якщо такого переходу немає)."""
        for o in self.nb[a]:
            if o[0] == b:
                return o[1]
        return None

    def cell_kind(self, cell):
        if cell == self.start:
            return "старт"
        if cell == self.finish:
            return "фініш"
        return "ділянка 3×" if cell in self.rough else "звичайна"

    # ---------------------------------------------------------- винагорода
    @staticmethod
    def reward(cost, is_goal):
        """Винагорода за перехід: мінус вартість, плюс бонус на фініші."""
        return -cost + (GOAL_BONUS if is_goal else 0)

    # ----------------------------------------------------------- аналітика
    def finish_reachable(self):
        """Пошук у ширину від старту: чи досяжний фініш."""
        seen = {self.start_i}
        queue = deque([self.start_i])
        while queue:
            cur = queue.popleft()
            if cur == self.finish_i:
                return True
            for o in self.nb[cur]:
                if o[0] not in seen:
                    seen.add(o[0])
                    queue.append(o[0])
        return False

    def optimal_cost(self):
        """Мінімальна вартість шляху старт → фініш (алгоритм Дейкстри); None, якщо недосяжно."""
        dist = {self.start_i: 0}
        heap = [(0, self.start_i)]
        while heap:
            d, s = heapq.heappop(heap)
            if s == self.finish_i:
                return d
            if d > dist.get(s, math.inf):
                continue
            for to, _key, cost, _label in self.nb[s]:
                nd = d + cost
                if nd < dist.get(to, math.inf):
                    dist[to] = nd
                    heapq.heappush(heap, (nd, to))
        return None

    # ------------------------------ оптимум за дисконтованою винагородою (те, що максимізує агент)
    def optimal_values(self, gamma, tol=1e-12, max_iter=100000):
        """Ітерація цінностей: V*(s) = max_a [r + γ·V*(s′)], на фініші продовження немає.
        Точне розв'язання задачі, яку Q-навчання розв'язує наближено. None — для станів, звідки фініш недосяжний."""
        values = {s: 0.0 for s in self.free}
        reach = self._reaches_finish()
        for _ in range(max_iter):
            delta = 0.0
            for s in self.free:
                if s == self.finish_i or s not in reach:
                    continue
                best = max(self._step_value(o, values, gamma) for o in self.nb[s] if o[0] in reach)
                delta = max(delta, abs(best - values[s]))
                values[s] = best
            if delta < tol:
                break
        return [values[s] if s in reach and s in values else None for s in range(self.cols * self.rows)]

    def _step_value(self, o, values, gamma):
        to, _key, cost, _label = o
        if to == self.finish_i:
            return self.reward(cost, True)
        return self.reward(cost, False) + gamma * values[to]

    def _reaches_finish(self):
        """Стани, з яких можна дійти до фінішу (обхід від фінішу назад)."""
        back = {s: [] for s in self.free}
        for s in self.free:
            for o in self.nb[s]:
                back[o[0]].append(s)
        seen, queue = {self.finish_i}, deque([self.finish_i])
        while queue:
            cur = queue.popleft()
            for p in back.get(cur, ()):
                if p not in seen:
                    seen.add(p)
                    queue.append(p)
        return seen

    def optimal_return(self, gamma):
        """V*(старт) — найбільша дисконтована сума винагород; None, якщо фініш недосяжний."""
        if not self.finish_reachable():
            return None
        return self.optimal_values(gamma)[self.start_i]

    def optimal_path(self, gamma):
        """Шлях оптимальної політики від старту (за V*); при рівних — перша дія за DIRS."""
        values = self.optimal_values(gamma)
        if values[self.start_i] is None:
            return [self.start_i]
        vals = {s: v for s, v in enumerate(values) if v is not None}
        path, s = [self.start_i], self.start_i
        while s != self.finish_i and len(path) <= self.cols * self.rows:
            opts = [o for o in self.nb[s] if o[0] in vals]
            best = opts[0]
            for o in opts:
                if self._step_value(o, vals, gamma) > self._step_value(best, vals, gamma) + 1e-12:
                    best = o
            s = best[0]
            path.append(s)
        return path

    def path_cost(self, path):
        """Сума вартостей переходів уздовж шляху; None, якщо в шляху є неіснуючий перехід."""
        total = 0
        for a, b in zip(path, path[1:]):
            cost = next((o[2] for o in self.nb[a] if o[0] == b), None)
            if cost is None:
                return None
            total += cost
        return total

    def path_return(self, path, gamma):
        """Дисконтована сума винагород уздовж шляху (бонус — лише якщо шлях закінчується на фініші)."""
        total, discount = 0.0, 1.0
        for a, b in zip(path, path[1:]):
            cost = next(o[2] for o in self.nb[a] if o[0] == b)
            total += discount * self.reward(cost, b == self.finish_i)
            discount *= gamma
        return total

    # -------------------------------------------------------- серіалізація
    def to_dict(self):
        """Карта як словник (перешкоди й ділянки відсортовані — порядок не важить)."""
        def order(cells):
            return [list(p) for p in sorted(cells, key=lambda p: (p[1], p[0]))]
        return {"cols": self.cols, "rows": self.rows, "obstacles": order(self.obstacles),
                "rough": order(self.rough), "start": list(self.start), "finish": list(self.finish)}

    def fingerprint(self):
        """Короткий відбиток карти: однакові карти → однаковий рядок (для порівняння результатів)."""
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
