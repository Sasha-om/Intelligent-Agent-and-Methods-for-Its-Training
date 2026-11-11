"""
Тренажер Q-навчання — Python-версія (tkinter).

Повний відповідник «Тренажер Q-навчання.html»: той самий алгоритм, інтерфейс,
редактор середовища (до 100×100 клітинок), швидке навчання, таблиця Q(s,a)
з експортом у CSV / JSON та ті самі оптимізації для великих сіток.

Запуск:  python q_learning_trainer.py
Потрібен лише стандартний Python 3 (tkinter входить до складу).
"""

import json
import random
import time
import tkinter as tk
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from tkinter import filedialog, messagebox, ttk

# ---------------------------------------------------------------- середовище
CELL, PAD = 58, 24 #Скільки пікселів потрібго для клітинки та поля навколо клітинки
MIN_COLS, MAX_COLS, MIN_ROWS, MAX_ROWS = 3, 100, 3, 100
DEFAULT_ENV = {
    "cols": 7, "rows": 6,
    "obstacles": [],
    "rough": [],
    "start": (0, 0), "finish": (6, 5),
}
DIRS = [(0, -1, "ВГОРУ"), (0, 1, "ВНИЗ"), (-1, 0, "ВЛІВО"), (1, 0, "ВПРАВО")]
ARROWS = {"ВГОРУ": "↑", "ВНИЗ": "↓", "ВЛІВО": "←", "ВПРАВО": "→"}
GOAL_BONUS = 50
WARM = 0
HISTORY_LEN = 40

# ------------------------------------------------------------------ кольори
BG = "#12131a"
PANEL = "#1a1c25"
PANEL2 = "#14151c"
BORDER = "#2a2d3a"
LINE = "#22242e"
TEXT = "#e8e9f0"
TEXT_DIM = "#9a9db0"
TEXT_FAINT = "#6b6e80"
ACCENT = "#f0a868"
ACCENT_INK = "#241a0e"
AGENT = "#4fd1c5"
POSITIVE = "#8fd19e"
NEGATIVE = "#e18a8a"
OBST_FILL = "#1b1d26"
HATCH = "#3a3e4a"
OBST_BIG = "#3a3e4c"
ROUGH_BIG = "#4a3a22"
TOOL_ACTIVE_BG = "#16302e"
STATUS_STYLE = {  # фон, рамка, текст, підпис
    "running": ("#1c2a1b", "#375c36", "#bfe8c4", "Навчання…"),
    "paused": ("#242631", "#3a3e4d", "#c7c9d6", "Пауза"),
    "done": ("#241c10", "#6b4c26", "#f0c896", "Завершено"),
}

SANS = "Segoe UI"
MONO = "Consolas"
BOX_W = 740  # максимальна ширина сітки на екрані, px


def F(px, weight="normal", mono=False):
    """Шрифт заданого розміру в пікселях (як у CSS)."""
    return (MONO if mono else SANS, -max(6, round(px)), weight)


@lru_cache(maxsize=None)
def blend(c1, c2, t):
    """Змішування кольорів — замінює прозорість (fill-opacity) з HTML."""
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def auto_wrap(label):
    """Перенос тексту за фактичною шириною мітки (як звичайний абзац у HTML)."""
    label.bind("<Configure>", lambda e: label.config(wraplength=max(200, e.width - 4)))


def signed(v, digits):
    return ("+" if v >= 0 else "") + f"{v:.{digits}f}"


# ============================================================ модель + агент
class QModel:
    """Середовище (сітка-граф) і табличний Q-агент.

    Стани зберігаються як цілі індекси i = r*cols + c, таблиця Q — плоский
    список q[i*4 + slot], де slot — номер дії в DIRS. Сусіди кожної клітинки
    обчислюються один раз (аналог кешу NB у HTML-версії).
    """

    def __init__(self):
        self.alpha, self.gamma, self.epsilon = 0.35, 0.9, 0.12
        self.load_env(DEFAULT_ENV)

    # ---------------------------------------------------------- середовище
    def load_env(self, env):
        self.cols, self.rows = env["cols"], env["rows"]
        self.obstacles = set(env["obstacles"])
        self.rough = set(env["rough"])
        self.start, self.finish = env["start"], env["finish"]
        self.rebuild()

    def in_bounds(self, c, r):
        return 0 <= c < self.cols and 0 <= r < self.rows

    def is_big(self):
        # на великих сітках вимикаємо підписи в клітинках
        return self.cols > 20 or self.rows > 20

    def idx(self, cell):
        return cell[1] * self.cols + cell[0]

    def cell_of(self, i):
        return i % self.cols, i // self.cols

    def step_cap(self):
        # ліміт кроків в епізоді росте разом із розміром сітки
        return max(100, self.cols * self.rows * 3)

    def rebuild(self):
        """Перебудова графа сусідів після зміни середовища; Q очищується."""
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
                        cost = 3 if (nc, nr) in self.rough else 1
                        lst.append((nr * cols + nc, i * 4 + slot, cost, label))
                nb[i] = tuple(lst)
        self.nb, self.free = nb, free
        self.start_i, self.finish_i = self.idx(self.start), self.idx(self.finish)
        self.reset_q()

    def reset_q(self):
        n = self.cols * self.rows * 4
        self.q = [0.0] * n
        self.visited = bytearray(n)  # чи оновлювалось Q для цієї пари (s,a)

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

    # ---------------------------------------------------------------- агент
    def max_q_from(self, i):
        opts = self.nb[i]
        if not opts:
            return 0.0
        q = self.q
        m = q[opts[0][1]]
        for o in opts:
            v = q[o[1]]
            if v > m:
                m = v
        return m

    def best_of(self, opts):
        q = self.q
        best = opts[0]
        bq = q[best[1]]
        for o in opts:
            v = q[o[1]]
            if v > bq:
                bq, best = v, o
        return best

    def choose_action(self, i):
        opts = self.nb[i]
        if not opts:
            return None
        if random.random() < self.epsilon:
            return opts[int(random.random() * len(opts))]
        return self.best_of(opts)

    def apply_step(self, i):
        action = self.choose_action(i)
        if action is None:
            return None
        to, key, cost, label = action
        is_goal = to == self.finish_i
        reward = -cost + (GOAL_BONUS if is_goal else 0)
        old_q = self.q[key]
        max_next = 0.0 if is_goal else self.max_q_from(to)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next - old_q)
        self.q[key] = new_q
        self.visited[key] = 1
        return {"to": to, "key": key, "old_q": old_q, "new_q": new_q, "reward": reward,
                "cost": cost, "dir": label, "goal": is_goal}

    def run_episode_silent(self):
        """Повний епізод без анімації. Цикл розгорнуто вручну заради швидкості."""
        q, nb, vis = self.q, self.nb, self.visited
        alpha, gamma, eps = self.alpha, self.gamma, self.epsilon
        fin, cap = self.finish_i, self.step_cap()
        rnd = random.random
        s, steps, ep_r = self.start_i, 0, 0.0
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
        return ep_r

    def greedy_path(self):
        path = [self.start_i]
        visited = {self.start_i}
        cur = self.start_i
        for _ in range(self.cols * self.rows):
            opts = self.nb[cur]
            if not opts:
                break
            nxt = self.best_of(opts)[0]
            if nxt in visited:
                break
            path.append(nxt)
            if nxt == self.finish_i:
                break
            visited.add(nxt)
            cur = nxt
        return path

    def edge_key(self, a, b):
        for o in self.nb[a]:
            if o[0] == b:
                return o[1]
        return None

    # ------------------------------------------------------- таблиця Q
    def cell_kind(self, cell):
        if cell == self.start:
            return "старт"
        if cell == self.finish:
            return "фініш"
        return "ділянка 3×" if cell in self.rough else "звичайна"

    def q_table_data(self):
        """Один рядок на кожен прохідний стан (спільне джерело для екрана, CSV, JSON)."""
        rows = []
        q, vis = self.q, self.visited
        for i in self.free:
            c, r = self.cell_of(i)
            qd = {label: None for _, _, label in DIRS}
            explored = False
            for _, key, _, label in self.nb[i]:
                qd[label] = q[key]
                if vis[key]:
                    explored = True
            best = None
            for _, _, label in DIRS:
                v = qd[label]
                if v is not None and (best is None or v > qd[best]):
                    best = label
            rows.append({"i": i, "id": f"{c},{r}", "col": c, "row": r,
                         "kind": self.cell_kind((c, r)), "q": qd,
                         "v": 0.0 if best is None else qd[best],
                         "best": best if explored else None})
        return rows


# ============================================================ віджети
class FlatButton(tk.Label):
    """Пласка кнопка в стилі HTML-версії (звичайна / основна / інструмент)."""

    def __init__(self, parent, text, command, primary=False, font=None, pady=8):
        super().__init__(parent, text=text, font=font or F(13, "bold"), pady=pady,
                         bd=0, highlightthickness=1, cursor="hand2")
        self.command, self.primary = command, primary
        self.enabled, self.active, self.hover = True, False, False
        self.bind("<Button-1>", lambda e: self.enabled and self.command())
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self._paint()

    def _set_hover(self, on):
        self.hover = on
        self._paint()

    def set_enabled(self, on):
        if self.enabled != on:
            self.enabled = on
            self._paint()

    def set_active(self, on):
        self.active = on
        self._paint()

    def _paint(self):
        if self.primary:
            bg, fg, border = ACCENT, ACCENT_INK, ACCENT
        elif self.active:
            bg, fg, border = TOOL_ACTIVE_BG, AGENT, AGENT
        else:
            bg, fg, border = PANEL, TEXT, BORDER
        if self.hover and self.enabled:
            bg = blend(bg, "#ffffff", 0.06)
        if not self.enabled:  # opacity .45 з HTML
            bg, fg, border = (blend(PANEL, x, 0.45) for x in (bg, fg, border))
        self.config(bg=bg, fg=fg, highlightbackground=border, highlightcolor=border,
                    cursor="hand2" if self.enabled else "arrow")


class Panel(tk.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, bg=PANEL, highlightbackground=BORDER, highlightcolor=BORDER,
                         highlightthickness=1, padx=20, pady=18)
        self.title = tk.Label(self, text=title.upper(), bg=PANEL, fg=TEXT_DIM,
                              font=F(12, "bold"), anchor="w", justify="left")
        self.title.pack(fill="x", pady=(0, 12))


class QTableView(tk.Frame):
    """Таблиця Q(s,a) з «липким» заголовком. Малюються лише видимі рядки,
    тож навіть 10 000 станів прокручуються без затримок."""

    COLUMNS = [("Стан (x,y)", 1.1, "w"), ("Тип", 1.15, "w"), ("↑ ВГОРУ", 1, "e"),
               ("↓ ВНИЗ", 1, "e"), ("← ВЛІВО", 1, "e"), ("→ ВПРАВО", 1, "e"),
               ("V(s) = max Q", 1.15, "e"), ("Найкраща дія", 1.35, "e")]
    RH, HH, MAX_H = 26, 34, 440

    def __init__(self, parent):
        super().__init__(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        self.head = tk.Canvas(self, height=self.HH, bg=PANEL2, highlightthickness=0)
        self.body = tk.Canvas(self, bg=PANEL, highlightthickness=0, height=200,
                              yscrollincrement=self.RH)
        self.sb = ttk.Scrollbar(self, orient="vertical", command=self._yview,
                                style="Dark.Vertical.TScrollbar")
        self.body.configure(yscrollcommand=self.sb.set)
        self.head.grid(row=0, column=0, sticky="ew")
        self.body.grid(row=1, column=0, sticky="nsew")
        self.sb.grid(row=0, column=1, rowspan=2, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rows, self.cur, self.upd = [], None, None
        self.body.bind("<Configure>", lambda e: self._layout())
        self.body._wheel = self.head._wheel = self._wheel

    def set_data(self, rows, cur, upd):
        self.rows, self.cur, self.upd = rows, cur, upd
        total = len(rows) * self.RH
        h = min(self.MAX_H - self.HH, total)
        if int(self.body.cget("height")) != h:
            self.body.config(height=h)
        self.body.config(scrollregion=(0, 0, max(1, self.body.winfo_width()), total))
        self._draw()

    def _xs(self):
        w = max(600, self.body.winfo_width())
        total = sum(c[1] for c in self.COLUMNS)
        xs, x = [], 0.0
        for _, weight, _ in self.COLUMNS:
            xs.append((x, x + w * weight / total))
            x += w * weight / total
        return xs

    def _layout(self):
        self.head.delete("all")
        for (x0, x1), (title, _, anchor) in zip(self._xs(), self.COLUMNS):
            tx = x0 + 10 if anchor == "w" else x1 - 10
            self.head.create_text(tx, self.HH / 2, text=title, anchor=anchor,
                                  fill=TEXT_DIM, font=F(12, "bold", mono=True))
        self.head.create_line(0, self.HH - 1, self.head.winfo_width(), self.HH - 1, fill=BORDER)
        self.body.config(scrollregion=(0, 0, self.body.winfo_width(), len(self.rows) * self.RH))
        self._draw()

    def _yview(self, *args):
        self.body.yview(*args)
        self._draw()

    def _wheel(self, e):
        if len(self.rows) * self.RH <= self.body.winfo_height():
            return False
        self.body.yview_scroll(int(-e.delta / 120) * 3, "units")
        self._draw()
        return True

    def _draw(self):
        b, RH = self.body, self.RH
        b.delete("row")
        if not self.rows:
            return
        xs = self._xs()
        width = xs[-1][1]
        first = max(0, int(b.canvasy(0) // RH))
        last = min(len(self.rows), first + b.winfo_height() // RH + 2)
        mono, mono_b = F(12, mono=True), F(12, "bold", mono=True)
        for n in range(first, last):
            row = self.rows[n]
            y0, y1 = n * RH, (n + 1) * RH
            ym = (y0 + y1) / 2

            def put(col, text, fill=TEXT, font=mono):
                x0, x1 = xs[col]
                anchor = self.COLUMNS[col][2]
                b.create_text(x0 + 10 if anchor == "w" else x1 - 10, ym, text=text,
                              anchor=anchor, fill=fill, font=font, tags="row")

            if row["i"] == self.cur:
                put(0, f"● ({row['col']},{row['row']})", AGENT, mono_b)
            else:
                put(0, f"({row['col']},{row['row']})")
            put(1, row["kind"], TEXT_DIM)
            for k, (_, _, label) in enumerate(DIRS):
                v = row["q"][label]
                if v is None:
                    put(2 + k, "—", TEXT_FAINT)
                    continue
                if row["best"] == label:
                    put(2 + k, f"{v:.3f}", ACCENT, mono_b)
                else:
                    put(2 + k, f"{v:.3f}")
                if self.upd == (row["i"], label):
                    x0, x1 = xs[2 + k]
                    b.create_rectangle(x0 + 1, y0 + 1, x1 - 1, y1 - 1, outline=AGENT,
                                       width=2, tags="row")
            put(6, f"{row['v']:.3f}")
            if row["best"]:
                put(7, f"{ARROWS[row['best']]} {row['best']}")
            else:
                put(7, "не досліджено", TEXT_FAINT)
            b.create_line(0, y1 - 1, width, y1 - 1, fill=LINE, tags="row")


# ============================================================ застосунок
class App:
    def __init__(self, root):
        self.root = root
        self.m = QModel()
        self.speed = 3
        self.max_episodes = 600
        self.state = self.m.start_i
        self.episode = 0
        self.steps = 0
        self.ep_reward = 0.0
        self.running = False
        self.fast = False
        self.after_id = None
        self.reward_history = []
        self.last_transition = None
        self.env_ok = True
        self.status_kind = "paused"
        self.tool = "obstacle"
        self.paint = None
        self.last_painted = None
        self.q_visible = False
        self.q_last = 0.0
        self.q_timer = None
        # розмір сітки підлаштовується під екран (як max-height: 80vh у HTML)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.box_w = max(420, min(BOX_W, sw - 560))
        self.box_h = max(360, min(self.box_w, int(sh * 0.72)))

        root.title("Тренажер Q-навчання")
        root.configure(bg=BG)
        self._styles()
        self._build_page()
        self.build_grid()

        # «прогрів»: 300 епізодів без анімації, як у HTML-версії
        for _ in range(WARM):
            self._push_history(self.m.run_episode_silent())
        self.episode = WARM
        self.state = self.m.start_i
        self.set_status("paused")
        self.render()
        # елементи, що залежать від ширини, перемальовуються після розкладки вікна
        self._relayout_id = None
        for w in (self.matrix, self.spark, self.progress):
            w.bind("<Configure>", self._on_resize)

    def _on_resize(self, _e):
        if self._relayout_id is None:
            self._relayout_id = self.root.after(30, self._relayout)

    def _relayout(self):
        self._relayout_id = None
        self.render()

    # ======================================================== побудова UI
    def _styles(self):
        st = ttk.Style(self.root)
        st.theme_use("clam")
        st.configure("Dark.Horizontal.TScale", background=ACCENT, troughcolor=LINE,
                     bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT, gripcount=0,
                     sliderlength=16, sliderthickness=14, troughrelief="flat", borderwidth=0)
        st.map("Dark.Horizontal.TScale", background=[("active", "#f6c08f")])
        st.configure("Dark.Vertical.TScrollbar", background=BORDER, troughcolor=PANEL2,
                     bordercolor=PANEL2, arrowcolor=TEXT_DIM, lightcolor=BORDER,
                     darkcolor=BORDER)
        st.map("Dark.Vertical.TScrollbar", background=[("active", HATCH)])

    def _build_page(self):
        # уся сторінка прокручується, як у браузері
        self.page_canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(self.root, orient="vertical", command=self.page_canvas.yview,
                            style="Dark.Vertical.TScrollbar")
        self.page_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.page_canvas.pack(side="left", fill="both", expand=True)
        wrap = tk.Frame(self.page_canvas, bg=BG, padx=20, pady=20)
        win = self.page_canvas.create_window(0, 0, window=wrap, anchor="nw")
        wrap.bind("<Configure>", lambda e: self.page_canvas.configure(
            scrollregion=self.page_canvas.bbox("all")))
        self.page_canvas.bind("<Configure>", lambda e: self.page_canvas.itemconfigure(
            win, width=max(e.width, wrap.winfo_reqwidth())))
        self.root.bind_all("<MouseWheel>", self._on_wheel)

        self._build_header(wrap)
        main = tk.Frame(wrap, bg=BG)
        main.pack(fill="x", pady=(18, 0))
        self._build_grid_panel(main)
        side = tk.Frame(main, bg=BG)
        side.pack(side="left", fill="both", expand=True, anchor="n", padx=(18, 0))
        self._build_editor(side)
        self._build_params(side)
        self._build_calc(side)
        self._build_path_panel(side)
        self._build_state_panel(side)
        self._build_q_panel(wrap)
        foot = tk.Label(wrap, text="Демонстраційний тренажер до науково-дослідницької роботи МАН "
                            "«Інтелектуальний агент та методи його навчання». Клітинки сітки — це "
                            "вузли графа, сусідні прохідні клітинки — його ребра; розмір сітки, "
                            "перешкоди, старт і фініш можна змінювати в редакторі середовища; шлях "
                            "ніде не заданий наперед, агент знаходить його сам методом проб і "
                            "помилок. Обчислення виконуються локально на комп'ютері — інтернет не "
                            "потрібен.",
                 bg=BG, fg=TEXT_FAINT, font=F(11.5), wraplength=600, justify="left",
                 anchor="w")
        foot.pack(fill="x", pady=(18, 0))
        auto_wrap(foot)

    def _build_header(self, parent):
        head = tk.Frame(parent, bg=BG)
        head.pack(fill="x")
        titles = tk.Frame(head, bg=BG)
        titles.pack(side="left")
        tk.Label(titles, text="Q-навчання у вільному середовищі з перешкодами", bg=BG, fg=TEXT,
                 font=F(19, "bold")).pack(anchor="w")
        tk.Label(titles, text="Табличне Q-навчання методом проб і помилок · шлях не заданий "
                              "заздалегідь — агент прокладає його сам",
                 bg=BG, fg=TEXT_DIM, font=F(12.5)).pack(anchor="w", pady=(4, 0))
        badge = tk.Frame(head, bg=PANEL, highlightbackground=BORDER, highlightthickness=1,
                         padx=14, pady=6)
        badge.pack(side="right", anchor="n")
        tk.Label(badge, text="Епізод ", bg=PANEL, fg=TEXT, font=F(12.5, mono=True)).pack(side="left")
        self.ep_now = tk.Label(badge, text="0", bg=PANEL, fg=TEXT, font=F(12.5, "bold", mono=True))
        self.ep_now.pack(side="left")
        self.ep_max = tk.Label(badge, text=" / 600", bg=PANEL, fg=TEXT, font=F(12.5, mono=True))
        self.ep_max.pack(side="left")

    def _build_grid_panel(self, parent):
        p = Panel(parent, "Середовище (сітка станів = граф суміжних клітинок)")
        p.pack(side="left", anchor="n")
        self.cv = tk.Canvas(p, bg=PANEL, highlightthickness=0, width=self.box_w, height=self.box_h,
                            cursor="crosshair")
        self.cv.pack()
        self.cv.bind("<ButtonPress-1>", self.on_press)
        self.cv.bind("<B1-Motion>", self.on_motion)
        self.cv.bind("<Motion>", self.on_motion)
        self.cv.bind("<ButtonRelease-1>", self.on_release)
        self.cv.bind("<Leave>", lambda e: self.paint is None and self.show_hover(None))

        tk.Frame(p, bg=LINE, height=1).pack(fill="x", pady=(10, 10))
        legend = tk.Frame(p, bg=PANEL)
        legend.pack(fill="x")

        def item(row, col, draw, text, mono=False, span=1):
            f = tk.Frame(legend, bg=PANEL)
            f.grid(row=row, column=col, columnspan=span, sticky="w", padx=(0, 22), pady=3)
            if draw:
                c = tk.Canvas(f, width=20, height=14, bg=PANEL, highlightthickness=0)
                draw(c)
                c.pack(side="left", padx=(0, 6))
            tk.Label(f, text=text, bg=PANEL, fg=TEXT_DIM, font=F(11.5, mono=mono)).pack(side="left")

        item(0, 0, lambda c: c.create_line(1, 7, 19, 7, fill=ACCENT, width=3),
             "поточний найкращий (жадібний) шлях")
        item(0, 1, lambda c: c.create_oval(3, 1, 15, 13, outline=AGENT, width=2, dash=(2, 2)),
             "жива позиція агента (проби й помилки)")
        item(1, 0, lambda c: c.create_rectangle(2, 0, 15, 13, fill=OBST_FILL, outline=HATCH),
             "перешкода — рух туди неможливий")
        item(1, 1, lambda c: c.create_rectangle(2, 0, 15, 13, fill=blend(PANEL, ACCENT, .45), width=0),
             "яскравість = навчене значення V(s)")
        item(2, 0, None, "вага переходу — 1, ділянки «3×» коштують дорожче", mono=True, span=2)

    def _build_editor(self, parent):
        p = Panel(parent, "Редактор середовища")
        p.pack(fill="x")
        dims = tk.Frame(p, bg=PANEL)
        dims.pack(fill="x", pady=(0, 12))
        self.cols_var = tk.StringVar(value="7")
        self.rows_var = tk.StringVar(value="6")
        for col, (label, var, hi) in enumerate([("Стовпці", self.cols_var, MAX_COLS),
                                                ("Рядки", self.rows_var, MAX_ROWS)]):
            f = tk.Frame(dims, bg=PANEL)
            f.pack(side="left", fill="x", expand=True, padx=(0, 14) if col == 0 else 0)
            tk.Label(f, text=label, bg=PANEL, fg=TEXT_DIM, font=F(12)).pack(side="left")
            sp = tk.Spinbox(f, from_=MIN_COLS, to=hi, textvariable=var, width=5, justify="right",
                            font=F(12.5, mono=True), bg=PANEL2, fg=TEXT, insertbackground=TEXT,
                            buttonbackground=PANEL2, relief="flat", highlightthickness=1,
                            highlightbackground=BORDER, highlightcolor=AGENT,
                            command=self.on_dims)
            sp.pack(side="right", ipady=4)
            sp.bind("<Return>", self.on_dims)
            sp.bind("<FocusOut>", self.on_dims)

        tools = tk.Frame(p, bg=PANEL)
        tools.pack(fill="x", pady=(0, 10))
        self.tool_btns = {}
        for k, (tool, label) in enumerate([("obstacle", "Перешкода"), ("rough", "Ділянка 3×"),
                                           ("start", "Старт"), ("finish", "Фініш"),
                                           ("erase", "Гумка")]):
            b = FlatButton(tools, label, lambda t=tool: self.select_tool(t),
                           font=F(11.5, "bold"), pady=6)
            b.grid(row=0, column=k, sticky="ew", padx=(0 if k == 0 else 6, 0))
            tools.columnconfigure(k, weight=1, uniform="tools")
            self.tool_btns[tool] = b
        self.tool_btns["obstacle"].set_active(True)

        btns = tk.Frame(p, bg=PANEL)
        btns.pack(fill="x")
        for k, (label, cmd) in enumerate([("Очистити", self.clear_env), ("Випадково", self.random_env),
                                          ("Стандартна", self.default_env)]):
            FlatButton(btns, label, cmd, font=F(12, "bold"), pady=6).grid(
                row=0, column=k, sticky="ew", padx=(0 if k == 0 else 8, 0))
            btns.columnconfigure(k, weight=1, uniform="ed")

        tk.Label(p, text="Оберіть інструмент і клацніть або проведіть мишею по клітинках сітки. "
                         "Повторне клацання перешкодою чи ділянкою 3× прибирає її. Після будь-якої "
                         "зміни середовища навчання починається заново. Розмір — до 100×100 клітинок; "
                         "для великих сіток (понад 20×20) встановіть γ ≈ 0.99 і користуйтеся "
                         "«Швидким навчанням».",
                 bg=PANEL, fg=TEXT_FAINT, font=F(11.5), wraplength=370, justify="left",
                 anchor="w").pack(fill="x", pady=(10, 0))
        self.warn = tk.Label(p, text="Фініш недосяжний зі старту: приберіть частину перешкод.",
                             bg="#2a1a1c", fg=NEGATIVE, font=F(12), wraplength=360, justify="left",
                             anchor="w", padx=10, pady=8, highlightthickness=1,
                             highlightbackground="#5c3436")

    def _slider(self, parent, row, col, label, frm, to, res, value, fmt, on_change):
        box = tk.Frame(parent, bg=PANEL)
        box.grid(row=row, column=col, sticky="ew", padx=(0, 18) if col == 0 else 0,
                 pady=(0, 14) if row == 0 else 0)
        top = tk.Frame(box, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top, text=label, bg=PANEL, fg=TEXT_DIM, font=F(12)).pack(side="left")
        val = tk.Label(top, text=fmt(value), bg=PANEL, fg=TEXT_DIM, font=F(12, mono=True))
        val.pack(side="right")
        var = tk.DoubleVar(value=value)

        def changed(_=None):
            v = round(round(var.get() / res) * res, 2)
            val.config(text=fmt(v))
            on_change(v)

        ttk.Scale(box, from_=frm, to=to, variable=var, command=changed,
                  style="Dark.Horizontal.TScale").pack(fill="x", pady=(6, 0))
        return var, changed

    def _build_params(self, parent):
        p = Panel(parent, "Параметри навчання")
        p.pack(fill="x", pady=(16, 0))
        grid = tk.Frame(p, bg=PANEL)
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1, uniform="g")
        grid.columnconfigure(1, weight=1, uniform="g")
        self.sliders = {
            "alpha": self._slider(grid, 0, 0, "Швидкість навчання, α", 0, 1, 0.01, 0.35,
                                  lambda v: f"{v:.2f}", lambda v: setattr(self.m, "alpha", v)),
            "gamma": self._slider(grid, 0, 1, "Дисконтування, γ", 0, 1, 0.01, 0.9,
                                  lambda v: f"{v:.2f}", lambda v: setattr(self.m, "gamma", v)),
            "eps": self._slider(grid, 1, 0, "Дослідження, ε", 0, 1, 0.01, 0.12,
                                lambda v: f"{v:.2f}", lambda v: setattr(self.m, "epsilon", v)),
            "speed": self._slider(grid, 1, 1, "Швидкість анімації", 1, 10, 1, 3,
                                  lambda v: f"×{int(v)}", lambda v: setattr(self, "speed", int(v))),
        }
        row = tk.Frame(p, bg=PANEL)
        row.pack(fill="x", pady=(14, 14))
        tk.Label(row, text="Максимум епізодів", bg=PANEL, fg=TEXT_DIM, font=F(12)).pack(side="left")
        self.max_ep_var = tk.StringVar(value="600")
        e = tk.Entry(row, textvariable=self.max_ep_var, width=9, justify="right", bg=PANEL2,
                     fg=TEXT, insertbackground=TEXT, relief="flat", font=F(12.5, mono=True),
                     highlightthickness=1, highlightbackground=BORDER, highlightcolor=AGENT)
        e.pack(side="right", ipady=4)
        e.bind("<Return>", self.on_max_ep)
        e.bind("<FocusOut>", self.on_max_ep)

        btns = tk.Frame(p, bg=PANEL)
        btns.pack(fill="x")
        self.btn_start = FlatButton(btns, "Старт", self.start_run, primary=True)
        self.btn_pause = FlatButton(btns, "Пауза", self.pause_run)
        self.btn_reset = FlatButton(btns, "Скинути", self.reset_all)
        self.btn_step = FlatButton(btns, "Крок", lambda: None if self.running else self.step())
        for k, b in enumerate([self.btn_start, self.btn_pause, self.btn_reset, self.btn_step]):
            b.grid(row=0, column=k, sticky="ew", padx=(0 if k == 0 else 8, 0))
            btns.columnconfigure(k, weight=1, uniform="b")
        self.btn_fast = FlatButton(p, "Швидке навчання (без анімації)", self.start_fast)
        self.btn_fast.pack(fill="x", pady=(8, 0))

    def _build_calc(self, parent):
        p = Panel(parent, "Обчислення на поточному кроці")
        p.pack(fill="x", pady=(16, 0))
        box = tk.Frame(p, bg=PANEL2, highlightbackground=BORDER, highlightthickness=1,
                       padx=14, pady=12)
        box.pack(fill="x")
        self.calc_label = tk.Label(box, bg=PANEL2, fg=TEXT_DIM, font=F(12, mono=True),
                                   justify="left", anchor="w", wraplength=350)
        self.calc_label.pack(fill="x")
        self.delta_label = tk.Label(box, text="", bg=PANEL2, fg=ACCENT,
                                    font=F(12, "bold", mono=True), anchor="w")
        self._reset_calc()

    def _reset_calc(self):
        self.calc_label.config(text="натисніть «Крок» або «Старт», щоб побачити перехід")
        self.delta_label.pack_forget()

    def _build_path_panel(self, parent):
        p = Panel(parent, "Q-значення вздовж поточного найкращого шляху")
        p.pack(fill="x", pady=(16, 0))
        f = tk.Frame(p, bg=PANEL)
        f.pack(fill="x")
        self.matrix = tk.Canvas(f, height=40, bg=PANEL, highlightthickness=0, width=340)
        msb = ttk.Scrollbar(f, orient="vertical", command=self.matrix.yview,
                            style="Dark.Vertical.TScrollbar")
        self.matrix.configure(yscrollcommand=msb.set)
        self.matrix.pack(side="left", fill="x", expand=True)
        msb.pack(side="right", fill="y")
        self.matrix_h = 0

        def wheel(e):
            if self.matrix_h <= int(self.matrix.cget("height")):
                return False
            self.matrix.yview_scroll(int(-e.delta / 120) * 2, "units")
            return True
        self.matrix._wheel = wheel

    def _build_state_panel(self, parent):
        p = Panel(parent, "Стан симуляції")
        p.pack(fill="x", pady=(16, 0))
        top = tk.Frame(p, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top, text="Прогрес навчання", bg=PANEL, fg=TEXT_DIM, font=F(12)).pack(side="left")
        self.prog_text = tk.Label(top, text="0 / 600", bg=PANEL, fg=TEXT_DIM, font=F(12, mono=True))
        self.prog_text.pack(side="right")
        self.progress = tk.Canvas(p, height=7, bg=LINE, highlightthickness=0)
        self.progress.pack(fill="x", pady=(6, 12))
        self.prog_fill = self.progress.create_rectangle(0, 0, 0, 7, fill=ACCENT, width=0)

        tiles = tk.Frame(p, bg=PANEL)
        tiles.pack(fill="x", pady=(0, 12))
        tiles.columnconfigure(0, weight=1, uniform="t")
        tiles.columnconfigure(1, weight=1, uniform="t")
        self.steps_val = self._tile(tiles, 0, "Кроків в епізоді", "0", TEXT)
        self.reward_val = self._tile(tiles, 1, "Винагорода за епізод", "+0.0", POSITIVE)

        tk.Label(p, text="Винагорода за останні епізоди", bg=PANEL, fg=TEXT_FAINT,
                 font=F(11)).pack(anchor="w", pady=(0, 4))
        self.spark = tk.Canvas(p, height=46, bg=PANEL, highlightthickness=0)
        self.spark.pack(fill="x", pady=(0, 14))
        self.pill = tk.Label(p, text="", font=F(12.5, "bold"), padx=14, pady=6,
                             highlightthickness=1)
        self.pill.pack(anchor="w")

    def _tile(self, parent, col, title, value, color):
        t = tk.Frame(parent, bg=PANEL2, highlightbackground=BORDER, highlightthickness=1,
                     padx=12, pady=10)
        t.grid(row=0, column=col, sticky="ew", padx=(0, 6) if col == 0 else (6, 0))
        tk.Label(t, text=title, bg=PANEL2, fg=TEXT_FAINT, font=F(11)).pack(anchor="w")
        v = tk.Label(t, text=value, bg=PANEL2, fg=color, font=F(16, "bold", mono=True))
        v.pack(anchor="w", pady=(2, 0))
        return v

    def _build_q_panel(self, parent):
        p = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1,
                     padx=20, pady=18)
        p.pack(fill="x", pady=(18, 0))
        head = tk.Frame(p, bg=PANEL)
        head.pack(fill="x")
        tk.Label(head, text="ТАБЛИЦЯ Q-ЗНАЧЕНЬ Q(S, A)", bg=PANEL, fg=TEXT_DIM,
                 font=F(12, "bold")).pack(side="left")
        acts = tk.Frame(head, bg=PANEL)
        acts.pack(side="right")
        self.btn_toggle_q = FlatButton(acts, "Показати таблицю", self.toggle_q,
                                       font=F(12, "bold"), pady=6)
        for b in [self.btn_toggle_q,
                  FlatButton(acts, "Завантажити CSV", self.export_csv, font=F(12, "bold"), pady=6),
                  FlatButton(acts, "Завантажити JSON", self.export_json, font=F(12, "bold"), pady=6)]:
            b.config(padx=14)
            b.pack(side="left", padx=(8, 0))
        note = tk.Label(p, text="Рядки — стани (прохідні клітинки), стовпці — дії. «—» означає, що "
                                "дія неможлива (стіна, перешкода або край сітки). Найкраща дія "
                                "виділена кольором, рамкою — значення, оновлене на останньому кроці. "
                                "CSV відкривається в Excel, JSON містить також параметри навчання та "
                                "карту.",
                        bg=PANEL, fg=TEXT_FAINT, font=F(11.5), wraplength=600, justify="left",
                        anchor="w")
        note.pack(fill="x", pady=(8, 0))
        auto_wrap(note)
        self.q_view = QTableView(p)  # показується кнопкою

    # ====================================================== сітка (Canvas)
    def build_grid(self):
        m = self.m
        big = m.is_big()
        vbw, vbh = PAD * 2 + m.cols * CELL, PAD * 2 + m.rows * CELL
        self.s = s = min(self.box_w / vbw, self.box_h / vbh)
        self.cv.config(width=round(vbw * s), height=round(vbh * s))
        self.cv.delete("all")
        self.fill_items, self.val_items, self.fill_level, self.base_color = {}, {}, {}, {}
        self.shown_vals = []
        for r in range(m.rows):
            for c in range(m.cols):
                self._draw_cell(c, r)
        # верхні шари: шлях, точки, підсвітка, агент
        self.path_line = self.cv.create_line(0, 0, 0, 0, fill=blend(PANEL, ACCENT, .85),
                                             width=2.5 if big else 2.5 * s, capstyle="round",
                                             joinstyle="round", state="hidden")
        self.hover_item = self.cv.create_rectangle(0, 0, 0, 0, outline=AGENT,
                                                   width=2 if big else 2 * s, state="hidden")
        dash = max(2, round(4 * (1 if big else s)))
        self.halo = self.cv.create_oval(0, 0, 0, 0, outline=AGENT, width=2.5 if big else 2.5 * s,
                                        dash=(dash, dash))
        self.agent_item = self.cv.create_oval(0, 0, 0, 0, fill=AGENT, outline=BG,
                                              width=2 if big else 2 * s)

    def _draw_cell(self, c, r):
        m, s, cv = self.m, self.s, self.cv
        big = m.is_big()
        i = r * m.cols + c
        tag = f"c{i}"
        x, y, w = (PAD + c * CELL) * s, (PAD + r * CELL) * s, CELL * s
        cell = (c, r)
        self.fill_items.pop(i, None)
        self.val_items.pop(i, None)
        self.fill_level.pop(i, None)
        if cell in m.obstacles:
            if big:
                cv.create_rectangle(x, y, x + w, y + w, fill=OBST_BIG, width=0, tags=tag)
            else:
                cv.create_rectangle(x, y, x + w, y + w, fill=OBST_FILL, outline=BORDER,
                                    width=max(1, s), tags=tag)
                k = 7 * s
                cv.create_line(x + k, y + w - k, x + w - k, y + k, fill=HATCH, width=1.4 * s, tags=tag)
                cv.create_line(x + k, y + k, x + w - k, y + w - k, fill=HATCH, width=1.4 * s, tags=tag)
            return
        rough = cell in m.rough
        base = ROUGH_BIG if big and rough else PANEL2
        self.base_color[i] = base
        self.fill_items[i] = cv.create_rectangle(
            x + s, y + s, x + w - s, y + w - s, fill=blend(base, ACCENT, 0.04),
            outline=blend(PANEL2, BORDER, .5) if big else BORDER,
            width=1 if big else max(1, s), tags=tag)
        if cell in (m.start, m.finish):
            tw = 2.5 if big else 2.5 * s
            cv.create_rectangle(x + s + tw / 2, y + s + tw / 2, x + w - s - tw / 2, y + w - s - tw / 2,
                                outline=ACCENT, width=tw, tags=tag)
        if not big:
            if rough:
                cv.create_text(x + w - 6 * s, y + 10 * s, text="3×", anchor="e", fill=TEXT_FAINT,
                               font=F(9 * s, "bold"), tags=tag)
            self.val_items[i] = cv.create_text(x + 6 * s, y + w - 4 * s, text="0", anchor="sw",
                                               fill=TEXT_DIM, font=F(9 * s, mono=True),
                                               state="hidden", tags=tag)
            if cell == m.start:
                cv.create_text(x + w / 2, y + w / 2, text="СТАРТ", fill=TEXT_FAINT,
                               font=F(8.5 * s, "bold"), tags=tag)
            if cell == m.finish:
                cv.create_text(x + w / 2, y + w / 2, text="ФІНІШ", fill=TEXT_FAINT,
                               font=F(8.5 * s, "bold"), tags=tag)

    def redraw_cells(self, cells):
        """Перемальовує лише змінені клітинки (швидке редагування великих сіток)."""
        for c, r in cells:
            i = r * self.m.cols + c
            self.cv.delete(f"c{i}")
            self._draw_cell(c, r)
            self.cv.tag_lower(f"c{i}", self.path_line)

    def center(self, i):
        c, r = self.m.cell_of(i)
        s = self.s
        return (PAD + c * CELL + CELL / 2) * s, (PAD + r * CELL + CELL / 2) * s

    # ============================================================ логіка
    def _push_history(self, r):
        self.reward_history.append(r)
        if len(self.reward_history) > HISTORY_LEN:
            self.reward_history.pop(0)

    def step(self):
        if not self.env_ok:
            return
        if self.episode >= self.max_episodes:
            self.stop_run("done")
            return
        res = self.m.apply_step(self.state)
        if res is None:
            self.finish_episode()
            self.render()
            return
        self.last_transition = dict(res, frm=self.state, step_no=self.steps + 1)
        self.ep_reward += res["reward"]
        self.steps += 1
        self.state = res["to"]
        if res["goal"] or self.steps >= self.m.step_cap():
            self.finish_episode()
        self.render()

    def finish_episode(self):
        self._push_history(self.ep_reward)
        self.episode += 1
        self.state = self.m.start_i
        self.steps = 0
        self.ep_reward = 0.0
        if self.episode >= self.max_episodes:
            self.stop_run("done")

    def start_run(self):
        if self.running or self.episode >= self.max_episodes or not self.env_ok:
            return
        self.running = True
        self.set_status("running")
        self._schedule_tick()

    def _schedule_tick(self):
        ms = max(35, 500 - self.speed * 45)
        self.after_id = self.root.after(ms, self._tick)

    def _tick(self):
        if not self.running or self.fast:
            return
        self.step()
        if self.running:
            self._schedule_tick()

    def start_fast(self):
        """Швидке навчання: цілі епізоди без анімації, порціями по ~25 мс."""
        if self.running or self.episode >= self.max_episodes or not self.env_ok:
            return
        self.running = self.fast = True
        self.set_status("running")
        self.state, self.steps, self.ep_reward = self.m.start_i, 0, 0.0
        self._last_draw = 0.0
        self.after_id = self.root.after(0, self._fast_loop)

    def _fast_loop(self):
        if not (self.running and self.fast):
            return
        t0 = time.perf_counter()
        run = self.m.run_episode_silent
        while time.perf_counter() - t0 < 0.025 and self.episode < self.max_episodes:
            self._push_history(run())
            self.episode += 1
        # перемальовуємо не частіше ніж 4 рази на секунду, решту часу — навчання
        now = time.perf_counter()
        if now - self._last_draw > 0.25 or self.episode >= self.max_episodes:
            self._last_draw = now
            self.render()
        if self.episode >= self.max_episodes:
            self.stop_run("done")
        else:
            self.after_id = self.root.after(1, self._fast_loop)

    def _cancel(self):
        self.running = self.fast = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def pause_run(self):
        self._cancel()
        self.set_status("paused")

    def stop_run(self, kind):
        self._cancel()
        self.set_status(kind)

    def reset_all(self):
        self.pause_run()
        self.m.reset_q()
        self.state = self.m.start_i
        self.episode = 0
        self.steps = 0
        self.ep_reward = 0.0
        self.reward_history = []
        self.last_transition = None
        self._reset_calc()
        self.set_status("paused")
        self.render()

    def set_status(self, kind):
        self.status_kind = kind
        bg, border, fg, text = STATUS_STYLE[kind]
        if kind == "running" and self.fast:
            text = "Швидке навчання…"
        self.pill.config(text="●  " + text, bg=bg, fg=fg, highlightbackground=border,
                         highlightcolor=border)
        self.btn_start.set_enabled(kind != "done" and self.env_ok)
        self.btn_fast.set_enabled(kind != "done" and self.env_ok)
        self.btn_step.set_enabled(self.env_ok)

    def on_max_ep(self, _=None):
        try:
            v = int(self.max_ep_var.get())
        except ValueError:
            v = 0
        self.max_episodes = v if v > 0 else 600
        self.max_ep_var.set(str(self.max_episodes))
        if self.episode < self.max_episodes and self.status_kind == "done":
            self.set_status("paused")
        self.render()

    # ========================================================== редактор
    def select_tool(self, tool):
        self.tool = tool
        for t, b in self.tool_btns.items():
            b.set_active(t == tool)

    def env_changed(self, cells=None):
        """Єдина точка входу після зміни середовища. cells=None — повна перебудова."""
        m = self.m
        m.rebuild()
        self.env_ok = m.finish_reachable()
        if self.env_ok:
            self.warn.pack_forget()
        else:
            self.warn.pack(fill="x", pady=(10, 0))
        self.cols_var.set(str(m.cols))
        self.rows_var.set(str(m.rows))
        if cells is None:
            self.build_grid()
        else:
            self.redraw_cells(cells)
        self.reset_all()

    def on_dims(self, _=None):
        try:
            c = int(self.cols_var.get())
        except ValueError:
            c = self.m.cols
        try:
            r = int(self.rows_var.get())
        except ValueError:
            r = self.m.rows
        c = min(MAX_COLS, max(MIN_COLS, c))
        r = min(MAX_ROWS, max(MIN_ROWS, r))
        if (c, r) != (self.m.cols, self.m.rows):
            self.resize_grid(c, r)
        else:
            self.cols_var.set(str(c))
            self.rows_var.set(str(r))

    def resize_grid(self, c, r):
        m = self.m
        finish_was_corner = m.finish == (m.cols - 1, m.rows - 1)
        m.cols, m.rows = c, r
        m.obstacles = {p for p in m.obstacles if m.in_bounds(*p)}
        m.rough = {p for p in m.rough if m.in_bounds(*p)}

        def clamp(p):
            return min(p[0], c - 1), min(p[1], r - 1)

        m.start = clamp(m.start)
        m.finish = (c - 1, r - 1) if finish_was_corner else clamp(m.finish)
        if m.finish == m.start:
            m.finish = (0, 0) if m.start == (c - 1, r - 1) else (c - 1, r - 1)
        m.obstacles.discard(m.start)
        m.obstacles.discard(m.finish)
        self.env_changed()

    def cell_from_event(self, e):
        c = int((e.x / self.s - PAD) // CELL)
        r = int((e.y / self.s - PAD) // CELL)
        return (c, r) if self.m.in_bounds(c, r) else None

    def show_hover(self, cell):
        if cell is None:
            self.cv.itemconfig(self.hover_item, state="hidden")
            return
        c, r = cell
        s = self.s
        x, y = (PAD + c * CELL + 2) * s, (PAD + r * CELL + 2) * s
        self.cv.coords(self.hover_item, x, y, x + (CELL - 4) * s, y + (CELL - 4) * s)
        self.cv.itemconfig(self.hover_item, state="normal")

    def on_press(self, e):
        cell = self.cell_from_event(e)
        if cell is None:
            return
        if self.tool in ("start", "finish"):
            self.place_terminal(cell)
            return
        group = self.m.rough if self.tool == "rough" else self.m.obstacles
        self.paint = "erase" if (self.tool == "erase" or cell in group) else "set"
        self.paint_cell(cell)

    def on_motion(self, e):
        cell = self.cell_from_event(e)
        self.show_hover(cell)
        if self.paint and cell and cell != self.last_painted:
            self.paint_cell(cell)

    def on_release(self, _e):
        self.paint = None
        self.last_painted = None

    def paint_cell(self, cell):
        m = self.m
        self.last_painted = cell
        if cell in (m.start, m.finish):
            return
        had_o, had_r = cell in m.obstacles, cell in m.rough
        if self.paint == "erase":
            if self.tool in ("erase", "obstacle"):
                m.obstacles.discard(cell)
            if self.tool in ("erase", "rough"):
                m.rough.discard(cell)
        elif self.tool == "obstacle":
            m.obstacles.add(cell)
            m.rough.discard(cell)
        elif self.tool == "rough":
            m.rough.add(cell)
            m.obstacles.discard(cell)
        if had_o != (cell in m.obstacles) or had_r != (cell in m.rough):
            self.env_changed([cell])

    def place_terminal(self, cell):
        m = self.m
        if cell in (m.start, m.finish):
            return
        if self.tool == "start":
            old, m.start = m.start, cell
        else:
            old, m.finish = m.finish, cell
        m.obstacles.discard(cell)
        self.env_changed([old, cell])

    def clear_env(self):
        self.m.obstacles, self.m.rough = set(), set()
        self.env_changed()

    def random_env(self):
        m = self.m
        for _ in range(200):
            m.obstacles, m.rough = set(), set()
            for r in range(m.rows):
                for c in range(m.cols):
                    if (c, r) in (m.start, m.finish):
                        continue
                    x = random.random()
                    if x < 0.22:
                        m.obstacles.add((c, r))
                    elif x < 0.28:
                        m.rough.add((c, r))
            m.rebuild()
            if m.finish_reachable():
                break
        self.env_changed()

    def default_env(self):
        m = self.m
        m.cols, m.rows = DEFAULT_ENV["cols"], DEFAULT_ENV["rows"]
        m.obstacles, m.rough = set(DEFAULT_ENV["obstacles"]), set(DEFAULT_ENV["rough"])
        m.start, m.finish = DEFAULT_ENV["start"], DEFAULT_ENV["finish"]
        self.env_changed()

    # ============================================================ рендер
    def render(self):
        m, cv = self.m, self.cv
        big = m.is_big()
        q, nb = m.q, m.nb
        V = {}
        min_v, max_v = float("inf"), float("-inf")
        for i in m.free:
            opts = nb[i]
            if opts:
                v = q[opts[0][1]]
                for o in opts:
                    x = q[o[1]]
                    if x > v:
                        v = x
            else:
                v = 0.0
            V[i] = v
            if v < min_v:
                min_v = v
            if v > max_v:
                max_v = v
        rng = (max_v - min_v) or 1
        # колір змінюється лише там, де він справді змінився (аналог fillCache)
        fill_level, fill_items, base_color = self.fill_level, self.fill_items, self.base_color
        for i, v in V.items():
            lvl = round((0.04 + 0.55 * (v - min_v) / rng) * 1000)
            if fill_level.get(i) != lvl:
                fill_level[i] = lvl
                cv.itemconfig(fill_items[i], fill=blend(base_color[i], ACCENT, lvl / 1000))

        for t in self.shown_vals:
            cv.itemconfig(t, state="hidden")
        self.shown_vals = []
        path = m.greedy_path()
        pts = []
        for i in path:
            pts.extend(self.center(i))
            t = self.val_items.get(i)
            if t:
                cv.itemconfig(t, text=f"{V[i]:.1f}", state="normal")
                self.shown_vals.append(t)
        if len(path) >= 2:
            cv.coords(self.path_line, *pts)
            cv.itemconfig(self.path_line, state="normal")
        else:
            cv.itemconfig(self.path_line, state="hidden")
        # на великій сітці розміри агента й точок задаються в пікселях екрана
        s = self.s
        dot_r = min(1.6, CELL * 0.3 * s) if big else 3.2 * s
        cv.delete("dot")
        for i in path:
            x, y = self.center(i)
            cv.create_oval(x - dot_r, y - dot_r, x + dot_r, y + dot_r, fill=ACCENT, width=0, tags="dot")
        cv.tag_lower("dot", self.hover_item)
        x, y = self.center(self.state)
        h, a = (9, 4) if big else (19 * s, 7 * s)
        cv.coords(self.halo, x - h, y - h, x + h, y + h)
        cv.coords(self.agent_item, x - a, y - a, x + a, y + a)

        t = self.last_transition
        if t:
            (fx, fy), (tx, ty) = m.cell_of(t["frm"]), m.cell_of(t["to"])
            bonus = f", бонус за фініш +{GOAL_BONUS}" if t["goal"] else ""
            self.calc_label.config(text=(
                f"крок {t['step_no']} · клітинка ({fx},{fy}) → дія: {t['dir']} → ({tx},{ty})\n"
                f"r = {signed(t['reward'], 2)}  (вартість переходу {t['cost']}{bonus})\n"
                f"Q(s,a) ← Q + α·[ r + γ·maxQ(s′,·) − Q ]"))
            self.delta_label.config(text=f"{t['old_q']:.2f} → {t['new_q']:.2f}")
            if not self.delta_label.winfo_ismapped():
                self.delta_label.pack(fill="x")

        self._render_matrix(path, max_v)

        self.ep_now.config(text=str(self.episode))
        self.ep_max.config(text=f" / {self.max_episodes}")
        self.prog_text.config(text=f"{self.episode} / {self.max_episodes}")
        w = self.progress.winfo_width()
        self.progress.coords(self.prog_fill, 0, 0, w * min(1.0, self.episode / self.max_episodes), 7)
        self.steps_val.config(text=str(self.steps))
        self.reward_val.config(text=signed(self.ep_reward, 1),
                               fg=POSITIVE if self.ep_reward >= 0 else NEGATIVE)
        self._render_sparkline()
        self.render_q_table()

    def _render_matrix(self, path, max_v):
        mc, m = self.matrix, self.m
        mc.delete("all")
        w = mc.winfo_width() if mc.winfo_width() > 1 else 340
        if len(path) < 2:
            mc.create_text(2, 8, anchor="nw", width=w - 4, fill=TEXT_FAINT, font=F(12),
                           text="агент ще не проклав стабільного шляху до фінішу — натисніть «Старт»")
            total = 40
        else:
            RH = 31
            mono_b = F(12, "bold", mono=True)
            for k in range(len(path) - 1):
                a, b = path[k], path[k + 1]
                (ax, ay), (bx, by) = m.cell_of(a), m.cell_of(b)
                val = m.q[m.edge_key(a, b)]
                pct = max(4, min(100, (val / (max_v or 1)) * 100))
                y = k * RH + RH / 2
                mc.create_text(2, y, anchor="w", text=f"({ax},{ay}) → ({bx},{by})", fill=ACCENT,
                               font=mono_b)
                mc.create_text(w - 2, y, anchor="e", text=f"{val:.2f}", fill=ACCENT, font=mono_b)
                bx0 = w - 2 - 42 - 8 - 56
                mc.create_rectangle(bx0, y - 3, bx0 + 56, y + 3, fill=LINE, width=0)
                mc.create_rectangle(bx0, y - 3, bx0 + 56 * pct / 100, y + 3, fill=ACCENT, width=0)
                if k < len(path) - 2:
                    mc.create_line(0, (k + 1) * RH, w, (k + 1) * RH, fill=LINE)
            total = (len(path) - 1) * RH
        self.matrix_h = total
        mc.config(scrollregion=(0, 0, w, total))
        h = min(200, total)
        if int(mc.cget("height")) != h:
            mc.config(height=h)

    def _render_sparkline(self):
        sp = self.spark
        sp.delete("all")
        hist = self.reward_history
        if len(hist) < 2:
            return
        w = sp.winfo_width() if sp.winfo_width() > 1 else 360
        mn, mx = min(hist), max(hist)
        rg = (mx - mn) or 1
        n = len(hist)
        pts = []
        for k, v in enumerate(hist):
            pts += [k / (n - 1) * (w - 4) + 2, 42 - (v - mn) / rg * 36]
        sp.create_line(*pts, fill=POSITIVE, width=2, capstyle="round", joinstyle="round")

    # ======================================================= таблиця Q
    def toggle_q(self):
        self.q_visible = not self.q_visible
        if self.q_visible:
            self.q_view.pack(fill="x", pady=(12, 0))
        else:
            self.q_view.pack_forget()
        self.btn_toggle_q.config(text="Сховати таблицю" if self.q_visible else "Показати таблицю")
        self.render_q_table(True)

    def render_q_table(self, force=False):
        if not self.q_visible:
            return
        # на великих сітках таблиця перебудовується не частіше ніж раз на 0.5 с
        gap = 0.5 if self.m.cols * self.m.rows > 400 else 0.0
        now = time.perf_counter()
        if not force and now - self.q_last < gap:
            if self.q_timer is None:
                delay = int((gap - (now - self.q_last)) * 1000) + 1
                self.q_timer = self.root.after(delay, self._q_timer_fire)
            return
        self.q_last = now
        upd = None
        t = self.last_transition
        if t:
            upd = (t["frm"], t["dir"])
        self.q_view.set_data(self.m.q_table_data(), self.state, upd)

    def _q_timer_fire(self):
        self.q_timer = None
        self.render_q_table(True)

    def file_base(self):
        return f"q-table_{self.m.cols}x{self.m.rows}_ep{self.episode}"

    def csv_text(self):
        """CSV для Excel: розділювач «;», десяткова кома, рядки через CRLF."""
        def num(v):
            return "" if v is None else f"{v:.4f}".replace(".", ",")
        lines = [";".join(["Стан", "x", "y", "Тип", *[d[2] for d in DIRS], "V(s)", "Найкраща дія"])]
        for row in self.m.q_table_data():
            lines.append(";".join([f"({row['col']},{row['row']})", str(row["col"]), str(row["row"]),
                                   row["kind"], *[num(row["q"][d[2]]) for d in DIRS],
                                   num(row["v"]), row["best"] or ""]))
        return "\r\n".join(lines)

    def json_data(self):
        m = self.m

        def cid(cell):
            return f"{cell[0]},{cell[1]}"

        def order(cells):
            return [cid(p) for p in sorted(cells, key=lambda p: (p[1], p[0]))]

        created = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        return {
            "created": created,
            "params": {"alpha": m.alpha, "gamma": m.gamma, "epsilon": m.epsilon,
                       "episode": self.episode, "maxEpisodes": self.max_episodes},
            "environment": {"cols": m.cols, "rows": m.rows, "start": cid(m.start),
                            "finish": cid(m.finish), "obstacles": order(m.obstacles),
                            "rough": order(m.rough)},
            "table": [{"state": r["id"], "x": r["col"], "y": r["row"], "type": r["kind"],
                       "Q": r["q"], "V": round(r["v"], 6), "bestAction": r["best"]}
                      for r in m.q_table_data()],
            "greedyPath": [cid(m.cell_of(i)) for i in m.greedy_path()],
        }

    def export_csv(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, initialfile=self.file_base() + ".csv", defaultextension=".csv",
            filetypes=[("CSV (Excel)", "*.csv"), ("Усі файли", "*.*")])
        if path:
            # utf-8-sig додає BOM, щоб Excel правильно показав кирилицю
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                f.write(self.csv_text())
            messagebox.showinfo("Збережено", f"Таблицю збережено:\n{path}", parent=self.root)

    def export_json(self):
        path = filedialog.asksaveasfilename(
            parent=self.root, initialfile=self.file_base() + ".json", defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("Усі файли", "*.*")])
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.json_data(), f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Збережено", f"Таблицю збережено:\n{path}", parent=self.root)

    # ===================================================== прокрутка колесом
    def _on_wheel(self, e):
        try:
            w = self.root.winfo_containing(e.x_root, e.y_root)
        except (KeyError, tk.TclError):
            w = None
        while w is not None:
            handler = getattr(w, "_wheel", None)
            if handler and handler(e):
                return
            w = getattr(w, "master", None)
        self.page_canvas.yview_scroll(int(-e.delta / 120) * 3, "units")


def main():
    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = min(1320, sw - 20), min(900, sh - 80)
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{max(0, (sh - h) // 2 - 20)}")
    root.minsize(min(1000, w), min(600, h))
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
