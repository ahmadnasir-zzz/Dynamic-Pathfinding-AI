# AI 2002 – Assignment 2, Q6 | Dynamic Pathfinding Agent
# Run: python pathfinding_agent.py 

import tkinter as tk
from tkinter import font as tkfont
import heapq, math, random, time

# ── Settings ──────────────────────────────────────────────────
MIN_CELL   = 12
MAX_CELL   = 40
DEF_ROWS   = 20
DEF_COLS   = 25
ANIM_DELAY = 18     # ms per animation frame
MOVE_DELAY = 130    # ms per agent step
DYN_PROB   = 0.06   # wall-spawn probability per step

EMPTY = 0; WALL = 1; VISITED = 2; PATH = 3; FRONTIER = 4

COLORS = {
    EMPTY:    "#1C2235",
    WALL:     "#0D0F17",
    FRONTIER: "#F1C40F",   # Yellow
    VISITED:  "#2E86C1",   # Blue
    PATH:     "#27AE60",   # Green
    "start":  "#2ECC71",
    "goal":   "#E74C3C",
    "agent":  "#9B59B6",   # Purple
    "grid":   "#2C3E50",
}

# ── Heuristics ────────────────────────────────────────────────
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# ── A* Search (expanded list) ─────────────────────────────────
def run_astar(grid, rows, cols, start, goal, h_fn):
    g_score = {start: 0}
    parent  = {start: None}
    closed  = set()
    counter = 0
    heap    = [(h_fn(start, goal), 0, start)]
    visited_order = []

    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        visited_order.append(cur)

        if cur == goal:
            path, node = [], goal
            while node is not None:
                path.append(node); node = parent[node]
            path.reverse()
            return path, visited_order, len(closed)

        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
                nb = (nr, nc)
                if nb in closed: continue
                ng = g_score[cur] + 1
                if ng < g_score.get(nb, float('inf')):
                    g_score[nb] = ng
                    parent[nb]  = cur
                    counter    += 1
                    heapq.heappush(heap, (ng + h_fn(nb, goal), counter, nb))

    return [], visited_order, len(closed)

# ── Greedy Best-First Search (strict visited list) ────────────
def run_gbfs(grid, rows, cols, start, goal, h_fn):
    visited  = {start: None}
    counter  = 0
    heap     = [(h_fn(start, goal), 0, start)]
    vis_ord  = []
    expanded = 0

    while heap:
        _, _, cur = heapq.heappop(heap)
        vis_ord.append(cur)
        expanded += 1

        if cur == goal:
            path, node = [], goal
            while node is not None:
                path.append(node); node = visited[node]
            path.reverse()
            return path, vis_ord, expanded

        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
                nb = (nr, nc)
                if nb not in visited:
                    visited[nb] = cur
                    counter    += 1
                    heapq.heappush(heap, (h_fn(nb, goal), counter, nb))

    return [], vis_ord, expanded


# ── Main Application ──────────────────────────────────────────
class App:
    S_IDLE = "IDLE"; S_SEARCH = "SEARCH"; S_DONE = "DONE"; S_DYNAMIC = "DYNAMIC"
    E_WALL = "wall"; E_ERASE  = "erase";  E_START = "start"; E_GOAL   = "goal"

    def __init__(self, root):
        self.root = root
        root.title("AI 2002 — Dynamic Pathfinding Agent")
        root.configure(bg="#12151F")
        root.resizable(True, True)

        self.rows      = DEF_ROWS
        self.cols      = DEF_COLS
        self.cell_size = 28
        self.grid      = [[EMPTY]*self.cols for _ in range(self.rows)]
        self.start     = (0, 0)
        self.goal      = (self.rows-1, self.cols-1)

        self.algo      = tk.StringVar(value="astar")
        self.heur      = tk.StringVar(value="manhattan")
        self.edit_mode = self.E_WALL
        self.state     = self.S_IDLE
        self.density   = tk.DoubleVar(value=28)

        self.visited_set  = set()
        self.frontier_set = set()
        self.path_set     = set()
        self.vis_order    = []
        self.final_path   = []
        self.vis_index    = 0
        self._anim_id     = None

        self.agent_pos   = None
        self.remain_path = []
        self.dyn_added   = set()
        self._dyn_id     = None

        self.m_nodes  = tk.StringVar(value="0")
        self.m_cost   = tk.StringVar(value="0")
        self.m_time   = tk.StringVar(value="0.00 ms")
        self.m_status = tk.StringVar(value="Set start (S) and goal (G), then click Run Search.")

        self._build_ui()
        self.redraw()

    # ── UI ────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root
        self.f_title   = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self.f_section = tkfont.Font(family="Segoe UI", size=9,  weight="bold")
        self.f_btn     = tkfont.Font(family="Segoe UI", size=9,  weight="bold")
        self.f_lbl     = tkfont.Font(family="Segoe UI", size=9)
        self.f_metric  = tkfont.Font(family="Consolas", size=11, weight="bold")
        self.f_small   = tkfont.Font(family="Segoe UI", size=8)

        panel = tk.Frame(root, bg="#1A1F30", width=230)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(6,0), pady=6)
        panel.pack_propagate(False)

        right = tk.Frame(root, bg="#12151F")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.canvas = tk.Canvas(right, bg="#0E1016", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>",  self._on_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

        sb = tk.Frame(right, bg="#0D0F17", height=26)
        sb.pack(fill=tk.X, side=tk.BOTTOM)
        tk.Label(sb, textvariable=self.m_status, bg="#0D0F17", fg="#7F8C8D",
                 font=self.f_small, anchor="w").pack(fill=tk.X, padx=8, pady=4)

        p = panel
        tk.Label(p, text="PATHFINDING AGENT", bg="#1A1F30", fg="#5DADE2",
                 font=self.f_title, anchor="w").pack(fill=tk.X, padx=10, pady=(10,0))
        tk.Label(p, text="AI 2002  ·  Assignment 2  ·  Q6", bg="#1A1F30", fg="#566573",
                 font=self.f_small, anchor="w").pack(fill=tk.X, padx=10, pady=(0,6))
        self._div(p)

        self._sec(p, "⊞  GRID CONFIGURATION")
        f = tk.Frame(p, bg="#1A1F30"); f.pack(fill=tk.X, padx=10, pady=4)
        self.v_rows = tk.IntVar(value=self.rows)
        self.v_cols = tk.IntVar(value=self.cols)
        tk.Label(f, text="Rows:", bg="#1A1F30", fg="#AEB6BF",
                 font=self.f_lbl).grid(row=0, column=0, sticky="w", pady=2)
        tk.Spinbox(f, from_=5, to=40, textvariable=self.v_rows, width=5,
                   bg="#2C3E50", fg="#ECF0F1", buttonbackground="#34495E",
                   relief=tk.FLAT, font=self.f_lbl).grid(row=0, column=1, sticky="ew", padx=(6,0))
        tk.Label(f, text="Cols:", bg="#1A1F30", fg="#AEB6BF",
                 font=self.f_lbl).grid(row=1, column=0, sticky="w", pady=2)
        tk.Spinbox(f, from_=5, to=55, textvariable=self.v_cols, width=5,
                   bg="#2C3E50", fg="#ECF0F1", buttonbackground="#34495E",
                   relief=tk.FLAT, font=self.f_lbl).grid(row=1, column=1, sticky="ew", padx=(6,0))
        f.columnconfigure(1, weight=1)

        df = tk.Frame(p, bg="#1A1F30"); df.pack(fill=tk.X, padx=10, pady=(4,0))
        tk.Label(df, text="Wall density:", bg="#1A1F30", fg="#AEB6BF",
                 font=self.f_lbl).pack(anchor="w")
        dr = tk.Frame(df, bg="#1A1F30"); dr.pack(fill=tk.X)
        tk.Scale(dr, from_=0, to=60, orient=tk.HORIZONTAL, variable=self.density,
                 bg="#1A1F30", fg="#ECF0F1", troughcolor="#2C3E50",
                 highlightthickness=0, font=self.f_small,
                 sliderlength=14, length=140, showvalue=False).pack(side=tk.LEFT)
        self._dens_lbl = tk.Label(dr, text="28%", bg="#1A1F30", fg="#F1C40F",
                                   font=self.f_metric, width=4)
        self._dens_lbl.pack(side=tk.LEFT)
        self.density.trace_add("write",
            lambda *a: self._dens_lbl.config(text=f"{int(self.density.get())}%"))

        gf = tk.Frame(p, bg="#1A1F30"); gf.pack(fill=tk.X, padx=10, pady=6)
        self._btn(gf, "Apply Grid",   self._apply_grid,  "#1A5276").pack(fill=tk.X, pady=2)
        row = tk.Frame(gf, bg="#1A1F30"); row.pack(fill=tk.X)
        self._btn(row, "Random Maze", self._random_maze, "#145A32").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        self._btn(row, "Clear Grid",  self._clear_grid,  "#6E2F1A").pack(side=tk.LEFT, expand=True, fill=tk.X)
        self._div(p)

        self._sec(p, "⚙  ALGORITHM")
        af = tk.Frame(p, bg="#1A1F30"); af.pack(fill=tk.X, padx=12, pady=4)
        for text, val in [("A* Search (f = g + h)", "astar"), ("Greedy BFS (f = h only)", "greedy")]:
            tk.Radiobutton(af, text=text, variable=self.algo, value=val,
                           bg="#1A1F30", fg="#AEB6BF", selectcolor="#2C3E50",
                           activebackground="#1A1F30", activeforeground="#ECF0F1",
                           font=self.f_lbl, cursor="hand2").pack(anchor="w", pady=1)

        self._sec(p, "📐  HEURISTIC")
        hf = tk.Frame(p, bg="#1A1F30"); hf.pack(fill=tk.X, padx=12, pady=4)
        for text, val in [("Manhattan  |x1-x2|+|y1-y2|", "manhattan"),
                           ("Euclidean  √(dx²+dy²)", "euclidean")]:
            tk.Radiobutton(hf, text=text, variable=self.heur, value=val,
                           bg="#1A1F30", fg="#AEB6BF", selectcolor="#2C3E50",
                           activebackground="#1A1F30", activeforeground="#ECF0F1",
                           font=self.f_lbl, cursor="hand2").pack(anchor="w", pady=1)
        self._div(p)

        self._sec(p, "✏  EDIT MODE")
        ef = tk.Frame(p, bg="#1A1F30"); ef.pack(fill=tk.X, padx=10, pady=4)
        r1 = tk.Frame(ef, bg="#1A1F30"); r1.pack(fill=tk.X, pady=2)
        r2 = tk.Frame(ef, bg="#1A1F30"); r2.pack(fill=tk.X, pady=2)
        self._edit_btns = {}
        for label, mode, frame, bg in [
            ("Draw Walls",  self.E_WALL,  r1, "#1A5276"),
            ("Erase Walls", self.E_ERASE, r1, "#512E5F"),
            ("Set Start",   self.E_START, r2, "#145A32"),
            ("Set Goal",    self.E_GOAL,  r2, "#6E2F1A"),
        ]:
            b = self._btn(frame, label, lambda m=mode: self._set_edit(m), bg)
            b.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
            self._edit_btns[mode] = (b, bg)
        self._highlight_edit()
        self._div(p)

        self._sec(p, "▶  RUN")
        rb = tk.Frame(p, bg="#1A1F30"); rb.pack(fill=tk.X, padx=10, pady=4)
        self._btn(rb, "▶  Run Search",       self._run_search,  "#1A5276", h=2).pack(fill=tk.X, pady=2)
        self._btn(rb, "⚡  Dynamic Mode",     self._run_dynamic, "#7D6608", h=2).pack(fill=tk.X, pady=2)
        self._btn(rb, "■  Stop / Reset",     self._stop,        "#641E16"     ).pack(fill=tk.X, pady=2)
        self._div(p)

        self._sec(p, "📊  METRICS")
        mf = tk.Frame(p, bg="#1A1F30"); mf.pack(fill=tk.X, padx=10, pady=6)
        for label, var, col in [("Nodes Expanded", self.m_nodes, "#F1C40F"),
                                  ("Path Cost",      self.m_cost,  "#27AE60"),
                                  ("Exec Time",      self.m_time,  "#5DADE2")]:
            row = tk.Frame(mf, bg="#1A1F30"); row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, bg="#1A1F30", fg="#7F8C8D",
                     font=self.f_lbl, anchor="w").pack(side=tk.LEFT)
            tk.Label(row, textvariable=var, bg="#1A1F30", fg=col,
                     font=self.f_metric, anchor="e").pack(side=tk.RIGHT)
        self._div(p)

        self._sec(p, "🎨  LEGEND")
        lf = tk.Frame(p, bg="#1A1F30"); lf.pack(fill=tk.X, padx=10, pady=4)
        for color, label in [("#2ECC71", "Start"), ("#E74C3C", "Goal"),
                               ("#F1C40F", "Frontier  (in queue)"),
                               ("#2E86C1", "Expanded / Visited"),
                               ("#27AE60", "Final Path"),
                               ("#9B59B6", "Agent  (Dynamic Mode)"),
                               ("#0D0F17", "Wall")]:
            row = tk.Frame(lf, bg="#1A1F30"); row.pack(fill=tk.X, pady=1)
            tk.Canvas(row, width=13, height=13, bg=color,
                      highlightthickness=1, highlightbackground="#2C3E50").pack(side=tk.LEFT)
            tk.Label(row, text=f"  {label}", bg="#1A1F30", fg="#AEB6BF",
                     font=self.f_small, anchor="w").pack(side=tk.LEFT)

    def _btn(self, parent, text, cmd, bg="#1A5276", h=1):
        return tk.Button(parent, text=text, command=cmd, bg=bg, fg="#ECF0F1",
                         activebackground="#2E86C1", activeforeground="white",
                         relief=tk.FLAT, font=self.f_btn, height=h, cursor="hand2", padx=4, pady=2)

    def _sec(self, p, text):
        tk.Label(p, text=text, bg="#1A1F30", fg="#85929E",
                 font=self.f_section, anchor="w").pack(fill=tk.X, padx=10, pady=(6,2))

    def _div(self, p):
        tk.Frame(p, bg="#2C3E50", height=1).pack(fill=tk.X, padx=8, pady=3)

    def _set_edit(self, mode):
        self.edit_mode = mode
        self._highlight_edit()
        self.m_status.set({
            self.E_WALL:  "Click / drag to draw walls.",
            self.E_ERASE: "Click / drag to erase walls.",
            self.E_START: "Click a cell to place the Start node (S).",
            self.E_GOAL:  "Click a cell to place the Goal node (G).",
        }[mode])

    def _highlight_edit(self):
        active = {self.E_WALL: "#2E86C1", self.E_ERASE: "#8E44AD",
                  self.E_START: "#1E8449", self.E_GOAL: "#C0392B"}
        for mode, (btn, base) in self._edit_btns.items():
            btn.config(bg=active[mode] if mode == self.edit_mode else base)

    # ── Grid helpers ──────────────────────────────────────────
    def _apply_grid(self):
        self._stop()
        self.rows  = max(5, min(40, self.v_rows.get()))
        self.cols  = max(5, min(55, self.v_cols.get()))
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)
        self.grid  = [[EMPTY]*self.cols for _ in range(self.rows)]
        self._random_maze()

    def _random_maze(self):
        d = self.density.get() / 100.0
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) in (self.start, self.goal):
                    self.grid[r][c] = EMPTY
                else:
                    self.grid[r][c] = WALL if random.random() < d else EMPTY
        self._reset_overlay()
        self.redraw()

    def _clear_grid(self):
        self._stop()
        self.grid = [[EMPTY]*self.cols for _ in range(self.rows)]
        self._reset_overlay()
        self.redraw()

    def _reset_overlay(self):
        self.visited_set = set(); self.frontier_set = set(); self.path_set = set()
        self.vis_order   = [];    self.final_path   = [];    self.vis_index = 0
        self.dyn_added   = set(); self.agent_pos    = None;  self.remain_path = []
        self.m_nodes.set("0"); self.m_cost.set("0"); self.m_time.set("0.00 ms")

    # ── Canvas interaction ────────────────────────────────────
    def _offsets(self):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        cs = self.cell_size
        return max(0, (cw - cs*self.cols)//2), max(0, (ch - cs*self.rows)//2)

    def _cell_at(self, event):
        ox, oy = self._offsets()
        cs = self.cell_size
        c = (event.x - ox) // cs
        r = (event.y - oy) // cs
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return int(r), int(c)
        return None

    def _on_click(self, event):
        if self.state in (self.S_SEARCH, self.S_DYNAMIC): return
        cell = self._cell_at(event)
        if not cell: return
        r, c = cell
        if self.edit_mode == self.E_START:
            self.grid[self.start[0]][self.start[1]] = EMPTY
            self.start = (r, c); self.grid[r][c] = EMPTY; self._reset_overlay()
        elif self.edit_mode == self.E_GOAL:
            self.grid[self.goal[0]][self.goal[1]] = EMPTY
            self.goal = (r, c); self.grid[r][c] = EMPTY; self._reset_overlay()
        elif self.edit_mode == self.E_WALL:
            if (r,c) not in (self.start, self.goal): self.grid[r][c] = WALL
        elif self.edit_mode == self.E_ERASE:
            if (r,c) not in (self.start, self.goal): self.grid[r][c] = EMPTY
        self.redraw()

    def _on_drag(self, event):
        if self.state in (self.S_SEARCH, self.S_DYNAMIC): return
        cell = self._cell_at(event)
        if not cell or cell in (self.start, self.goal): return
        r, c = cell
        if   self.edit_mode == self.E_WALL:  self.grid[r][c] = WALL
        elif self.edit_mode == self.E_ERASE: self.grid[r][c] = EMPTY
        self.redraw()

    # ── Drawing ───────────────────────────────────────────────
    def _recalc_cell(self):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.cell_size = max(MIN_CELL, min(MAX_CELL, cw//self.cols, ch//self.rows))

    def redraw(self):
        self._recalc_cell()
        cs = self.cell_size
        ox, oy = self._offsets()
        cv = self.canvas
        cv.delete("all")

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = ox + c*cs; y1 = oy + r*cs
                x2 = x1 + cs;   y2 = y1 + cs
                cell = (r, c)

                if self.grid[r][c] == WALL:
                    cv.create_rectangle(x1, y1, x2, y2, fill="#0D0F17", outline="#0D0F17")
                    if cs >= 14:
                        cv.create_line(x1, y1, x2, y1, fill="#1C2030")
                        cv.create_line(x1, y1, x1, y2, fill="#1C2030")
                    continue

                if   cell in self.path_set:     fill = COLORS[PATH]
                elif cell in self.visited_set:  fill = COLORS[VISITED]
                elif cell in self.frontier_set: fill = COLORS[FRONTIER]
                else:                           fill = COLORS[EMPTY]

                cv.create_rectangle(x1, y1, x2, y2, fill=fill,
                                    outline=COLORS["grid"], width=1)

                if cell == self.start:
                    p = max(2, cs//6)
                    cv.create_rectangle(x1+p, y1+p, x2-p, y2-p,
                                        fill=COLORS["start"], outline="#82E0AA", width=2)
                    if cs >= 18:
                        cv.create_text(x1+cs//2, y1+cs//2, text="S", fill="white",
                                       font=("Segoe UI", max(8,cs//3), "bold"))
                elif cell == self.goal:
                    p = max(2, cs//6)
                    cv.create_rectangle(x1+p, y1+p, x2-p, y2-p,
                                        fill=COLORS["goal"], outline="#F1948A", width=2)
                    if cs >= 18:
                        cv.create_text(x1+cs//2, y1+cs//2, text="G", fill="white",
                                       font=("Segoe UI", max(8,cs//3), "bold"))

                if self.state == self.S_DYNAMIC and cell == self.agent_pos:
                    cx = x1+cs//2; cy = y1+cs//2; rad = max(4, cs//2-3)
                    cv.create_oval(cx-rad, cy-rad, cx+rad, cy+rad,
                                   fill=COLORS["agent"], outline="#D2B4DE", width=2)

    # ── Algorithm runner ──────────────────────────────────────
    def _invoke(self, from_pos=None):
        h_fn = manhattan if self.heur.get() == "manhattan" else euclidean
        src  = from_pos or self.start
        t0   = time.perf_counter()
        if self.algo.get() == "astar":
            path, vis, n = run_astar(self.grid, self.rows, self.cols, src, self.goal, h_fn)
        else:
            path, vis, n = run_gbfs( self.grid, self.rows, self.cols, src, self.goal, h_fn)
        ms = (time.perf_counter() - t0) * 1000
        self.m_nodes.set(str(n))
        self.m_cost.set(str(len(path)-1) if path else "—")
        self.m_time.set(f"{ms:.2f} ms")
        return path, vis, ms

    # ── Run Search (animated) ─────────────────────────────────
    def _run_search(self):
        self._cancel(); self._reset_overlay()
        self.state = self.S_SEARCH
        self.root.title("AI 2002 – Pathfinding  [SEARCHING…]")

        path, vis, ms = self._invoke()
        self.vis_order = vis; self.final_path = path; self.vis_index = 0

        if not path:
            self.m_status.set("❌  No path found — clear some walls and try again.")
            self.state = self.S_DONE
            self.root.title("AI 2002 – Pathfinding  [NO PATH]")
            return

        self.m_status.set(f"✅  Path found  ·  Cost = {len(path)-1}  ·  "
                          f"Nodes = {self.m_nodes.get()}  ·  {ms:.2f} ms")
        self._anim_step()

    def _anim_step(self):
        if self.state != self.S_SEARCH: return
        for _ in range(4):
            if self.vis_index < len(self.vis_order):
                self.visited_set.add(self.vis_order[self.vis_index])
                self.vis_index += 1
            else:
                for cell in self.final_path: self.path_set.add(cell)
                self.state = self.S_DONE
                self.root.title("AI 2002 – Pathfinding  [DONE]")
                self.redraw(); return
        self.redraw()
        self._anim_id = self.root.after(ANIM_DELAY, self._anim_step)

    # ── Dynamic Mode ──────────────────────────────────────────
    def _run_dynamic(self):
        self._cancel(); self._reset_overlay()
        self.state = self.S_DYNAMIC
        self.root.title("AI 2002 – Pathfinding  [DYNAMIC MODE]")

        path, vis, ms = self._invoke()
        if not path:
            self.m_status.set("❌  No initial path — clear some walls first.")
            self.state = self.S_DONE; return

        self.visited_set = set(vis); self.path_set = set(path)
        self.final_path  = path[:];  self.remain_path = path[:]
        self.agent_pos   = path[0]
        self.m_status.set(f"⚡  Dynamic Mode  ·  {self.algo.get().upper()}  ·  "
                          f"{self.heur.get().capitalize()}  ·  Initial cost = {len(path)-1}")
        self.redraw()
        self._dyn_id = self.root.after(MOVE_DELAY, self._dyn_step)

    def _dyn_step(self):
        if self.state != self.S_DYNAMIC: return

        if self.agent_pos == self.goal:
            self.m_status.set(f"🏁  Goal reached!  ·  Nodes = {self.m_nodes.get()}  ·  {self.m_time.get()}")
            self.state = self.S_DONE
            self.root.title("AI 2002 – Pathfinding  [DONE]")
            self.redraw(); return

        protected = {self.agent_pos, self.start, self.goal}
        if len(self.remain_path) > 1: protected.add(self.remain_path[1])

        total_free = sum(1 for r in range(self.rows) for c in range(self.cols)
                         if self.grid[r][c] == EMPTY and (r,c) not in protected)
        spawn_p = DYN_PROB / max(1, total_free * 0.05)

        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in protected and self.grid[r][c] == EMPTY:
                    if random.random() < spawn_p:
                        self.grid[r][c] = WALL
                        self.dyn_added.add((r,c))

        blocked = any(self.grid[r][c] == WALL for (r,c) in self.remain_path[1:])

        if blocked:
            path, vis, ms = self._invoke(from_pos=self.agent_pos)
            self.visited_set = set(vis)
            if path:
                self.remain_path = path[:]; self.path_set = set(path); self.final_path = path[:]
                self.m_status.set(f"🔄  Re-planned  ·  New cost = {len(path)-1}  ·  {ms:.2f} ms")
            else:
                self.m_status.set("❌  Agent is trapped — no route to goal!")
                self.state = self.S_DONE
                self.root.title("AI 2002 – Pathfinding  [BLOCKED]")
                self.redraw(); return

        if len(self.remain_path) > 1:
            self.remain_path.pop(0)
            self.agent_pos = self.remain_path[0]
            self.m_cost.set(str(len(self.final_path)-1))

        self.redraw()
        self._dyn_id = self.root.after(MOVE_DELAY, self._dyn_step)

    # ── Stop / Cancel ─────────────────────────────────────────
    def _cancel(self):
        if self._anim_id: self.root.after_cancel(self._anim_id); self._anim_id = None
        if self._dyn_id:  self.root.after_cancel(self._dyn_id);  self._dyn_id  = None

    def _stop(self):
        self._cancel()
        for (r,c) in self.dyn_added:
            if self.grid[r][c] == WALL: self.grid[r][c] = EMPTY
        self._reset_overlay()
        self.state = self.S_IDLE
        self.root.title("AI 2002 — Dynamic Pathfinding Agent")
        self.m_status.set("Stopped. Edit the grid or press Run Search / Dynamic Mode.")
        self.redraw()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1120x700")
    root.minsize(780, 500)
    app = App(root)
    app._random_maze()
    root.mainloop()
