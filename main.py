import tkinter as tk
from tkinter import font as tkfont
import heapq, math, random, time

ANIM_DELAY = 20
MOVE_DELAY = 150
DYN_PROB   = 0.05

EMPTY = 0; WALL = 1; VISITED = 2; PATH = 3; FRONTIER = 4

C = {
    EMPTY:    "#FFFFFF",
    WALL:     "#2C3E50",
    FRONTIER: "#F1C40F",
    VISITED:  "#3498DB",
    PATH:     "#2ECC71",
    "start":  "#27AE60",
    "goal":   "#E74C3C",
    "agent":  "#8E44AD",
    "line":   "#DDDDDD",
}


def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def run_astar(grid, rows, cols, start, goal, h_fn):
    g = {start: 0}
    parent = {start: None}
    closed = set()
    heap = [(h_fn(start, goal), 0, start)]
    vis = []; ctr = 0
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed: continue
        closed.add(cur); vis.append(cur)
        if cur == goal:
            path, node = [], goal
            while node: path.append(node); node = parent[node]
            path.reverse(); return path, vis, len(closed)
        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
                nb = (nr, nc)
                if nb in closed: continue
                ng = g[cur] + 1
                if ng < g.get(nb, 9999):
                    g[nb] = ng; parent[nb] = cur; ctr += 1
                    heapq.heappush(heap, (ng + h_fn(nb, goal), ctr, nb))
    return [], vis, len(closed)

def run_gbfs(grid, rows, cols, start, goal, h_fn):
    visited = {start: None}
    heap = [(h_fn(start, goal), 0, start)]
    vis = []; expanded = 0; ctr = 0
    while heap:
        _, _, cur = heapq.heappop(heap)
        vis.append(cur); expanded += 1
        if cur == goal:
            path, node = [], goal
            while node: path.append(node); node = visited[node]
            path.reverse(); return path, vis, expanded
        r, c = cur
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != WALL:
                nb = (nr, nc)
                if nb not in visited:
                    visited[nb] = cur; ctr += 1
                    heapq.heappush(heap, (h_fn(nb, goal), ctr, nb))
    return [], vis, expanded


class App:
    def __init__(self, root):
        self.root = root
        root.title("Dynamic Pathfinding Agent - AI 2002 Q6")
        root.resizable(True, True)

        self.rows = 20
        self.cols = 25
        self.cell = 26
        self.grid = [[EMPTY]*self.cols for _ in range(self.rows)]
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)

        self.algo    = tk.StringVar(value="astar")
        self.heur    = tk.StringVar(value="manhattan")
        self.density = tk.IntVar(value=30)
        self.placing = None   # "start" or "goal" when user is placing

        self.visited_set  = set()
        self.path_set     = set()
        self.vis_order    = []
        self.final_path   = []
        self.vis_index    = 0
        self._anim_id     = None

        self.agent_pos   = None
        self.remain_path = []
        self.dyn_added   = set()
        self._dyn_id     = None
        self.running     = False

        self.lbl_nodes = tk.StringVar(value="Nodes: -")
        self.lbl_cost  = tk.StringVar(value="Cost: -")
        self.lbl_time  = tk.StringVar(value="Time: -")
        self.lbl_status= tk.StringVar(value="Left-click: wall  |  Right-click: erase  |  Set Start/Goal then click grid")

        self._build_ui()
        self._gen_maze()

    def _build_ui(self):
        fn  = tkfont.Font(family="Segoe UI", size=9)
        fnb = tkfont.Font(family="Segoe UI", size=9, weight="bold")
        fns = tkfont.Font(family="Segoe UI", size=8)

        # Top controls bar
        top = tk.Frame(self.root, bg="#F0F0F0", pady=4)
        top.pack(fill=tk.X, padx=6)

        def lbl(parent, text):
            tk.Label(parent, text=text, bg="#F0F0F0", fg="#555", font=fns).pack(side=tk.LEFT, padx=(6,2))

        def sep():
            tk.Frame(top, bg="#CCCCCC", width=1).pack(side=tk.LEFT, fill=tk.Y, pady=4, padx=4)

        # Grid size
        lbl(top, "Rows:")
        self.v_rows = tk.IntVar(value=self.rows)
        tk.Spinbox(top, from_=5, to=40, textvariable=self.v_rows, width=3, font=fn).pack(side=tk.LEFT)
        lbl(top, "Cols:")
        self.v_cols = tk.IntVar(value=self.cols)
        tk.Spinbox(top, from_=5, to=55, textvariable=self.v_cols, width=3, font=fn).pack(side=tk.LEFT)
        tk.Button(top, text="Apply", command=self._apply_grid, font=fn, padx=6).pack(side=tk.LEFT, padx=4)

        sep()

        # Density
        lbl(top, "Density:")
        tk.Scale(top, from_=0, to=60, orient=tk.HORIZONTAL, variable=self.density,
                 length=80, showvalue=False, sliderlength=12, bg="#F0F0F0",
                 highlightthickness=0).pack(side=tk.LEFT)
        tk.Label(top, textvariable=self.density, bg="#F0F0F0", font=fns, width=2).pack(side=tk.LEFT)
        tk.Button(top, text="Random Maze", command=self._gen_maze, font=fn, padx=6).pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Clear",       command=self._clear,    font=fn, padx=6).pack(side=tk.LEFT, padx=2)

        sep()

        # Algorithm
        lbl(top, "Algorithm:")
        tk.Radiobutton(top, text="A*",   variable=self.algo, value="astar",  bg="#F0F0F0", font=fn).pack(side=tk.LEFT)
        tk.Radiobutton(top, text="GBFS", variable=self.algo, value="greedy", bg="#F0F0F0", font=fn).pack(side=tk.LEFT)

        sep()

        # Heuristic
        lbl(top, "Heuristic:")
        tk.Radiobutton(top, text="Manhattan", variable=self.heur, value="manhattan", bg="#F0F0F0", font=fn).pack(side=tk.LEFT)
        tk.Radiobutton(top, text="Euclidean", variable=self.heur, value="euclidean", bg="#F0F0F0", font=fn).pack(side=tk.LEFT)

        sep()

        # Placement
        self.btn_start = tk.Button(top, text="Set Start", command=lambda: self._set_placing("start"), font=fn, padx=6)
        self.btn_start.pack(side=tk.LEFT, padx=2)
        self.btn_goal  = tk.Button(top, text="Set Goal",  command=lambda: self._set_placing("goal"),  font=fn, padx=6)
        self.btn_goal.pack(side=tk.LEFT, padx=2)

        sep()

        # Run buttons
        tk.Button(top, text="Run Search",   command=self._run_search,  font=fnb, padx=6, fg="#27AE60").pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Dynamic Mode", command=self._run_dynamic, font=fnb, padx=6, fg="#E67E22").pack(side=tk.LEFT, padx=2)
        tk.Button(top, text="Stop",         command=self._stop,        font=fn,  padx=6).pack(side=tk.LEFT, padx=2)

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="white", highlightthickness=1, highlightbackground="#CCCCCC")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.canvas.bind("<Button-1>",        self._click)
        self.canvas.bind("<B1-Motion>",       self._drag)
        self.canvas.bind("<Button-3>",        self._rclick)
        self.canvas.bind("<B3-Motion>",       self._rdrag)
        self.canvas.bind("<Configure>",       lambda e: self.redraw())

        # Bottom status + metrics
        bot = tk.Frame(self.root, bg="#F8F8F8")
        bot.pack(fill=tk.X, padx=6, pady=(0,4))
        tk.Label(bot, textvariable=self.lbl_status, bg="#F8F8F8", fg="#555", font=fns, anchor="w").pack(side=tk.LEFT, padx=6)
        for var in [self.lbl_time, self.lbl_cost, self.lbl_nodes]:
            tk.Label(bot, textvariable=var, bg="#F8F8F8", fg="#333",
                     font=tkfont.Font(family="Consolas", size=9, weight="bold")).pack(side=tk.RIGHT, padx=10)

    def _set_placing(self, mode):
        self.placing = mode
        self.btn_start.config(relief=tk.SUNKEN if mode == "start" else tk.RAISED)
        self.btn_goal.config( relief=tk.SUNKEN if mode == "goal"  else tk.RAISED)
        self.lbl_status.set(f"Click on the grid to place the {'Start' if mode == 'start' else 'Goal'} node.")

    def _apply_grid(self):
        self._stop()
        self.rows  = max(5, min(40, self.v_rows.get()))
        self.cols  = max(5, min(55, self.v_cols.get()))
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)
        self.grid  = [[EMPTY]*self.cols for _ in range(self.rows)]
        self._gen_maze()

    def _gen_maze(self):
        d = self.density.get() / 100.0
        for r in range(self.rows):
            for c in range(self.cols):
                self.grid[r][c] = WALL if random.random() < d and (r,c) not in (self.start, self.goal) else EMPTY
        self._reset(); self.redraw()

    def _clear(self):
        self._stop()
        self.grid = [[EMPTY]*self.cols for _ in range(self.rows)]
        self._reset(); self.redraw()

    def _reset(self):
        self.visited_set = set(); self.path_set = set()
        self.vis_order = []; self.final_path = []; self.vis_index = 0
        self.dyn_added = set(); self.agent_pos = None; self.remain_path = []
        self.lbl_nodes.set("Nodes: -"); self.lbl_cost.set("Cost: -"); self.lbl_time.set("Time: -")

    def _offsets(self):
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        self.cell = max(8, min(40, cw // self.cols, ch // self.rows))
        cs = self.cell
        return max(0,(cw - cs*self.cols)//2), max(0,(ch - cs*self.rows)//2)

    def _cell_at(self, x, y):
        ox, oy = self._offsets(); cs = self.cell
        c = (x - ox) // cs; r = (y - oy) // cs
        if 0 <= r < self.rows and 0 <= c < self.cols: return int(r), int(c)
        return None

    def _click(self, event):
        if self.running: return
        cell = self._cell_at(event.x, event.y)
        if not cell: return
        r, c = cell
        if self.placing == "start":
            self.grid[self.start[0]][self.start[1]] = EMPTY
            self.start = (r, c); self.grid[r][c] = EMPTY
            self.placing = None
            self.btn_start.config(relief=tk.RAISED)
            self.btn_goal.config(relief=tk.RAISED)
            self.lbl_status.set("Start placed. Left-click: wall  |  Right-click: erase")
        elif self.placing == "goal":
            self.grid[self.goal[0]][self.goal[1]] = EMPTY
            self.goal = (r, c); self.grid[r][c] = EMPTY
            self.placing = None
            self.btn_start.config(relief=tk.RAISED)
            self.btn_goal.config(relief=tk.RAISED)
            self.lbl_status.set("Goal placed. Left-click: wall  |  Right-click: erase")
        elif (r, c) not in (self.start, self.goal):
            self.grid[r][c] = WALL
        self.redraw()

    def _drag(self, event):
        if self.running or self.placing: return
        cell = self._cell_at(event.x, event.y)
        if cell and cell not in (self.start, self.goal):
            self.grid[cell[0]][cell[1]] = WALL; self.redraw()

    def _rclick(self, event):
        if self.running: return
        cell = self._cell_at(event.x, event.y)
        if cell and cell not in (self.start, self.goal):
            self.grid[cell[0]][cell[1]] = EMPTY; self.redraw()

    def _rdrag(self, event):
        self._rclick(event)

    def redraw(self):
        ox, oy = self._offsets()
        cs = self.cell
        cv = self.canvas; cv.delete("all")

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = ox + c*cs; y1 = oy + r*cs
                x2 = x1 + cs;   y2 = y1 + cs
                cell = (r, c)

                if self.grid[r][c] == WALL:
                    cv.create_rectangle(x1, y1, x2, y2, fill=C[WALL], outline=C[WALL])
                    continue

                if   cell in self.path_set:    fill = C[PATH]
                elif cell in self.visited_set: fill = C[VISITED]
                else:                          fill = C[EMPTY]

                cv.create_rectangle(x1, y1, x2, y2, fill=fill, outline=C["line"], width=1)

                if cell == self.start:
                    cv.create_rectangle(x1+2, y1+2, x2-2, y2-2, fill=C["start"], outline="")
                    if cs >= 16: cv.create_text(x1+cs//2, y1+cs//2, text="S", fill="white",
                                                font=("Segoe UI", max(7, cs//3), "bold"))
                elif cell == self.goal:
                    cv.create_rectangle(x1+2, y1+2, x2-2, y2-2, fill=C["goal"], outline="")
                    if cs >= 16: cv.create_text(x1+cs//2, y1+cs//2, text="G", fill="white",
                                                font=("Segoe UI", max(7, cs//3), "bold"))

                if self.agent_pos == cell:
                    cx = x1+cs//2; cy = y1+cs//2; rad = max(3, cs//2-3)
                    cv.create_oval(cx-rad, cy-rad, cx+rad, cy+rad, fill=C["agent"], outline="white", width=1)

    def _invoke(self, src=None):
        h_fn = manhattan if self.heur.get() == "manhattan" else euclidean
        src  = src or self.start
        t0   = time.perf_counter()
        if self.algo.get() == "astar":
            path, vis, n = run_astar(self.grid, self.rows, self.cols, src, self.goal, h_fn)
        else:
            path, vis, n = run_gbfs( self.grid, self.rows, self.cols, src, self.goal, h_fn)
        ms = (time.perf_counter() - t0) * 1000
        self.lbl_nodes.set(f"Nodes: {n}")
        self.lbl_cost.set(f"Cost: {len(path)-1}" if path else "Cost: -")
        self.lbl_time.set(f"Time: {ms:.1f}ms")
        return path, vis, ms

    def _run_search(self):
        self._stop(); self._reset()
        self.running = True
        path, vis, ms = self._invoke()
        self.vis_order = vis; self.final_path = path; self.vis_index = 0
        if not path:
            self.lbl_status.set("No path found. Clear some walls and try again.")
            self.running = False; return
        self.lbl_status.set(f"Path found | Cost={len(path)-1} | Nodes={self.lbl_nodes.get().split()[1]} | {ms:.1f}ms")
        self._anim()

    def _anim(self):
        if not self.running: return
        for _ in range(4):
            if self.vis_index < len(self.vis_order):
                self.visited_set.add(self.vis_order[self.vis_index]); self.vis_index += 1
            else:
                for cell in self.final_path: self.path_set.add(cell)
                self.running = False; self.redraw(); return
        self.redraw()
        self._anim_id = self.root.after(ANIM_DELAY, self._anim)

    def _run_dynamic(self):
        self._stop(); self._reset()
        self.running = True
        path, vis, ms = self._invoke()
        if not path:
            self.lbl_status.set("No initial path found. Clear some walls first.")
            self.running = False; return
        self.visited_set = set(vis); self.path_set = set(path)
        self.final_path  = path[:]; self.remain_path = path[:]
        self.agent_pos   = path[0]
        self.lbl_status.set(f"Dynamic Mode | {self.algo.get().upper()} | {self.heur.get().capitalize()} | Cost={len(path)-1}")
        self.redraw()
        self._dyn_id = self.root.after(MOVE_DELAY, self._dyn_step)

    def _dyn_step(self):
        if not self.running: return

        if self.agent_pos == self.goal:
            self.lbl_status.set(f"Goal reached! | {self.lbl_nodes.get()} | {self.lbl_time.get()}")
            self.running = False; self.redraw(); return

        protected = {self.agent_pos, self.start, self.goal}
        if len(self.remain_path) > 1: protected.add(self.remain_path[1])

        free = sum(1 for r in range(self.rows) for c in range(self.cols)
                   if self.grid[r][c] == EMPTY and (r,c) not in protected)
        sp = DYN_PROB / max(1, free * 0.05)
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in protected and self.grid[r][c] == EMPTY:
                    if random.random() < sp:
                        self.grid[r][c] = WALL; self.dyn_added.add((r,c))

        if any(self.grid[r][c] == WALL for r,c in self.remain_path[1:]):
            path, vis, ms = self._invoke(src=self.agent_pos)
            self.visited_set = set(vis)
            if path:
                self.remain_path = path[:]; self.path_set = set(path); self.final_path = path[:]
                self.lbl_status.set(f"Re-planned | New cost={len(path)-1} | {ms:.1f}ms")
            else:
                self.lbl_status.set("Agent trapped - no route to goal.")
                self.running = False; self.redraw(); return

        if len(self.remain_path) > 1:
            self.remain_path.pop(0); self.agent_pos = self.remain_path[0]
            self.lbl_cost.set(f"Cost: {len(self.final_path)-1}")

        self.redraw()
        self._dyn_id = self.root.after(MOVE_DELAY, self._dyn_step)

    def _stop(self):
        self.running = False
        if self._anim_id: self.root.after_cancel(self._anim_id); self._anim_id = None
        if self._dyn_id:  self.root.after_cancel(self._dyn_id);  self._dyn_id  = None
        for r,c in self.dyn_added:
            if self.grid[r][c] == WALL: self.grid[r][c] = EMPTY
        self._reset()
        self.lbl_status.set("Stopped. Left-click: wall  |  Right-click: erase")
        self.redraw()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1050x650")
    root.minsize(700, 450)
    App(root)
    root.mainloop()
