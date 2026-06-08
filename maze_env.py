"""
4-Room Maze Environment
=======================
Gymnasium-compatible 2-D navigation environment with:
  - Continuous state space  : (x, y)  in [0, 20]^2
  - Continuous action space : Δ(x, y) in [-1, +1]^2
  - Hardcoded 4-room occupancy grid (20 x 20)
  - Potential-based reward shaping

Coordinate convention
---------------------
  x  : horizontal axis, increases left → right
  y  : vertical   axis, increases bottom → top  (math convention)

  Grid indexing: grid[row][col]
    row = floor(y)   → axis 0 of the numpy array
    col = floor(x)   → axis 1 of the numpy array

  Cell (row, col) occupies the continuous region
    x ∈ [col, col+1)  ×  y ∈ [row, row+1)
  and is initialised to its centre: (col + 0.5, row + 0.5)

4-room layout  (W = wall ██, . = free space)
---------------------------------------------
  Each doorway is 3 cells wide (centre ± 1).

  Row 19 ██████████████████████████████████████████
  Row 18 ██  .  .  .  .  .  .  .  .  ██  .  .  .  .  .  .  .  .  ██
   ...                  Room 2 (TL)   ██       Room 4 (TR)
  Row 17 ██  .  .  .  .  .  .  .  .  ██  .  .  .  .  .  .  .  .  ██
  Row 16 ██  .  .  .  .  .  .  .  .  .   .  .  .  .  .  .  .  .  ██  ┐
  Row 15 ██  .  .  .  .  .  .  .  .  .   .  .  .  .  .  .  .  .  ██  ├ doorway (col 10, rows 14-16)
  Row 14 ██  .  .  .  .  .  .  .  .  .   .  .  .  .  .  .  .  .  ██  ┘
  Row 13 ██  .  .  .  .  .  .  .  .  ██  .  .  .  .  .  .  .  .  ██
  Row 11 ██  .  .  .  .  .  .  .  .  ██  .  .  .  .  .  .  .  .  ██
  Row 10 ████████  .  .  .  ████████████████  .  .  .  ████████████  ← horizontal divider
  Row  9 ██  .  .  .  .  .  .  .  .  ██  .  .  .  .  .  .  .  .  ██
   ...        Room 1 (BL)            ██       Room 3 (BR)
  Row  7 ██  .  .  .  .  .  .  .  .  ██  .  .  .  .  .  .  .  .  ██
  Row  6 ██  .  .  .  .  .  .  .  .  .   .  .  .  .  .  .  .  .  ██  ┐
  Row  5 ██  .  .  .  .  .  .  .  .  .   .  .  .  .  .  .  .  .  ██  ├ doorway (col 10, rows 4-6)
  Row  4 ██  .  .  .  .  .  .  .  .  .   .  .  .  .  .  .  .  .  ██  ┘
  Row  3 ██  .  .  .  .  .  .  .  .  ██  .  .  .  .  .  .  .  .  ██
  Row  1 ██  .  .  .  .  .  .  .  .  ██  .  .  .  .  .  .  .  .  ██
  Row  0 ██████████████████████████████████████████████████████████

            Col  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
                                      └──┘        └──────────────┘
                               doorway (row 10,   doorway (row 10,
                                cols 4-6)          cols 14-16)

Connections (all rooms reachable from any room; each doorway 3 cells wide):
  Room 1 (BL) ↔ Room 3 (BR) : col=10, rows  4- 6  (vertical divider doorway)
  Room 2 (TL) ↔ Room 4 (TR) : col=10, rows 14-16  (vertical divider doorway)
  Room 1 (BL) ↔ Room 2 (TL) : row=10, cols  4- 6  (horizontal divider doorway)
  Room 3 (BR) ↔ Room 4 (TR) : row=10, cols 14-16  (horizontal divider doorway)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────────────────────────────────────

class FourRoomMazeEnv(gym.Env):
    """
    Gymnasium environment: 4-room continuous-space maze.

    Parameters
    ----------
    max_steps   : int   Episode horizon (truncation condition).
    d_min       : float Minimum Euclidean distance between sampled start & goal.
    goal_radius : float ε — goal is reached when ||pos − goal|| < goal_radius.
    alpha       : float Scale for potential-based reward shaping term r2.
    render_mode : str   'human' or 'rgb_array' (None disables rendering).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        max_steps: int   = 500,
        d_min: float     = 5.0,
        goal_radius: float = 0.5,
        alpha: float     = 0.02,
        render_mode: str = None,
    ):
        super().__init__()

        self.grid_size   = 20
        self.max_steps   = max_steps
        self.d_min       = d_min
        self.goal_radius = goal_radius
        self.alpha       = alpha
        self.render_mode = render_mode

        # ── Occupancy grid ────────────────────────────────────────────
        # grid[row][col] = 1 → wall   0 → free
        self.grid = self._build_four_room_grid()

        # Precompute free-cell centres once; used in _sample_valid_pair.
        # Cell (row, col) has centre (col+0.5, row+0.5) = (x, y).
        self.free_cells = np.array(
            [[c + 0.5, r + 0.5]
             for r in range(self.grid_size)
             for c in range(self.grid_size)
             if self.grid[r][c] == 0],
            dtype=np.float32,
        )  # shape (N_free, 2)

        # ── Gymnasium spaces ──────────────────────────────────────────
        # Observation: (x/G, y/G, gx/G, gy/G) normalised to [0, 1]^4
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        # Action: continuous displacement Δ(x, y) in [-1, +1]^2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── Episode state (populated in reset()) ─────────────────────
        self.pos:        np.ndarray = None   # agent position  (x, y)
        self.goal:       np.ndarray = None   # goal  position  (x, y)
        self.step_count: int        = 0

        # ── Rendering ─────────────────────────────────────────────────
        self._fig = None
        self._ax  = None

    # ─────────────────────────────────────────────────────────────────
    #  Maze construction
    # ─────────────────────────────────────────────────────────────────

    def _build_four_room_grid(self) -> np.ndarray:
        """
        Construct the 20×20 4-room occupancy grid.

        Returns
        -------
        grid : ndarray, shape (20, 20), dtype int32
            grid[row][col] = 1 (wall) or 0 (free)
        """
        G    = self.grid_size
        grid = np.zeros((G, G), dtype=np.int32)

        # ── Outer boundary walls ──────────────────────────────────────
        grid[0,   :]  = 1   # bottom row  y ∈ [0,  1)
        grid[G-1, :]  = 1   # top    row  y ∈ [19,20)
        grid[:,   0]  = 1   # left   col  x ∈ [0,  1)
        grid[:, G-1]  = 1   # right  col  x ∈ [19,20)

        # ── Vertical divider at col=10 (x ∈ [10,11)) ─────────────────
        #   3-cell doorways centred at row=5 and row=15:
        #     rows 4-6  → Room1 ↔ Room3
        #     rows 14-16 → Room2 ↔ Room4
        grid[:,    G//2]    = 1
        #grid[G//4, G//2]    = 0   # (old single-cell doorway — replaced below)
        #grid[3*G//4, G//2]  = 0   # (old single-cell doorway — replaced below)

        # ── Horizontal divider at row=10 (y ∈ [10,11)) ───────────────
        #   3-cell doorways centred at col=5 and col=15:
        #     cols 4-6  → Room1 ↔ Room2
        #     cols 14-16 → Room3 ↔ Room4
        grid[G//2,    :]    = 1
        #grid[G//2,  G//4]   = 0   # (old single-cell doorway — replaced below)
        #grid[G//2, 3*G//4]  = 0   # (old single-cell doorway — replaced below)
        for delta in [-1, 0, 1]:
            grid[G//4   + delta, G//2] = 0   # rows 4, 5, 6  in col 10
            grid[3*G//4 + delta, G//2] = 0   # rows 14,15,16 in col 10
            grid[G//2, G//4   + delta] = 0   # cols 4, 5, 6  in row 10
            grid[G//2, 3*G//4 + delta] = 0   # cols 14,15,16 in row 10

        return grid

    # ─────────────────────────────────────────────────────────────────
    #  Geometry helpers
    # ─────────────────────────────────────────────────────────────────

    def _cell(self, pos: np.ndarray):
        """Return (row, col) grid indices for a continuous (x, y) position."""
        x, y = pos
        return int(np.floor(y)), int(np.floor(x))

    def _in_bounds(self, pos: np.ndarray) -> bool:
        """True iff pos lies within [0, grid_size)^2."""
        x, y = pos
        G = self.grid_size
        return (0.0 <= x < G) and (0.0 <= y < G)

    def _is_free(self, pos: np.ndarray) -> bool:
        """True iff pos is in-bounds and occupies a free cell (grid value 0)."""
        if not self._in_bounds(pos):
            return False
        row, col = self._cell(pos)
        return self.grid[row][col] == 0

    def _ray_march(
        self,
        start: np.ndarray,
        delta: np.ndarray,
        n_checks: int = 10,
    ):
        """
        Validate the straight-line path from `start` to `start + delta`.

        Samples `n_checks` evenly-spaced intermediate points (inclusive of
        start and end) and tests each with `_is_free`.  If any point is
        blocked the move is rejected and the agent stays at `start`.

        Returns
        -------
        (is_valid, new_pos) : (bool, ndarray)
        """
        end = start + delta
        for t in np.linspace(0.0, 1.0, n_checks):
            pt = start + t * delta
            if not self._is_free(pt):
                return False, start.copy()
        return True, end.copy()

    # ─────────────────────────────────────────────────────────────────
    #  Sampling
    # ─────────────────────────────────────────────────────────────────

    def _sample_valid_pair(self, max_attempts: int = 2000):
        """
        Draw (start, goal) from free cell centres such that
        ||start − goal||₂ ≥ d_min.

        Raises RuntimeError if no valid pair is found within max_attempts.
        """
        n = len(self.free_cells)
        for _ in range(max_attempts):
            i = self.np_random.integers(n)
            j = self.np_random.integers(n)
            if i == j:
                continue
            start = self.free_cells[i]
            goal  = self.free_cells[j]
            if np.linalg.norm(start - goal) >= self.d_min:
                return start.copy(), goal.copy()

        raise RuntimeError(
            f"Could not sample a valid (start, goal) pair with "
            f"d_min={self.d_min} after {max_attempts} attempts. "
            f"Consider reducing d_min."
        )

    # ─────────────────────────────────────────────────────────────────
    #  Observation & reward
    # ─────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Normalise (x, y, gx, gy) to [0, 1]^4."""
        G = float(self.grid_size)
        return np.array(
            [self.pos[0] / G, self.pos[1] / G,
             self.goal[0] / G, self.goal[1] / G],
            dtype=np.float32,
        )

    def _compute_reward(
        self,
        old_pos:    np.ndarray,
        new_pos:    np.ndarray,
        valid_move: bool,
    ):
        """
        Reward = r1 + r2

        r1 — sparse terminal signal
          +1.0  agent reached goal (||new_pos − goal|| < goal_radius)
          −1.0  invalid action (wall hit or out-of-bounds)
           0.0  otherwise

        r2 — potential-based shaping  (Ng et al., 1999)
          α · (||old_pos − goal|| − ||new_pos − goal||)
          Positive when agent moves closer; zero for invalid moves
          (new_pos == old_pos, so distances cancel).
          Guaranteed not to alter the optimal policy.

        Returns
        -------
        (reward, goal_reached) : (float, bool)
        """
        goal_reached = bool(
            np.linalg.norm(new_pos - self.goal) < self.goal_radius
        )

        if not valid_move:
            r1 = -1.0
        elif goal_reached:
            r1 = +1.0
        else:
            r1 = 0.0

        old_dist = np.linalg.norm(old_pos - self.goal)
        new_dist = np.linalg.norm(new_pos - self.goal)
        r2 = self.alpha * (old_dist - new_dist)

        return float(r1 + r2), goal_reached

    # ─────────────────────────────────────────────────────────────────
    #  Gymnasium API
    # ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.goal = self._sample_valid_pair()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        old_pos = self.pos.copy()
        valid_move, new_pos = self._ray_march(self.pos, action)

        # Position only updates on valid moves
        self.pos = new_pos

        reward, goal_reached = self._compute_reward(old_pos, new_pos, valid_move)

        self.step_count += 1
        terminated = goal_reached
        truncated  = (self.step_count >= self.max_steps)

        info = {
            "goal_reached":     goal_reached,
            "valid_move":       valid_move,
            "distance_to_goal": float(np.linalg.norm(self.pos - self.goal)),
            "step_count":       self.step_count,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ─────────────────────────────────────────────────────────────────
    #  Rendering
    # ─────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode is None:
            return

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            plt.tight_layout()

        ax = self._ax
        ax.clear()
        G  = self.grid_size

        # Occupancy grid: 0 (free) → white,  1 (wall) → black
        ax.imshow(
            self.grid,
            cmap="binary",
            origin="lower",
            extent=[0, G, 0, G],
            vmin=0, vmax=1,
        )

        # Goal zone (green circle)
        ax.add_patch(
            plt.Circle(self.goal, self.goal_radius,
                       color="limegreen", alpha=0.85, zorder=3)
        )
        ax.plot(*self.goal, "+", color="darkgreen",
                markersize=8, markeredgewidth=2, zorder=4)

        # Agent position (red dot)
        ax.plot(*self.pos, "o", color="crimson",
                markersize=10, zorder=5, label="Agent")

        # Decorations
        ax.set_xlim(0, G)
        ax.set_ylim(0, G)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, G + 1, 5))
        ax.set_yticks(range(0, G + 1, 5))
        ax.grid(True, color="gray", linewidth=0.3, alpha=0.4)
        ax.set_title(
            f"Step {self.step_count:4d}  |  "
            f"pos ({self.pos[0]:.1f}, {self.pos[1]:.1f})  |  "
            f"goal ({self.goal[0]:.1f}, {self.goal[1]:.1f})  |  "
            f"dist {np.linalg.norm(self.pos - self.goal):.2f}",
            fontsize=9,
        )
        legend_handles = [
            mpatches.Patch(color="limegreen", label="Goal zone"),
            mpatches.Patch(color="crimson",   label="Agent"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

        if self.render_mode == "human":
            plt.pause(0.05)
        elif self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            buf = np.frombuffer(
                self._fig.canvas.tostring_rgb(), dtype=np.uint8
            )
            w, h = self._fig.canvas.get_width_height()
            return buf.reshape(h, w, 3)

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def print_grid(grid: np.ndarray) -> None:
    """Print an ASCII representation of the occupancy grid."""
    G = grid.shape[0]
    print("    " + "".join(f"{c:<2d}" for c in range(G)))
    print("    " + "──" * G)
    for r in range(G - 1, -1, -1):   # top row first in ASCII output
        row_str = "".join("██" if grid[r][c] == 1 else "  " for c in range(G))
        print(f"{r:2d} |{row_str}|")
    print("    " + "──" * G)
    print("    " + "".join(f"{c:<2d}" for c in range(G)))


def save_maze_figure(env: FourRoomMazeEnv, path: str = "maze_layout.png") -> None:
    """
    Save a standalone annotated figure of the maze (no agent or goal).
    Used for visual verification of the layout.
    """
    G    = env.grid_size
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(env.grid, cmap="binary", origin="lower",
              extent=[0, G, 0, G], vmin=0, vmax=1)

    # Room labels
    room_labels = [
        (5.0,  5.0,  "Room 1\n(BL)"),
        (5.0,  15.0, "Room 2\n(TL)"),
        (15.0, 5.0,  "Room 3\n(BR)"),
        (15.0, 15.0, "Room 4\n(TR)"),
    ]
    for x, y, label in room_labels:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=11, color="steelblue", fontweight="bold")

    # Doorway labels — one annotation per doorway, centred on the middle cell
    doorway_labels = [
        (10.5,  5.5, "D1"),   # col=10, rows 4-6  (Room1↔Room3)
        (10.5, 15.5, "D2"),   # col=10, rows 14-16 (Room2↔Room4)
        ( 5.5, 10.5, "D3"),   # row=10, cols 4-6  (Room1↔Room2)
        (15.5, 10.5, "D4"),   # row=10, cols 14-16 (Room3↔Room4)
    ]
    for x, y, label in doorway_labels:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, color="darkorange", fontweight="bold", zorder=4)

    ax.set_xlim(0, G)
    ax.set_ylim(0, G)
    ax.set_aspect("equal")
    ax.set_xticks(range(0, G + 1, 5))
    ax.set_yticks(range(0, G + 1, 5))
    ax.grid(True, color="gray", linewidth=0.4, alpha=0.5)
    ax.set_title("4-Room Maze Layout  (D = doorway)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Maze figure saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks() -> None:
    print("=" * 65)
    print("  4-Room Maze — Sanity Checks  (3-cell doorways)")
    print("=" * 65)

    env = FourRoomMazeEnv(d_min=5.0, render_mode=None)

    # ── [1] Grid statistics ───────────────────────────────────────────
    # Old layout (1-cell doorways): 293 free cells, 107 wall cells
    # New layout (3-cell doorways): 301 free cells,  99 wall cells
    # Gain: 4 doorways × 2 extra cells each = 8 additional free cells
    print("\n[1] Grid statistics")
    print_grid(env.grid)
    n_total = env.grid_size ** 2
    n_wall  = int(env.grid.sum())
    n_free  = n_total - n_wall
    print(f"\n  Total cells : {n_total}")
    print(f"  Wall cells  : {n_wall}   (expected  99)")
    print(f"  Free cells  : {n_free}  (expected 301)")
    print(f"  Precomputed : {len(env.free_cells)}")
    assert n_wall == 99,  f"Expected 99 wall cells, got {n_wall}"
    assert n_free == 301, f"Expected 301 free cells, got {n_free}"
    assert n_free == len(env.free_cells), "Free-cell count mismatch!"
    print("  Grid statistics ✓")

    # ── [2] Doorway verification — all 3 cells of every doorway ───────
    # Each of the 4 doorways is 3 cells wide (centre ± 1).
    # All 12 doorway cells must be free (grid == 0).
    print("\n[2] All 12 doorway cells (should be 0 = free)")
    doorway_cells = [
        # Vertical divider (col=10): lower doorway, rows 4-6
        ("col=10 row=4  (Room1↔Room3, lower edge )", ( 4, 10)),
        ("col=10 row=5  (Room1↔Room3, centre)     ", ( 5, 10)),
        ("col=10 row=6  (Room1↔Room3, upper edge )", ( 6, 10)),
        # Vertical divider (col=10): upper doorway, rows 14-16
        ("col=10 row=14 (Room2↔Room4, lower edge )", (14, 10)),
        ("col=10 row=15 (Room2↔Room4, centre)     ", (15, 10)),
        ("col=10 row=16 (Room2↔Room4, upper edge )", (16, 10)),
        # Horizontal divider (row=10): left doorway, cols 4-6
        ("row=10 col=4  (Room1↔Room2, left edge ) ", (10,  4)),
        ("row=10 col=5  (Room1↔Room2, centre)     ", (10,  5)),
        ("row=10 col=6  (Room1↔Room2, right edge) ", (10,  6)),
        # Horizontal divider (row=10): right doorway, cols 14-16
        ("row=10 col=14 (Room3↔Room4, left edge ) ", (10, 14)),
        ("row=10 col=15 (Room3↔Room4, centre)     ", (10, 15)),
        ("row=10 col=16 (Room3↔Room4, right edge) ", (10, 16)),
    ]
    for label, (r, c) in doorway_cells:
        val    = env.grid[r][c]
        status = "✓ free" if val == 0 else "✗ BLOCKED"
        print(f"  grid[{r:2d}][{c:2d}] = {val}  {status}  ({label})")
        assert val == 0, f"Doorway cell grid[{r}][{c}] is blocked!"
    print("  All 12 doorway cells are free ✓")

    # ── [3] Divider wall integrity — cells adjacent to doorway edges ──
    # Cells immediately outside the 3-cell doorway range must be walls.
    # This verifies the doorways are exactly 3 cells wide, no more.
    print("\n[3] Cells adjacent to doorway edges (should be 1 = wall)")
    wall_checks = [
        # Vertical divider (col=10): sentinels just outside rows 4-6 and 14-16
        ("col=10 row= 3 (wall, just below lower doorway)", ( 3, 10), 1),
        ("col=10 row= 7 (wall, just above lower doorway)", ( 7, 10), 1),
        ("col=10 row=13 (wall, just below upper doorway)", (13, 10), 1),
        ("col=10 row=17 (wall, just above upper doorway)", (17, 10), 1),
        # Horizontal divider (row=10): sentinels just outside cols 4-6 and 14-16
        ("row=10 col= 3 (wall, just left  of left doorway )", (10,  3), 1),
        ("row=10 col= 7 (wall, just right of left doorway )", (10,  7), 1),
        ("row=10 col=13 (wall, just left  of right doorway)", (10, 13), 1),
        ("row=10 col=17 (wall, just right of right doorway)", (10, 17), 1),
        # Divider intersection must remain a wall
        ("col=10 row=10 (intersection of both dividers)    ", (10, 10), 1),
    ]
    for label, (r, c), expected in wall_checks:
        val    = env.grid[r][c]
        status = "✓" if val == expected else "✗ WRONG"
        print(f"  {status}  grid[{r:2d}][{c:2d}] = {val}  (expected {expected})  {label}")
        assert val == expected, f"grid[{r}][{c}] = {val}, expected {expected}"
    print("  All sentinel wall checks passed ✓")

    # ── [4] Reset & observation ────────────────────────────────────────
    print("\n[4] Reset")
    obs, _ = env.reset(seed=42)
    print(f"  Observation  : {obs}")
    print(f"  Agent pos    : {env.pos}")
    print(f"  Goal  pos    : {env.goal}")
    dist = np.linalg.norm(env.pos - env.goal)
    print(f"  Start↔Goal dist : {dist:.2f}  (d_min={env.d_min})")
    assert env._is_free(env.pos),  "Start position is in a wall!"
    assert env._is_free(env.goal), "Goal  position is in a wall!"
    assert dist >= env.d_min,      "Start↔Goal distance violates d_min!"
    assert obs.min() >= 0.0 and obs.max() <= 1.0, "Observation out of [0,1]!"
    print("  All reset assertions passed ✓")

    # ── [5] Step — valid move in open space ───────────────────────────
    print("\n[5] Step — valid move inside Room 1")
    env.pos = np.array([5.5, 5.5], dtype=np.float32)
    action  = np.array([0.5, 0.0], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  action={action}  valid={info['valid_move']}  "
          f"new_pos={env.pos}  reward={reward:+.4f}")
    assert info["valid_move"], "Expected valid move!"
    assert np.allclose(env.pos, [6.0, 5.5]), f"Unexpected position: {env.pos}"
    print("  ✓")

    # ── [6] Step — wall collision (row=7 is wall in col=10) ───────────
    # NOTE: row=6 (y∈[6,7)) is now a doorway — use row=7 (y=7.5) instead.
    print("\n[6] Step — collision with vertical divider (above doorway range)")
    env.pos = np.array([9.5, 7.5], dtype=np.float32)   # row 7 → wall in col=10
    action  = np.array([1.0, 0.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(action)
    print(f"  action={action}  valid={info['valid_move']}  "
          f"pos unchanged={np.allclose(env.pos, [9.5, 7.5])}  reward={reward:+.4f}")
    assert not info["valid_move"],             "Expected invalid move (wall at row 7)!"
    assert np.allclose(env.pos, [9.5, 7.5]),  "Position should not change on invalid move!"
    assert reward < 0,                         "Expected negative reward for wall hit!"
    print("  ✓")

    # ── [7] Step — pass through doorway centre (row=5) ────────────────
    print("\n[7] Step — pass through vertical divider doorway centre (row=5)")
    env.pos = np.array([9.5, 5.5], dtype=np.float32)
    action  = np.array([1.0, 0.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(action)
    print(f"  action={action}  valid={info['valid_move']}  "
          f"new_pos={env.pos}  reward={reward:+.4f}")
    assert info["valid_move"], "Expected valid move through doorway centre!"
    assert env.pos[0] > 10.0, "Agent should have crossed to Room 3 (x > 10)!"
    print("  ✓")

    # ── [8] Step — pass through doorway lower edge (row=4) ────────────
    print("\n[8] Step — pass through vertical divider doorway lower edge (row=4)")
    env.pos = np.array([9.5, 4.5], dtype=np.float32)
    action  = np.array([1.0, 0.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(action)
    print(f"  action={action}  valid={info['valid_move']}  "
          f"new_pos={env.pos}  reward={reward:+.4f}")
    assert info["valid_move"], "Expected valid move through doorway lower edge!"
    assert env.pos[0] > 10.0, "Agent should have crossed to Room 3 (x > 10)!"
    print("  ✓")

    # ── [9] Step — pass through doorway upper edge (row=6) ────────────
    print("\n[9] Step — pass through vertical divider doorway upper edge (row=6)")
    env.pos = np.array([9.5, 6.5], dtype=np.float32)
    action  = np.array([1.0, 0.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(action)
    print(f"  action={action}  valid={info['valid_move']}  "
          f"new_pos={env.pos}  reward={reward:+.4f}")
    assert info["valid_move"], "Expected valid move through doorway upper edge!"
    assert env.pos[0] > 10.0, "Agent should have crossed to Room 3 (x > 10)!"
    print("  ✓")

    # ── [10] Step — pass through horizontal doorway (col=5, row=10) ───
    print("\n[10] Step — pass through horizontal divider doorway (col=5)")
    env.pos = np.array([5.5, 9.5], dtype=np.float32)   # just below horizontal divider
    action  = np.array([0.0, 1.0], dtype=np.float32)   # move up through doorway
    obs, reward, _, _, info = env.step(action)
    print(f"  action={action}  valid={info['valid_move']}  "
          f"new_pos={env.pos}  reward={reward:+.4f}")
    assert info["valid_move"], "Expected valid move through horizontal doorway!"
    assert env.pos[1] > 10.0, "Agent should have crossed to Room 2 (y > 10)!"
    print("  ✓")

    # ── [11] Action clipping ──────────────────────────────────────────
    print("\n[11] Action clipping")
    env.reset(seed=0)
    raw_action = np.array([5.0, -3.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(raw_action)
    print(f"  raw action {raw_action} clipped to [-1,+1]^2 — step executed ✓")

    print("\n" + "=" * 65)
    print("  All sanity checks passed ✓")
    print("=" * 65)


if __name__ == "__main__":
    run_sanity_checks()

    # Save maze layout figure for visual inspection
    env = FourRoomMazeEnv()
    save_maze_figure(env, path="maze_layout.png")
