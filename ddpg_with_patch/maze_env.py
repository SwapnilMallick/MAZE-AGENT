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

  Row 19 ███████████████████████████████████████████████████████████
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
    patch_size  : int   Side length of the local occupancy patch (must be odd).
                        Default 11 ensures a doorway is visible from any point
                        inside a room (rooms are 9 cells wide; half-patch = 5).
    render_mode : str   'human' or 'rgb_array' (None disables rendering).

    Observation  (shape: 4 + patch_size² = 125 for patch_size=11)
    ---------------------------------------------------------------
    [0]   x  / G          agent x, normalised
    [1]   y  / G          agent y, normalised
    [2]   gx / G          goal  x, normalised
    [3]   gy / G          goal  y, normalised
    [4:]  patch.flatten() patch_size×patch_size occupancy window centred on
                          agent's cell; 0=free, 1=wall; out-of-bounds → 1.

    Reward
    ------
    +1.0  goal reached  (||pos − goal|| < goal_radius)
    −0.5  invalid move  (wall hit or out-of-bounds)
     0.0  otherwise
    No distance-based shaping: the local patch makes wall geometry directly
    observable, so the agent does not need an extrinsic navigation signal.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        max_steps:   int   = 500,
        d_min:       float = 5.0,
        goal_radius: float = 0.5,
        patch_size:  int   = 11,
        render_mode: str   = None,
    ):
        super().__init__()

        assert patch_size % 2 == 1, "patch_size must be odd"

        self.grid_size   = 20
        self.max_steps   = max_steps
        self.d_min       = d_min
        self.goal_radius = goal_radius
        self.patch_size  = patch_size
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

        # Padded grid for patch extraction (border = 1 / wall).
        # Computed once and reused every call to _get_local_patch.
        _half = patch_size // 2
        self._padded = np.ones(
            (self.grid_size + 2 * _half, self.grid_size + 2 * _half),
            dtype=np.float32,
        )
        self._padded[_half : _half + self.grid_size,
                     _half : _half + self.grid_size] = self.grid
        self._patch_half = _half

        # ── Gymnasium spaces ──────────────────────────────────────────
        # Observation: 4 scalars + patch_size² binary values ∈ [0, 1]
        obs_dim = 4 + patch_size ** 2          # 125 for patch_size=11
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # Action: continuous displacement Δ(x, y) in [-1, +1]^2
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── Episode state (populated in reset()) ─────────────────────
        self.pos:        np.ndarray = None   # agent position  (x, y)
        self.goal:       np.ndarray = None   # goal  position  (x, y)
        self.start_pos:  np.ndarray = None   # start position  (x, y)
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

    def _get_local_patch(self) -> np.ndarray:
        """
        Extract a patch_size × patch_size occupancy window centred on the
        agent's current cell, flattened to shape (patch_size²,).

        Uses the precomputed padded grid (self._padded) so out-of-bounds
        cells are read as 1 (wall) without branching.

        Layout: patch[i][j] = cell at (row + i − half, col + j − half)
          → patch[half][half] is always the agent's own cell (value 0,
            since the agent can only occupy free cells).
        """
        row, col = self._cell(self.pos)
        h  = self._patch_half          # 5 for 11×11
        pr = row + h                   # agent row in padded grid
        pc = col + h                   # agent col in padded grid
        return self._padded[pr - h : pr + h + 1,
                            pc - h : pc + h + 1].flatten()

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
        """
        Return normalised observation of shape (4 + patch_size²,).

        First 4 values: (x/G, y/G, gx/G, gy/G) ∈ [0, 1]
        Remaining 121:  flattened 11×11 local occupancy patch ∈ {0, 1}
        """
        G = float(self.grid_size)
        scalars = np.array(
            [self.pos[0] / G, self.pos[1] / G,
             self.goal[0] / G, self.goal[1] / G],
            dtype=np.float32,
        )
        return np.concatenate([scalars, self._get_local_patch()])

    def _compute_reward(
        self,
        new_pos:    np.ndarray,
        valid_move: bool,
    ):
        """
        Sparse reward — no distance shaping.

        +1.0  goal reached  (||new_pos − goal|| < goal_radius)
        −0.5  invalid move  (wall hit or out-of-bounds)
         0.0  otherwise

        Rationale for removing Euclidean shaping (r2):
          r2 = α·(d_old − d_new) rewards straight-line progress, which
          points through walls near doorways — the opposite of what the agent
          needs.  The 11×11 local patch makes wall geometry directly
          observable, so the network can learn wall-aware navigation without
          an extrinsic navigation signal.

        Rationale for −0.5 (not −1.0) wall penalty:
          With only +1 at the goal and nothing elsewhere, a −1.0 wall penalty
          creates a heavily asymmetric signal (the agent sees thousands of −1
          events before a single +1), pushing it toward cautious idling rather
          than goal-directed exploration.  −0.5 preserves the deterrent while
          keeping the signal closer to balanced.

        Returns
        -------
        (reward, goal_reached) : (float, bool)
        """
        goal_reached = bool(
            np.linalg.norm(new_pos - self.goal) < self.goal_radius
        )

        if not valid_move:
            reward = -0.5
        elif goal_reached:
            reward = +1.0
        else:
            reward = 0.0

        return float(reward), goal_reached

    # ─────────────────────────────────────────────────────────────────
    #  Gymnasium API
    # ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos, self.goal = self._sample_valid_pair()
        self.start_pos = self.pos.copy()
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        old_pos = self.pos.copy()
        valid_move, new_pos = self._ray_march(self.pos, action)

        # Position only updates on valid moves
        self.pos = new_pos

        reward, goal_reached = self._compute_reward(new_pos, valid_move)

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

        # Start position (blue star) — shown throughout the episode
        if self.start_pos is not None:
            ax.plot(*self.start_pos, "*", color="royalblue",
                    markersize=14, zorder=4)

        # Agent position (red dot)
        ax.plot(*self.pos, "o", color="crimson",
                markersize=10, zorder=5)

        # Decorations
        ax.set_xlim(0, G)
        ax.set_ylim(0, G)
        ax.set_aspect("equal")
        ax.set_xticks(range(0, G + 1, 5))
        ax.set_yticks(range(0, G + 1, 5))
        ax.grid(True, color="gray", linewidth=0.3, alpha=0.4)
        ax.set_title(
            f"Step {self.step_count:4d}  |  "
            f"start ({self.start_pos[0]:.1f},{self.start_pos[1]:.1f})  |  "
            f"pos ({self.pos[0]:.1f},{self.pos[1]:.1f})  |  "
            f"goal ({self.goal[0]:.1f},{self.goal[1]:.1f})  |  "
            f"dist {np.linalg.norm(self.pos - self.goal):.2f}",
            fontsize=8,
        )
        legend_handles = [
            mpatches.Patch(color="limegreen", label="Goal"),
            mpatches.Patch(color="royalblue", label="Start"),
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
            # On HiDPI/Retina displays the physical pixel count is larger
            # than the logical size returned by get_width_height().
            n_pix = len(buf) // 3
            if n_pix != w * h:
                scale = int(round((n_pix / (w * h)) ** 0.5))
                w, h = w * scale, h * scale
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
    print("  4-Room Maze — Sanity Checks  (3-cell doorways, patch obs)")
    print("=" * 65)

    env  = FourRoomMazeEnv(d_min=5.0, render_mode=None)
    G    = env.grid_size
    PS   = env.patch_size          # 11
    HALF = env.patch_size // 2     # 5
    OBS_DIM = 4 + PS ** 2          # 125

    # ── [1] Grid statistics ───────────────────────────────────────────
    print("\n[1] Grid statistics")
    print_grid(env.grid)
    n_total = G ** 2
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

    # ── [2] All 12 doorway cells free ────────────────────────────────
    print("\n[2] All 12 doorway cells (should be 0 = free)")
    doorway_cells = [
        ("col=10 row=4  (Room1↔Room3 lower edge)", ( 4, 10)),
        ("col=10 row=5  (Room1↔Room3 centre)    ", ( 5, 10)),
        ("col=10 row=6  (Room1↔Room3 upper edge)", ( 6, 10)),
        ("col=10 row=14 (Room2↔Room4 lower edge)", (14, 10)),
        ("col=10 row=15 (Room2↔Room4 centre)    ", (15, 10)),
        ("col=10 row=16 (Room2↔Room4 upper edge)", (16, 10)),
        ("row=10 col=4  (Room1↔Room2 left edge) ", (10,  4)),
        ("row=10 col=5  (Room1↔Room2 centre)    ", (10,  5)),
        ("row=10 col=6  (Room1↔Room2 right edge)", (10,  6)),
        ("row=10 col=14 (Room3↔Room4 left edge) ", (10, 14)),
        ("row=10 col=15 (Room3↔Room4 centre)    ", (10, 15)),
        ("row=10 col=16 (Room3↔Room4 right edge)", (10, 16)),
    ]
    for label, (r, c) in doorway_cells:
        val = env.grid[r][c]
        status = "✓ free" if val == 0 else "✗ BLOCKED"
        print(f"  grid[{r:2d}][{c:2d}] = {val}  {status}  ({label})")
        assert val == 0, f"Doorway cell grid[{r}][{c}] is blocked!"
    print("  All 12 doorway cells are free ✓")

    # ── [3] Sentinel wall checks ──────────────────────────────────────
    print("\n[3] Cells adjacent to doorway edges (should be 1 = wall)")
    wall_checks = [
        ("col=10 row= 3 (just below lower vert. doorway)", ( 3, 10), 1),
        ("col=10 row= 7 (just above lower vert. doorway)", ( 7, 10), 1),
        ("col=10 row=13 (just below upper vert. doorway)", (13, 10), 1),
        ("col=10 row=17 (just above upper vert. doorway)", (17, 10), 1),
        ("row=10 col= 3 (just left  of left  horiz. dw) ", (10,  3), 1),
        ("row=10 col= 7 (just right of left  horiz. dw) ", (10,  7), 1),
        ("row=10 col=13 (just left  of right horiz. dw) ", (10, 13), 1),
        ("row=10 col=17 (just right of right horiz. dw) ", (10, 17), 1),
        ("col=10 row=10 (divider intersection)           ", (10, 10), 1),
    ]
    for label, (r, c), expected in wall_checks:
        val = env.grid[r][c]
        status = "✓" if val == expected else "✗ WRONG"
        print(f"  {status}  grid[{r:2d}][{c:2d}] = {val}  (expected {expected})  {label}")
        assert val == expected, f"grid[{r}][{c}] = {val}, expected {expected}"
    print("  All sentinel wall checks passed ✓")

    # ── [4] Reset & observation ───────────────────────────────────────
    print(f"\n[4] Reset  (obs_dim = {OBS_DIM})")
    obs, _ = env.reset(seed=42)
    print(f"  Observation shape : {obs.shape}  (expected ({OBS_DIM},))")
    print(f"  Agent pos  : {env.pos}")
    print(f"  Goal  pos  : {env.goal}")
    dist = np.linalg.norm(env.pos - env.goal)
    print(f"  Start↔Goal dist : {dist:.2f}  (d_min={env.d_min})")
    assert obs.shape == (OBS_DIM,),               f"Obs shape {obs.shape} != ({OBS_DIM},)"
    assert env._is_free(env.pos),                 "Start position is in a wall!"
    assert env._is_free(env.goal),                "Goal  position is in a wall!"
    assert dist >= env.d_min,                     "Start↔Goal distance violates d_min!"
    assert obs.min() >= 0.0 and obs.max() <= 1.0, "Observation out of [0, 1]!"
    print("  All reset assertions passed ✓")

    # ── [4b] Local patch content ──────────────────────────────────────
    print(f"\n[4b] Local patch content  (patch_size={PS}, half={HALF})")

    env.pos = np.array([5.5, 5.5], dtype=np.float32)
    patch   = env._get_local_patch()
    assert patch.shape == (PS ** 2,), f"Patch shape {patch.shape} unexpected"
    assert np.all((patch == 0.0) | (patch == 1.0)), "Patch has non-binary values"
    print(f"  Shape ({patch.shape[0]},) = ({PS}x{PS},), values binary ✓")

    p2d = patch.reshape(PS, PS)
    assert p2d[HALF][HALF] == 0.0, "Centre cell should be free"
    print(f"  Centre  p2d[{HALF}][{HALF}]  = {p2d[HALF][HALF]:.0f}  (agent cell, free) ✓")

    # Left outer wall (col=0) is HALF=5 cols left of agent col=5
    assert p2d[HALF][0] == 1.0, "Expected left outer wall in patch"
    print(f"  L-wall  p2d[{HALF}][  0]  = {p2d[HALF][0]:.0f}  (outer wall col=0) ✓")

    # Vertical divider (col=10) is HALF=5 cols right → rightmost patch col
    # grid[5][10] = 0 (doorway), so this should be free
    assert p2d[HALF][PS - 1] == 0.0, "Expected doorway at right edge of patch"
    print(f"  Doorway p2d[{HALF}][{PS-1:2d}]  = {p2d[HALF][PS-1]:.0f}  (doorway col=10, row=5) ✓")

    # Out-of-bounds padding test
    env.pos = np.array([1.5, 1.5], dtype=np.float32)   # row=1, col=1
    p2d_oob = env._get_local_patch().reshape(PS, PS)
    assert p2d_oob[HALF][HALF] == 0.0, "Agent cell should be free"
    assert p2d_oob[0][0]       == 1.0, "OOB corner should be padded wall"
    print(f"  OOB     p2d_oob[0][0]  = {p2d_oob[0][0]:.0f}  (padded wall) ✓")
    print("  Patch content checks passed ✓")

    # ── [5] Valid non-goal move: reward = 0.0 ────────────────────────
    print("\n[5] Step — valid move inside Room 1  (reward = 0.0)")
    env.reset(seed=42)
    env.pos = np.array([5.5, 5.5], dtype=np.float32)
    action  = np.array([0.5, 0.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(action)
    print(f"  action={action}  valid={info['valid_move']}  new_pos={env.pos}  reward={reward:+.4f}")
    assert info["valid_move"],               "Expected valid move!"
    assert np.allclose(env.pos, [6.0, 5.5]), f"Unexpected position: {env.pos}"
    assert reward == 0.0,                    f"Expected 0.0, got {reward}"
    assert obs.shape == (OBS_DIM,),          "Post-step obs shape wrong"
    print("  ✓")

    # ── [6] Wall collision: reward = −0.5 ────────────────────────────
    print("\n[6] Step — wall collision  (reward = −0.5)")
    env.pos = np.array([9.5, 7.5], dtype=np.float32)
    action  = np.array([1.0, 0.0], dtype=np.float32)
    obs, reward, _, _, info = env.step(action)
    print(f"  action={action}  valid={info['valid_move']}  "
          f"pos unchanged={np.allclose(env.pos,[9.5,7.5])}  reward={reward:+.4f}")
    assert not info["valid_move"],            "Expected invalid move!"
    assert np.allclose(env.pos, [9.5, 7.5]), "Position should not change!"
    assert reward == -0.5,                   f"Expected -0.5, got {reward}"
    print("  ✓")

    # ── [7–10] Doorway crossings: all valid, reward = 0.0 ────────────
    doorway_steps = [
        ("[7] Vertical doorway centre   (row=5)",
         [9.5, 5.5], [1.0, 0.0], lambda p: p[0] > 10.0),
        ("[8] Vertical doorway lower    (row=4)",
         [9.5, 4.5], [1.0, 0.0], lambda p: p[0] > 10.0),
        ("[9] Vertical doorway upper    (row=6)",
         [9.5, 6.5], [1.0, 0.0], lambda p: p[0] > 10.0),
        ("[10] Horizontal doorway       (col=5)",
         [5.5, 9.5], [0.0, 1.0], lambda p: p[1] > 10.0),
    ]
    for label, start, act_v, cross_check in doorway_steps:
        print(f"\n{label}")
        env.pos = np.array(start, dtype=np.float32)
        action  = np.array(act_v, dtype=np.float32)
        obs, reward, _, _, info = env.step(action)
        print(f"  valid={info['valid_move']}  new_pos={env.pos}  reward={reward:+.4f}")
        assert info["valid_move"],    f"{label}: expected valid move"
        assert cross_check(env.pos), f"{label}: agent did not cross divider"
        assert reward == 0.0,        f"{label}: expected 0.0, got {reward}"
        print("  ✓")

    # ── [11] Action clipping ──────────────────────────────────────────
    print("\n[11] Action clipping")
    env.reset(seed=0)
    obs, _, _, _, _ = env.step(np.array([5.0, -3.0], dtype=np.float32))
    assert obs.shape == (OBS_DIM,), f"Post-step obs shape wrong: {obs.shape}"
    print(f"  raw [5., -3.] clipped to [-1,+1]^2, obs.shape={obs.shape} ✓")

    print("\n" + "=" * 65)
    print("  All sanity checks passed ✓")
    print("=" * 65)


if __name__ == "__main__":
    run_sanity_checks()

    # Save maze layout figure for visual inspection
    env = FourRoomMazeEnv()
    save_maze_figure(env, path="maze_layout.png")
