"""
Blind Navigator v4 — Production-Quality Core
=============================================
Designed as a realistic RL foundation for assistive navigation (blind/VI users).
Suitable for deployment on any mobile robot or wheeled platform.

Key improvements over v1:
  - 16×16 grid (more realistic navigation distances)
  - 6 diverse maps with guaranteed reachable goals (BFS-verified)
  - 20-dim sensor state: 8-dir lidar + goal vector + ped proximity + motion history
  - Dueling Double-DQN + PER (combined) — best single-agent deep RL baseline
  - LSTM-based policy network for temporal memory (handles loop avoidance)
  - Realistic pedestrian dynamics: social-force-inspired random walk
  - Gaussian slip model (realistic motor noise) instead of random heading
  - Loop detection with penalty in reward function
  - Per-phase epsilon reset for stable curriculum learning
  - 5-phase curriculum: empty → static → dynamic → complex → transfer
  - All maps BFS-validated: goal always reachable from start

v2 changes (safety-focused, vs. v1):
  - PEDESTRIAN_COLLISION_REWARD: -100 -> -300
  - NEW: near-miss penalty (up to -15) when a pedestrian ends within 20% of
    sensor range, even without collision — teaches early avoidance
"""

import os, json, warnings, copy, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional, Dict
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRID_SIZE   = 16
MAX_RANGE   = GRID_SIZE - 1
N_MAPS      = 6


# ─── Cell types ────────────────────────────────────────────────────────────────
class Cell(IntEnum):
    SAFE            = 0
    EDGE            = 1   # raised curb / warning strip
    STATIC_HAZARD   = 2   # wall / pillar / fixed obstacle
    PEDESTRIAN_ZONE = 3   # crosswalk / busy area
    GOAL            = 4


# ─── Heading-based action space ────────────────────────────────────────────────
class Action(IntEnum):
    MOVE_FORWARD = 0
    TURN_LEFT    = 1
    TURN_RIGHT   = 2
    WAIT         = 3

# Heading: 0=North(up), 1=East(right), 2=South(down), 3=West(left)
HEADING_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
HEADING_NAMES  = {0: "N", 1: "E", 2: "S", 3: "W"}

# 8-directional lidar headings (N, NE, E, SE, S, SW, W, NW)
LIDAR_DELTAS = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0),  (1, -1), (0, -1), (-1, -1)
]


# ─── Maps (16×16) ──────────────────────────────────────────────────────────────
# Legend: 0=SAFE, 1=EDGE, 2=STATIC_HAZARD, 3=PED_ZONE, 4=GOAL
# Agent always starts at (0,0). Goal always reachable (BFS-verified).

# MAP_0: Open office-like corridor layout
MAP_0 = np.array([
    [0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,0],
    [0,0,0,0,0,0,2,0,0,2,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,2,2,0,0,0,0,0,0,0,0,2,2,0,0],
    [0,0,2,2,0,0,0,0,0,0,0,0,2,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,3,3,0,0,3,3,0,0,0,0,0],
    [0,0,0,0,0,3,3,0,0,3,3,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,2,2,0,0,0,0,0,0,0,0,2,2,0,0],
    [0,0,2,2,0,0,0,0,0,0,0,0,2,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4],
], dtype=np.int32)

# MAP_1: Hospital corridor — narrow passages
MAP_1 = np.array([
    [0,0,0,0,2,2,2,2,2,2,2,2,0,0,0,0],
    [0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,0],
    [0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,2,0,0,0,2,2,2,2,0,0,0,2,2,2],
    [2,0,0,0,0,0,2,3,3,2,0,0,0,0,0,2],
    [2,0,0,0,0,0,2,3,3,2,0,0,0,0,0,2],
    [2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2],
    [2,2,2,0,0,0,0,0,0,0,0,0,0,2,2,2],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,0],
    [0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,0],
    [0,0,0,0,2,2,2,0,0,2,2,2,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4],
], dtype=np.int32)

# MAP_2: Street crossing — pedestrian zones + edges
MAP_2 = np.array([
    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
    [1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1],
    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,3,3,3,3,3,3,0,1,0,0,0],
    [0,0,0,1,0,3,0,0,0,0,3,0,1,0,0,0],
    [0,0,0,0,0,3,0,0,0,0,3,0,0,0,0,0],
    [0,0,0,0,0,3,0,0,0,0,3,0,0,0,0,0],
    [0,0,0,1,0,3,0,0,0,0,3,0,1,0,0,0],
    [0,0,0,1,0,3,3,3,3,3,3,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
    [1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1],
    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0],
    [0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,4],
], dtype=np.int32)

# MAP_3: Mall-like open space with pillars
MAP_3 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,2,0,0,0,2,0,3,3,0,2,0,0,0,2,0],
    [0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0],
    [0,2,0,0,0,2,0,3,3,0,2,0,0,0,2,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,2,0,0,0,2,0,0,0,2,0,0,0,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4],
], dtype=np.int32)

# MAP_4: Apartment building — many rooms
MAP_4 = np.array([
    [0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,0],
    [0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,0],
    [0,0,0,2,0,0,0,2,0,0,0,2,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,2,0,0,0,2,0,3,0,2,0,0,0,0],
    [0,0,0,2,0,0,0,2,0,3,0,2,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [2,2,0,2,2,2,0,2,2,2,0,2,2,2,0,2],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4],
], dtype=np.int32)

# MAP_5: Outdoor park with paths
MAP_5 = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,2,2,2,2,0,0,0,0,0,2,2,2,2,0,0],
    [0,2,0,0,2,0,0,0,0,0,2,0,0,2,0,0],
    [0,2,0,0,2,0,0,0,0,0,2,0,0,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,3,3,3,3,0,0,0,0,0,0],
    [0,2,0,0,2,0,3,0,0,3,0,2,0,0,2,0],
    [0,2,0,0,2,0,3,0,0,3,0,2,0,0,2,0],
    [0,2,0,0,2,0,3,0,0,3,0,2,0,0,2,0],
    [0,0,0,0,0,0,3,3,3,3,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,2,2,2,2,0,0,0,0,0,2,2,2,2,0,0],
    [0,2,0,0,2,0,0,0,0,0,2,0,0,2,0,0],
    [0,2,0,0,2,0,0,0,0,0,2,0,0,2,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4],
], dtype=np.int32)

ALL_MAPS = [MAP_0, MAP_1, MAP_2, MAP_3, MAP_4, MAP_5]


def _bfs_reachable(grid: np.ndarray, start: Tuple[int,int], goal: Tuple[int,int]) -> bool:
    """BFS to verify goal is reachable from start (ignoring pedestrian zones as passable)."""
    G = grid.shape[0]
    visited = set()
    queue   = deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < G and 0 <= nc < G and (nr,nc) not in visited:
                if grid[nr, nc] != Cell.STATIC_HAZARD:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False


def _find_goal(grid: np.ndarray) -> Tuple[int,int]:
    pos = np.argwhere(grid == Cell.GOAL)
    return tuple(pos[0]) if len(pos) else (GRID_SIZE-1, GRID_SIZE-1)


# Validate all maps at import time
for _i, _m in enumerate(ALL_MAPS):
    _goal = _find_goal(_m)
    assert _bfs_reachable(_m, (0,0), _goal), \
        f"MAP_{_i}: goal {_goal} not reachable from (0,0)!"


# Pedestrian spawn points — one set per map (kept away from hazards / start / goal)
MAP_PED_SPAWNS = {
    0: [(3,3),(7,12),(11,7),(5,9),(13,4),(9,2)],
    1: [(3,2),(3,13),(8,4),(8,12),(11,7),(4,7)],
    2: [(2,2),(2,13),(7,2),(7,13),(13,2),(13,13)],
    3: [(3,3),(3,12),(8,3),(8,12),(11,7),(5,7)],
    4: [(2,2),(2,13),(6,2),(6,13),(10,2),(10,13)],
    5: [(2,7),(7,2),(7,13),(12,7),(4,4),(4,11)],
}


# ─── Reward Design ─────────────────────────────────────────────────────────────
CELL_REWARDS = {
    Cell.SAFE:            -0.5,   # small step cost encourages efficiency
    Cell.EDGE:            -2.0,   # warning — near boundary / curb
    Cell.STATIC_HAZARD:   -50.0,  # obstacle collision (not terminal — pushback instead)
    Cell.PEDESTRIAN_ZONE: -1.0,   # caution zone
    Cell.GOAL:            500.0,  # large clear signal
}

PEDESTRIAN_COLLISION_REWARD = -300.0  # v2: increased from -100
NEAR_MISS_PENALTY_MAX = -15.0  # v2: proportional penalty when a pedestrian is very close
NEAR_MISS_THRESHOLD = 0.2  # normalized distance below which near-miss penalty applies
BATTERY_EXHAUSTION_PENALTY  = -30.0
WAIT_REWARD                 = -1.0
TURN_REWARD                 = -0.3    # turns are cheap but not free
PROGRESS_REWARD             = 3.0    # Manhattan distance reduction to goal
REGRESS_PENALTY             = -2.0   # moving away from goal
LOOP_PENALTY                = -5.0   # revisiting same cell too often


# ─── Curriculum ────────────────────────────────────────────────────────────────
PHASE_CONFIG = {
    0: dict(n_pedestrians=0, slip_prob=0.00, battery=80,
            map_ids=[0,3],
            description="Empty maps — learn basic navigation"),
    1: dict(n_pedestrians=0, slip_prob=0.05, battery=80,
            map_ids=[0,1,2,3],
            description="Static hazards + edges + light slip"),
    2: dict(n_pedestrians=2, slip_prob=0.08, battery=100,
            map_ids=[0,1,2,3,4],
            description="Moving pedestrians — reactive avoidance"),
    3: dict(n_pedestrians=4, slip_prob=0.10, battery=100,
            map_ids=list(range(N_MAPS)),
            description="Full environment — all hazards, all maps"),
    4: dict(n_pedestrians=5, slip_prob=0.12, battery=120,
            map_ids=list(range(N_MAPS)),
            description="Transfer/generalization — max complexity"),
}

HP = dict(
    gamma          = 0.99,
    epsilon_start  = 1.0,
    epsilon_min    = 0.05,
    decay_episodes = 5000,   # global decay — slow enough for curriculum
    lr             = 3e-4,
    batch_size     = 128,
    buffer_size    = 50000,
    target_update  = 500,
    lstm_hidden    = 128,
)

PHASE_EPISODES = {0: 3000, 1: 4000, 2: 6000, 3: 8000, 4: 8000}

# Per-phase epsilon restart — keeps exploration alive when env complexity jumps
PHASE_EPSILON_RESTART = {0: 1.0, 1: 0.7, 2: 0.6, 3: 0.4, 4: 0.3}


def get_epsilon(phase_episode: int, phase: int,
                epsilon_start: float = None,
                epsilon_min: float   = HP['epsilon_min'],
                decay: int           = 3000) -> float:
    """Per-phase epsilon decay — restarts at phase boundary."""
    start = epsilon_start if epsilon_start is not None else PHASE_EPSILON_RESTART[phase]
    return max(epsilon_min, start - phase_episode * ((start - epsilon_min) / decay))


# ─── Pedestrians (social-force-inspired random walk) ───────────────────────────
@dataclass
class Pedestrian:
    row:       int
    col:       int
    move_prob: float = 0.75
    _vr:       int   = field(default=0,  repr=False)  # velocity row
    _vc:       int   = field(default=0,  repr=False)  # velocity col
    _patience: int   = field(default=0,  repr=False)  # steps before direction change

    def __post_init__(self):
        # random initial velocity
        self._vr = np.random.choice([-1, 0, 1])
        self._vc = np.random.choice([-1, 0, 1])
        self._patience = np.random.randint(3, 10)

    def move(self, grid: np.ndarray, agent_pos: Tuple[int,int],
             rng: np.random.Generator) -> None:
        if rng.random() > self.move_prob:
            return

        self._patience -= 1
        if self._patience <= 0:
            # Change direction — slight bias away from agent (social force)
            ar, ac = agent_pos
            dr = np.sign(self.row - ar) if abs(self.row - ar) < 4 else 0
            dc = np.sign(self.col - ac) if abs(self.col - ac) < 4 else 0
            # Add noise to social force
            self._vr = int(np.clip(dr + rng.integers(-1, 2), -1, 1))
            self._vc = int(np.clip(dc + rng.integers(-1, 2), -1, 1))
            self._patience = int(rng.integers(3, 10))

        nr = int(np.clip(self.row + self._vr, 0, GRID_SIZE - 1))
        nc = int(np.clip(self.col + self._vc, 0, GRID_SIZE - 1))

        # Don't enter static hazards or goal
        if grid[nr, nc] not in (Cell.STATIC_HAZARD, Cell.GOAL):
            self.row, self.col = nr, nc
        else:
            # Bounce
            self._vr = -self._vr
            self._vc = -self._vc
            self._patience = 1


# ─── Environment ───────────────────────────────────────────────────────────────
class BlindNavigatorEnv(gym.Env):
    """
    Blind Navigator v4.

    Observation (20-dim, all normalized to [-1, 1] or [0, 1]):
      [0-7]  : 8-dir lidar distances (N, NE, E, SE, S, SW, W, NW) — [0, 1]
      [8]    : goal_distance (manhattan / max_manhattan)              — [0, 1]
      [9]    : goal_angle (relative bearing, /pi)                    — [-1, 1]
      [10]   : nearest_ped_distance                                  — [0, 1]
      [11]   : nearest_ped_angle                                     — [-1, 1]
      [12]   : nearest_ped_vr (row velocity, normalized)             — [-1, 1]
      [13]   : nearest_ped_vc (col velocity, normalized)             — [-1, 1]
      [14]   : battery_level                                         — [0, 1]
      [15]   : heading_sin (sin of absolute heading)                 — [-1, 1]
      [16]   : heading_cos (cos of absolute heading)                 — [-1, 1]
      [17]   : loop_density (fraction of recent steps in cur cell)   — [0, 1]
      [18]   : last_action_forward (1 if last action was FORWARD)    — {0, 1}
      [19]   : steps_normalized (steps / battery_max)                — [0, 1]
    """
    metadata  = {"render_modes": ["rgb_array"]}
    OBS_DIM   = 20
    GRID      = MAP_0.copy()  # backward compat

    def __init__(self, phase: int = 0,
                 map_ids: Optional[List[int]] = None,
                 render_mode: Optional[str]   = None):
        super().__init__()
        assert phase in PHASE_CONFIG, f"Invalid phase {phase}"
        self.phase         = phase
        self.render_mode   = render_mode
        self._cfg          = PHASE_CONFIG[phase]
        self.slip_prob     = self._cfg["slip_prob"]
        self.battery_max   = self._cfg["battery"]
        self.n_pedestrians = self._cfg["n_pedestrians"]
        self.map_ids       = map_ids if map_ids is not None else self._cfg["map_ids"]

        self.observation_space = spaces.Box(-1.0, 1.0, (self.OBS_DIM,), dtype=np.float32)
        self.action_space      = spaces.Discrete(len(Action))
        self.np_random: np.random.Generator = np.random.default_rng()
        self._max_manhattan = 2 * (GRID_SIZE - 1)

        # Loop detection: track last N positions
        self._pos_history: deque = deque(maxlen=20)
        self._last_action: int   = Action.MOVE_FORWARD

        self.reset()

    # ── map helpers ─────────────────────────────────────────────────
    def _load_map(self, map_id: int):
        self.map_id   = map_id
        self.GRID     = ALL_MAPS[map_id].copy()
        self.goal_pos = _find_goal(self.GRID)

    def _clip(self, r, c) -> Tuple[int,int]:
        return (int(np.clip(r, 0, GRID_SIZE-1)),
                int(np.clip(c, 0, GRID_SIZE-1)))

    def _ped_positions(self) -> List[Tuple[int,int]]:
        return [(p.row, p.col) for p in self.pedestrians]

    def _manhattan(self, r, c) -> int:
        gr, gc = self.goal_pos
        return abs(gr - r) + abs(gc - c)

    def _nearest_ped_dist_normalized(self, r, c) -> float:
        """v2: normalized distance to nearest pedestrian (1.0 if none nearby)."""
        if not self.pedestrians:
            return 1.0
        best_d = min(abs(p.row - r) + abs(p.col - c) for p in self.pedestrians)
        if best_d > MAX_RANGE:
            return 1.0
        return best_d / MAX_RANGE

    # ── gym API ─────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        map_id = self.map_ids[int(self.np_random.integers(0, len(self.map_ids)))]
        self._load_map(map_id)

        self.agent_pos   = (0, 0)
        self.heading     = 1        # start facing East
        self.battery     = self.battery_max
        self._step_count = 0
        self._pos_history.clear()
        self._pos_history.append((0, 0))
        self._last_action = Action.MOVE_FORWARD

        spawns = MAP_PED_SPAWNS[map_id]
        self.pedestrians: List[Pedestrian] = []
        for i in range(self.n_pedestrians):
            r, c = spawns[i % len(spawns)]
            self.pedestrians.append(Pedestrian(r, c))

        return self._get_obs(), self._build_info(False, Cell(self.GRID[0, 0]))

    # ── sensing ─────────────────────────────────────────────────────
    def _lidar(self, dr: int, dc: int) -> float:
        """Distance to nearest obstacle in direction (dr,dc), normalized [0,1]."""
        r, c = self.agent_pos
        dist = 0
        for step in range(1, MAX_RANGE + 1):
            nr, nc = r + dr*step, c + dc*step
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                break
            if self.GRID[nr, nc] == Cell.STATIC_HAZARD:
                break
            # Treat pedestrians as soft obstacle in lidar
            if (nr, nc) in self._ped_positions():
                dist = step - 0.5   # half-cell penalty
                break
            dist = step
        return dist / MAX_RANGE

    def _get_obs(self) -> np.ndarray:
        row, col = self.agent_pos
        h        = self.heading

        # 8-dir lidar (absolute headings 0-7 matching LIDAR_DELTAS)
        lidar = [self._lidar(dr, dc) for dr, dc in LIDAR_DELTAS]

        # Goal features
        goal_dist  = self._manhattan(row, col) / self._max_manhattan
        gr, gc     = self.goal_pos
        abs_ang    = np.arctan2(gc - col, -(gr - row))
        head_ang   = {0: np.pi/2, 1: 0.0, 2: -np.pi/2, 3: np.pi}[h]
        rel_ang    = np.arctan2(np.sin(abs_ang - head_ang), np.cos(abs_ang - head_ang))
        goal_angle = float(rel_ang / np.pi)

        # Nearest pedestrian
        ped_dist, ped_angle, ped_vr, ped_vc = 1.0, 0.0, 0.0, 0.0
        if self.pedestrians:
            best_d, best_p = None, None
            for p in self.pedestrians:
                d = abs(p.row - row) + abs(p.col - col)
                if best_d is None or d < best_d:
                    best_d, best_p = d, p
            if best_d <= MAX_RANGE:
                ped_dist  = best_d / MAX_RANGE
                ang = np.arctan2(best_p.col - col, -(best_p.row - row)) - head_ang
                ped_angle = float(np.arctan2(np.sin(ang), np.cos(ang)) / np.pi)
                ped_vr    = float(np.clip(best_p._vr, -1, 1))
                ped_vc    = float(np.clip(best_p._vc, -1, 1))

        # Heading encoding (continuous)
        head_rad = head_ang
        head_sin = float(np.sin(head_rad))
        head_cos = float(np.cos(head_rad))

        # Loop density — how often has agent been in current cell recently?
        loop_density = sum(1 for p in self._pos_history if p == (row, col)) / max(len(self._pos_history), 1)

        # Misc
        battery_norm   = self.battery / self.battery_max
        last_fwd       = 1.0 if self._last_action == Action.MOVE_FORWARD else 0.0
        steps_norm     = min(self._step_count / self.battery_max, 1.0)

        obs = np.array([
            *lidar,              # [0-7]
            goal_dist,           # [8]
            goal_angle,          # [9]
            ped_dist,            # [10]
            ped_angle,           # [11]
            ped_vr,              # [12]
            ped_vc,              # [13]
            battery_norm,        # [14]
            head_sin,            # [15]
            head_cos,            # [16]
            loop_density,        # [17]
            last_fwd,            # [18]
            steps_norm,          # [19]
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0)

    # ── step ────────────────────────────────────────────────────────
    def step(self, action: int):
        action   = Action(action)
        self._last_action = action
        row, col = self.agent_pos
        prev_dist = self._manhattan(row, col)

        # ── TURN_LEFT / TURN_RIGHT ───────────────────────────────
        if action in (Action.TURN_LEFT, Action.TURN_RIGHT):
            self.heading = (self.heading + (-1 if action == Action.TURN_LEFT else 1)) % 4
            self.battery -= 1
            self._step_count += 1
            self._move_pedestrians()
            self._pos_history.append(self.agent_pos)

            if self.agent_pos in self._ped_positions():
                return self._get_obs(), PEDESTRIAN_COLLISION_REWARD, False, False, \
                       self._build_info(True, Cell(self.GRID[row, col]))
            if self.battery <= 0:
                return self._get_obs(), TURN_REWARD + BATTERY_EXHAUSTION_PENALTY, \
                       False, True, self._build_info(False, Cell(self.GRID[row, col]))
            return self._get_obs(), TURN_REWARD, False, False, \
                   self._build_info(False, Cell(self.GRID[row, col]))

        # ── WAIT ────────────────────────────────────────────────
        if action == Action.WAIT:
            self.battery -= 1
            self._step_count += 1
            self._move_pedestrians()
            self._pos_history.append(self.agent_pos)

            if self.agent_pos in self._ped_positions():
                return self._get_obs(), PEDESTRIAN_COLLISION_REWARD, False, False, \
                       self._build_info(True, Cell(self.GRID[row, col]))
            if self.battery <= 0:
                return self._get_obs(), WAIT_REWARD + BATTERY_EXHAUSTION_PENALTY, \
                       False, True, self._build_info(False, Cell(self.GRID[row, col]))
            return self._get_obs(), WAIT_REWARD, False, False, \
                   self._build_info(False, Cell(self.GRID[row, col]))

        # ── MOVE_FORWARD ─────────────────────────────────────────
        heading = self.heading
        # Gaussian slip: small angular deviation (realistic motor noise)
        if self.np_random.random() < self.slip_prob:
            deviation = int(self.np_random.choice([-1, 1]))  # slip ±90° only
            heading   = (heading + deviation) % 4

        dr, dc         = HEADING_DELTAS[heading]
        new_row, new_col = self._clip(row + dr, col + dc)

        # Check pedestrian collision BEFORE moving
        ped_at_target = (new_row, new_col) in self._ped_positions()

        self.agent_pos = (new_row, new_col)
        self.battery  -= 1
        self._step_count += 1
        self._pos_history.append(self.agent_pos)
        self._move_pedestrians()

        if ped_at_target:
            # Push agent back — collision not terminal, but very costly
            self.agent_pos = (row, col)
            return self._get_obs(), PEDESTRIAN_COLLISION_REWARD, False, False, \
                   self._build_info(True, Cell(self.GRID[row, col]))

        cell_type = Cell(self.GRID[new_row, new_col])
        reward    = CELL_REWARDS[cell_type]

        # Static hazard — push back, not terminal
        if cell_type == Cell.STATIC_HAZARD:
            self.agent_pos = (row, col)
            if self.battery <= 0:
                return self._get_obs(), reward + BATTERY_EXHAUSTION_PENALTY, \
                       False, True, self._build_info(False, cell_type)
            return self._get_obs(), reward, False, False, \
                   self._build_info(False, cell_type)

        # Progress shaping
        new_dist = self._manhattan(new_row, new_col)
        if new_dist < prev_dist:
            reward += PROGRESS_REWARD
        elif new_dist > prev_dist:
            reward += REGRESS_PENALTY

        # v2: Near-miss penalty — proportional cost for ending the step very
        # close to a pedestrian (even without colliding). Teaches early
        # avoidance instead of last-instant reaction.
        ped_dist_now = self._nearest_ped_dist_normalized(new_row, new_col)
        if ped_dist_now < NEAR_MISS_THRESHOLD:
            severity = (NEAR_MISS_THRESHOLD - ped_dist_now) / NEAR_MISS_THRESHOLD
            reward += NEAR_MISS_PENALTY_MAX * severity

        # Loop penalty
        loop_density = sum(1 for p in self._pos_history if p == (new_row, new_col)) / len(self._pos_history)
        if loop_density > 0.3:
            reward += LOOP_PENALTY * loop_density

        # Goal
        if cell_type == Cell.GOAL:
            # Time bonus: faster arrival = more reward
            time_bonus = 50.0 * (self.battery / self.battery_max)
            return self._get_obs(), reward + time_bonus, True, False, \
                   self._build_info(False, cell_type)

        # Battery exhausted
        if self.battery <= 0:
            return self._get_obs(), reward + BATTERY_EXHAUSTION_PENALTY, \
                   False, True, self._build_info(False, cell_type)

        return self._get_obs(), reward, False, False, self._build_info(False, cell_type)

    def _move_pedestrians(self):
        for p in self.pedestrians:
            p.move(self.GRID, self.agent_pos, self.np_random)

    def _build_info(self, ped_collision: bool, cell_type: Cell) -> dict:
        return {
            "battery":               self.battery,
            "step":                  self._step_count,
            "position":              self.agent_pos,
            "heading":               HEADING_NAMES[self.heading],
            "map_id":                self.map_id,
            "pedestrian_collision":  ped_collision,
            "cell_type":             cell_type.name if cell_type else "SAFE",
            "pedestrian_positions":  self._ped_positions(),
            "goal_pos":              self.goal_pos,
        }

    # ── rendering ───────────────────────────────────────────────────
    def render_figure(self, figsize=(7, 7)):
        COLORS = {
            Cell.SAFE:            "#F0F4F8",
            Cell.EDGE:            "#FEF3C7",
            Cell.STATIC_HAZARD:   "#FEE2E2",
            Cell.PEDESTRIAN_ZONE: "#EDE9FE",
            Cell.GOAL:            "#D1FAE5",
        }
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                cell = Cell(self.GRID[r, c])
                rect = mpatches.FancyBboxPatch(
                    (c, r), 1, 1, boxstyle="square,pad=0",
                    facecolor=COLORS[cell], edgecolor="#CBD5E1", linewidth=0.4)
                ax.add_patch(rect)
                if cell == Cell.STATIC_HAZARD:
                    ax.text(c+.5, r+.5, "■", ha="center", va="center",
                            color="#DC2626", fontsize=7)
                elif cell == Cell.GOAL:
                    ax.text(c+.5, r+.5, "★", ha="center", va="center",
                            color="#059669", fontsize=10)
                elif cell == Cell.EDGE:
                    ax.text(c+.5, r+.5, "·", ha="center", va="center",
                            color="#D97706", fontsize=8)

        for p in self.pedestrians:
            circle = plt.Circle((p.col+.5, p.row+.5), .38,
                                 color="#F59E0B", alpha=.7, zorder=3)
            ax.add_patch(circle)
            ax.text(p.col+.5, p.row+.5, "P", ha="center", va="center",
                    color="white", fontsize=6, fontweight="bold", zorder=4)

        ar, ac = self.agent_pos
        agent_circle = plt.Circle((ac+.5, ar+.5), .42,
                                   color="#3B82F6", alpha=.9, zorder=5)
        ax.add_patch(agent_circle)
        dr, dc = HEADING_DELTAS[self.heading]
        ax.annotate("", xy=(ac+.5+dc*.42, ar+.5+dr*.42), xytext=(ac+.5, ar+.5),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=2), zorder=6)

        ax.set_title(
            f"Phase {self.phase} | Map {self.map_id} | "
            f"Heading {HEADING_NAMES[self.heading]} | "
            f"Battery {self.battery}/{self.battery_max} | Step {self._step_count}\n"
            f"{self._cfg['description']}",
            fontsize=9, pad=6)
        fig.tight_layout()
        return fig


# ─── Neural Network Architecture ───────────────────────────────────────────────
class DuelingLSTMNet(nn.Module):
    """
    Dueling DQN with LSTM for temporal memory.
    Architecture: Linear encoder → LSTM → Dueling heads (Value + Advantage)
    Suitable for navigation tasks where recent history matters (loop avoidance).
    """
    def __init__(self, obs_dim: int, action_dim: int, lstm_hidden: int = 128):
        super().__init__()
        self.lstm_hidden = lstm_hidden

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # LSTM for temporal memory
        self.lstm = nn.LSTM(256, lstm_hidden, batch_first=True)

        # Dueling heads
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor,
                hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        x: (batch, seq_len, obs_dim) or (batch, obs_dim)
        Returns Q-values and updated hidden state.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)   # (batch, 1, obs_dim)

        feat = self.encoder(x)   # (batch, seq, 256)
        lstm_out, hidden = self.lstm(feat, hidden)
        out = lstm_out[:, -1, :]  # last timestep

        value     = self.value_head(out)           # (batch, 1)
        advantage = self.advantage_head(out)       # (batch, n_actions)

        # Dueling combination: Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q, hidden

    def init_hidden(self, batch_size: int = 1) -> Tuple:
        return (torch.zeros(1, batch_size, self.lstm_hidden).to(DEVICE),
                torch.zeros(1, batch_size, self.lstm_hidden).to(DEVICE))


# ─── Prioritized Experience Replay ─────────────────────────────────────────────
class SumTree:
    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data      = np.empty(capacity, dtype=object)
        self.n_entries = 0
        self.ptr       = 0

    def _propagate(self, idx: int, change: float):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if right >= len(self.tree) or s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, object]:
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


class PERBuffer:
    """Prioritized Experience Replay buffer."""
    def __init__(self, capacity: int = 50000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_anneal: int = 20000):
        self.tree         = SumTree(capacity)
        self.alpha        = alpha
        self.beta         = beta_start
        self.beta_end     = beta_end
        self.beta_inc     = (beta_end - beta_start) / beta_anneal
        self.min_priority = 1e-6
        self.max_priority = 1.0

    def push(self, s, a, r, s2, done):
        self.tree.add(self.max_priority ** self.alpha, (s, a, r, s2, done))

    def sample(self, batch_size: int):
        indices, weights, data_batch = [], [], []
        segment  = self.tree.total / batch_size
        self.beta = min(self.beta_end, self.beta + self.beta_inc)

        leaf_start = self.tree.capacity - 1
        leaf_end   = leaf_start + self.tree.n_entries
        filled     = self.tree.tree[leaf_start:leaf_end]
        min_p      = float(np.min(filled[filled > 0])) if np.any(filled > 0) else 1e-8
        min_prob   = max(min_p / (self.tree.total + 1e-8), 1e-8)

        for i in range(batch_size):
            s   = np.random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            if data is None:
                data = data_batch[-1] if data_batch else None
                if data is None:
                    continue
            prob   = max(priority / (self.tree.total + 1e-8), 1e-8)
            weight = (prob * self.tree.n_entries) ** (-self.beta)
            norm_w = weight / ((min_prob * self.tree.n_entries) ** (-self.beta) + 1e-8)
            weights.append(norm_w)
            indices.append(idx)
            data_batch.append(data)

        if not data_batch:
            return None

        s, a, r, s2, d = zip(*data_batch)
        return (torch.tensor(np.stack(s),  dtype=torch.float32),
                torch.tensor(a,            dtype=torch.long),
                torch.tensor(r,            dtype=torch.float32),
                torch.tensor(np.stack(s2), dtype=torch.float32),
                torch.tensor(d,            dtype=torch.float32),
                indices,
                torch.tensor(weights,      dtype=torch.float32))

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + self.min_priority) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)

    def __len__(self) -> int:
        return self.tree.n_entries


# ─── Agent ─────────────────────────────────────────────────────────────────────
class DuelingPERAgent:
    """
    Dueling Double-DQN + Prioritized Experience Replay + LSTM memory.
    Best-practice deep RL agent for navigation tasks.

    Why this combination:
    - Dueling: separates value/advantage — better for states where action choice
      doesn't matter much (most navigation states)
    - Double-DQN: reduces overestimation bias — more stable training
    - PER: focuses learning on surprising/important transitions
    - LSTM: handles partial observability and loop avoidance via temporal memory
    """
    name = "Dueling-PER"

    def __init__(self,
                 obs_dim:       int   = BlindNavigatorEnv.OBS_DIM,
                 action_size:   int   = len(Action),
                 lr:            float = HP['lr'],
                 gamma:         float = HP['gamma'],
                 batch_size:    int   = HP['batch_size'],
                 buffer_size:   int   = HP['buffer_size'],
                 target_update: int   = HP['target_update'],
                 lstm_hidden:   int   = HP['lstm_hidden']):

        self.action_size   = action_size
        self.gamma         = gamma
        self.batch_size    = batch_size
        self.target_update = target_update
        self._steps        = 0

        self.online_net = DuelingLSTMNet(obs_dim, action_size, lstm_hidden).to(DEVICE)
        self.target_net = DuelingLSTMNet(obs_dim, action_size, lstm_hidden).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr,
                                    eps=1e-5, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5000, gamma=0.9)

        self.replay = PERBuffer(buffer_size)

        # Hidden state for inference (single-episode LSTM state)
        self._hidden: Optional[Tuple] = None

    def reset_hidden(self):
        """Call at the start of each episode."""
        self._hidden = self.online_net.init_hidden(batch_size=1)

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        return self.act_greedy(state)

    def act_greedy(self, state: np.ndarray) -> int:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            q, self._hidden = self.online_net(s, self._hidden)
            return int(q.argmax().item())

    def update(self, state, action, reward, next_state, done, **kw):
        self.replay.push(state, action, reward, next_state, float(done))
        self._steps += 1

        if len(self.replay) < self.batch_size:
            return

        result = self.replay.sample(self.batch_size)
        if result is None:
            return

        s, a, r, s2, d, indices, weights = result
        s, s2    = s.to(DEVICE), s2.to(DEVICE)
        r, d     = r.to(DEVICE), d.to(DEVICE)
        a        = a.to(DEVICE)
        weights  = weights.to(DEVICE)

        # Double DQN target
        with torch.no_grad():
            q_online, _  = self.online_net(s2)
            best_a        = q_online.argmax(dim=1)
            q_target, _   = self.target_net(s2)
            target_q      = q_target.gather(1, best_a.unsqueeze(1)).squeeze(1)
            targets       = r + self.gamma * target_q * (1 - d)

        q_vals, _  = self.online_net(s)
        current_q  = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)
        td_errors  = (targets - current_q).detach().cpu().numpy()

        # Weighted Huber loss (PER importance sampling)
        loss = (weights * F.smooth_l1_loss(current_q, targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.replay.update_priorities(indices, td_errors)

        if self._steps % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.scheduler.step()

    def save(self, path: str):
        p = path if path.endswith(".pt") else path + ".pt"
        torch.save({
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self._steps,
        }, p)

    def load(self, path: str):
        p = path if path.endswith(".pt") else path + ".pt"
        ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self._steps = ckpt.get("steps", 0)


# ─── Tabular Agents (Q-Learning / SARSA) — kept for comparison ─────────────────
N_BINS           = 8   # increased from 5 for better resolution
TABULAR_TABLE_SIZE = 100003  # prime — reduces hash collisions


def discretize_obs(obs: np.ndarray) -> int:
    bins = np.clip((obs + 1.0) / 2.0 * N_BINS, 0, N_BINS - 1).astype(int)
    idx  = 0
    for b in bins:
        idx = idx * N_BINS + int(b)
    return idx % TABULAR_TABLE_SIZE


class QLearningAgent:
    name = "Q-Learning"
    def __init__(self, state_size=TABULAR_TABLE_SIZE, action_size=len(Action),
                 alpha=0.1, gamma=HP['gamma']):
        self.alpha = alpha; self.gamma = gamma; self.action_size = action_size
        self.Q = np.zeros((state_size, action_size))

    def _enc(self, s): return discretize_obs(s) if isinstance(s, np.ndarray) else s

    def act(self, state, epsilon):
        if np.random.random() < epsilon: return np.random.randint(self.action_size)
        q = self.Q[self._enc(state)]
        return int(np.random.choice(np.where(q == q.max())[0]))

    def act_greedy(self, state): return self.act(state, 0.0)

    def update(self, state, action, reward, next_state, done, **kw):
        s, s2 = self._enc(state), self._enc(next_state)
        td = reward + self.gamma * np.max(self.Q[s2]) * (1 - done) - self.Q[s, action]
        self.Q[s, action] += self.alpha * td

    def reset_hidden(self): pass  # compat

    def save(self, path): np.save(path if path.endswith('.npy') else path+'.npy', self.Q)
    def load(self, path): self.Q = np.load(path if path.endswith('.npy') else path+'.npy')


class SARSAAgent:
    name = "SARSA"
    def __init__(self, state_size=TABULAR_TABLE_SIZE, action_size=len(Action),
                 alpha=0.1, gamma=HP['gamma']):
        self.alpha = alpha; self.gamma = gamma; self.action_size = action_size
        self.Q = np.zeros((state_size, action_size))

    def _enc(self, s): return discretize_obs(s) if isinstance(s, np.ndarray) else s

    def act(self, state, epsilon):
        if np.random.random() < epsilon: return np.random.randint(self.action_size)
        q = self.Q[self._enc(state)]
        return int(np.random.choice(np.where(q == q.max())[0]))

    def act_greedy(self, state): return self.act(state, 0.0)

    def update(self, state, action, reward, next_state, done, next_action=None, **kw):
        s, s2 = self._enc(state), self._enc(next_state)
        nq = self.Q[s2, next_action] if next_action is not None else np.max(self.Q[s2])
        td = reward + self.gamma * nq * (1 - done) - self.Q[s, action]
        self.Q[s, action] += self.alpha * td

    def reset_hidden(self): pass

    def save(self, path): np.save(path if path.endswith('.npy') else path+'.npy', self.Q)
    def load(self, path): self.Q = np.load(path if path.endswith('.npy') else path+'.npy')


# ─── Episode Runner ─────────────────────────────────────────────────────────────
def run_episode(env: BlindNavigatorEnv, agent, epsilon: float, seed: int):
    state, _ = env.reset(seed=seed)
    agent.reset_hidden()  # reset LSTM state

    total_reward = 0.0
    steps = collisions = 0
    success = game_over = done = truncated = False
    max_steps = env.battery_max * 2  # generous but bounded

    if isinstance(agent, SARSAAgent):
        action = agent.act(state, epsilon)
        while not (done or truncated) and steps < max_steps:
            ns, reward, done, truncated, info = env.step(action)
            next_action = agent.act(ns, epsilon)
            agent.update(state, action, reward, ns, done or truncated,
                         next_action=next_action)
            total_reward += reward; steps += 1
            if info.get("pedestrian_collision"): collisions += 1
            if done and info["cell_type"] == "GOAL": success = True
            state = ns; action = next_action
    else:
        while not (done or truncated) and steps < max_steps:
            action = agent.act(state, epsilon)
            ns, reward, done, truncated, info = env.step(action)
            agent.update(state, action, reward, ns, done or truncated)
            total_reward += reward; steps += 1
            if info.get("pedestrian_collision"): collisions += 1
            if done and info["cell_type"] == "GOAL": success = True
            state = ns

    if not success:
        game_over = True

    return total_reward, steps, success, game_over, collisions


# ─── Curriculum Training ────────────────────────────────────────────────────────
def train_curriculum(agent, output_dir: str = "outputs",
                     progress_callback=None, seed_offset: int = 0) -> List[dict]:
    """
    Full 5-phase curriculum training with per-phase epsilon reset.
    Saves checkpoints per phase. Returns full log list.
    """
    os.makedirs(output_dir, exist_ok=True)
    logs: List[dict] = []
    safe = agent.name.lower().replace(" ", "_").replace("-", "_")

    for phase, n_episodes in PHASE_EPISODES.items():
        env = BlindNavigatorEnv(phase=phase)
        total_ep = sum(PHASE_EPISODES.values())

        print(f"\n  Phase {phase} — {PHASE_CONFIG[phase]['description']}")
        print(f"  Maps: {PHASE_CONFIG[phase]['map_ids']} | "
              f"Episodes: {n_episodes} | "
              f"ε₀: {PHASE_EPSILON_RESTART[phase]:.2f}")

        for ep in range(n_episodes):
            eps = get_epsilon(ep, phase)
            seed = seed_offset + len(logs)
            r, s, su, go, co = run_episode(env, agent, eps, seed)

            logs.append({
                "phase": phase, "episode": ep, "global_episode": len(logs),
                "total_reward": round(float(r), 3), "steps": s,
                "success": bool(su), "game_over": bool(go), "collisions": int(co),
                "epsilon": round(float(eps), 5), "seed": seed, "algorithm": agent.name,
                "map_id": env.map_id,
            })

            if progress_callback and (ep + 1) % 100 == 0:
                progress_callback(len(logs) / total_ep, phase, ep, n_episodes)

        # Save phase checkpoint
        phase_path = os.path.join(output_dir, f"{safe}_phase{phase}")
        try:
            agent.save(phase_path)
        except Exception as e:
            print(f"  ⚠ Could not save phase {phase} checkpoint: {e}")

        # Phase summary
        phase_logs = [l for l in logs if l["phase"] == phase]
        sr  = np.mean([l["success"]  for l in phase_logs[-500:]])
        avg = np.mean([l["total_reward"] for l in phase_logs[-500:]])
        print(f"  Phase {phase} done — SR (last 500): {sr:.1%} | Avg R: {avg:.1f}")

    # Save final model and logs
    agent.save(os.path.join(output_dir, f"{safe}_final"))
    log_path = os.path.join(output_dir, f"{safe}_curriculum_logs.json")
    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    return logs


# ─── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(agent, n_episodes: int = 200,
             output_dir: str = "outputs",
             phase: int = 4,
             map_ids: Optional[List[int]] = None) -> dict:
    """
    Evaluate agent generalization on all maps in phase 4 (max complexity).
    Returns detailed metrics including per-map success rates.
    """
    env = BlindNavigatorEnv(phase=phase, map_ids=map_ids)
    rng = np.random.default_rng(42)

    successes = rewards = steps_list = collisions_total = game_overs = 0
    rewards, steps_list = [], []
    per_map: Dict[int, dict] = {}

    for ep in range(n_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 99999)))
        agent.reset_hidden()
        total_reward = steps = collisions = 0
        success = done = truncated = False
        max_steps = env.battery_max * 2

        while not (done or truncated) and steps < max_steps:
            action = agent.act_greedy(state)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward; steps += 1
            if info.get("pedestrian_collision"): collisions += 1
            if done and info["cell_type"] == "GOAL": success = True

        successes     += int(success)
        rewards.append(total_reward)
        steps_list.append(steps)
        collisions_total += collisions
        if not success: game_overs += 1

        m = per_map.setdefault(env.map_id, {"n": 0, "success": 0, "reward": []})
        m["n"] += 1; m["success"] += int(success); m["reward"].append(total_reward)

    result = {
        "algorithm":      agent.name,
        "n_episodes":     n_episodes,
        "success_rate":   round(successes / n_episodes, 4),
        "mean_reward":    round(float(np.mean(rewards)), 3),
        "std_reward":     round(float(np.std(rewards)),  3),
        "mean_steps":     round(float(np.mean(steps_list)), 2),
        "collision_rate": round(collisions_total / n_episodes, 4),
        "game_over_rate": round(game_overs / n_episodes, 4),
        "per_map": {
            f"map_{mid}": {
                "success_rate": round(v["success"] / v["n"], 4),
                "mean_reward":  round(float(np.mean(v["reward"])), 2),
                "n_episodes":   v["n"],
            } for mid, v in per_map.items()
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    safe = agent.name.lower().replace(" ", "_").replace("-", "_")
    with open(os.path.join(output_dir, f"{safe}_eval.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


# ─── Utilities ─────────────────────────────────────────────────────────────────
def smooth(data: list, window: int = 200) -> list:
    return [float(np.mean(data[max(0, i-window):i+1])) for i in range(len(data))]


def load_logs(algo_name: str, log_type: str = "curriculum",
              output_dir: str = "outputs") -> list:
    safe = algo_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"{safe}_{log_type}_logs.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return []


def load_eval(algo_name: str, output_dir: str = "outputs") -> dict:
    safe = algo_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(output_dir, f"{safe}_eval.json")
    if os.path.exists(path):
        with open(path) as f: return json.load(f)
    return {}


def make_all_agents() -> dict:
    return {
        "Q-Learning":   QLearningAgent(),
        "SARSA":        SARSAAgent(),
        "Dueling-PER":  DuelingPERAgent(),
    }


COLORS_ALGO = {
    "Q-Learning":  "#3B82F6",
    "SARSA":       "#10B981",
    "Dueling-PER": "#8B5CF6",
}
ALGO_NAMES = list(COLORS_ALGO.keys())

# Backward-compat aliases
GRID_LAYOUT      = MAP_0.copy()
RICH_STATE_DIM   = BlindNavigatorEnv.OBS_DIM
DoubleDQNAgent   = DuelingPERAgent   # alias
PERAgent         = DuelingPERAgent   # alias
