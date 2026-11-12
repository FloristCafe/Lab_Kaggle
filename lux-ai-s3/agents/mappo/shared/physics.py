"""Causal physical channels. Estimates never remove legal goals."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

SIZE = 24
CELLS = SIZE * SIZE
GRID = np.indices((SIZE, SIZE)).reshape(2, CELLS).T
DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
NEIGHBORS = GRID[:, None] + DIRECTIONS
EDGE_SRC, edge_direction = np.where(((NEIGHBORS >= 0) & (NEIGHBORS < SIZE)).all(axis=2))
EDGE_DST = NEIGHBORS[EDGE_SRC, edge_direction] @ np.array([SIZE, 1])
SPEEDS = tuple(np.float32(x) for x in (-.15, -.1, -.05, -.025, .025, .05, .1, .15))


def canonical(xy, side):
    """180 degree coordinate conversion, with no assumption about terrain symmetry."""
    return np.asarray(xy) if side == 0 else SIZE - 1 - np.asarray(xy)


def world_index(index, side):
    return index if side == 0 else CELLS - 1 - index


def relic_mirror(xy):
    x, y = xy
    return SIZE - 1 - y, SIZE - 1 - x


@lru_cache(maxsize=8192)
def drift_offset(speed, observation_step):
    """Offset after transitions from the initial observation to observation_step."""
    magnitude = np.abs(np.float32(speed))
    t = np.arange(-1, observation_step, dtype=np.float32)
    phase = np.remainder(t * magnitude, np.float32(1))
    return int(np.sum(phase[:-1] > phase[1:])) * (1 if speed > 0 else -1)


def terrain_symmetry(tile, offset):
    """Reflect in the generation transpose axis transported by the diagonal roll."""
    original = np.roll(tile, (-offset, offset), axis=(0, 1))
    transposed = original.T
    conflict = (original >= 0) & (transposed >= 0) & (original != transposed)
    if conflict.any():
        return None
    filled = np.where(original >= 0, original, transposed)
    return np.roll(filled, (offset, -offset), axis=(0, 1))


class TerrainBelief:
    def __init__(self):
        self.grids = {speed: np.full((SIZE, SIZE), -1, np.int8) for speed in SPEEDS}
        self.observed = None
        self.step = None
        self.conflicts = 0

    def observe(self, step, observed):
        if self.step is not None and step != self.step + 1:
            raise ValueError("non-contiguous terrain observation")
        self.step = step
        self.observed = observed.copy()
        candidates = {}
        for speed, grid in self.grids.items():
            offset = drift_offset(speed, step)
            last = drift_offset(speed, max(0, step - 1))
            predicted = np.roll(grid, (offset - last, last - offset), axis=(0, 1))
            if ((observed >= 0) & (predicted >= 0) & (observed != predicted)).any():
                continue
            merged = np.where(observed >= 0, observed, predicted)
            reflected = terrain_symmetry(merged, offset)
            if reflected is not None:
                candidates[speed] = reflected
        if self.grids and not candidates:
            self.conflicts += 1
        self.grids = candidates

    def forecast(self, horizon):
        if not self.grids:
            return self.observed if horizon == 0 else np.full_like(self.observed, -1)
        worlds = []
        for speed, grid in self.grids.items():
            shift = drift_offset(speed, self.step + horizon) - drift_offset(speed, self.step)
            worlds.append(np.roll(grid, (shift, -shift), axis=(0, 1)))
        stacked = np.stack(worlds)
        return np.where((stacked == stacked[0]).all(axis=0), stacked[0], -1)


@dataclass
class EnergyField:
    value: np.ndarray
    age: np.ndarray

    @classmethod
    def public(cls, reconstructor, t):
        seen = reconstructor.energy_seen
        return cls(np.where(seen >= 0, reconstructor.energy_field / 20, 0),
                   np.where(seen >= 0, np.minimum(t - seen, 100) / 100, 1))


def route_distances(blocked, xy):
    """Minimum steps in a frozen candidate map, ignoring interaction and replenishment."""
    allowed = ~blocked.reshape(CELLS)
    keep = allowed[EDGE_DST]  # allow units to exit a blocked starting tile
    graph = csr_matrix((np.ones(int(keep.sum())), (EDGE_SRC[keep], EDGE_DST[keep])),
                       shape=(CELLS, CELLS))
    return shortest_path(graph, directed=True, indices=xy @ np.array([SIZE, 1]), unweighted=True)


def reachability(terrain, units, alive, move_cost, enemy):
    result = np.zeros((16, 5, CELLS), dtype=np.float16)
    if not alive.any():
        return result
    xy = np.rint(units[alive, :2] * 23).astype(np.int64)
    optimistic = route_distances(terrain == 2, xy)
    known = route_distances((terrain == 2) | (terrain == -1), xy)
    clip = lambda d: np.minimum(d, 100) / 100
    threat = np.zeros(CELLS)
    for ex, ey in np.argwhere(enemy > 0):
        d = np.abs(GRID - [ex, ey]).sum(axis=1)
        threat = np.maximum(threat, np.maximum(0, 1 - d / 7))
    result[alive] = np.stack((
        clip(optimistic),
        clip(known),
        np.broadcast_to(threat, optimistic.shape),
        np.minimum(optimistic * move_cost, 400) / 400,
        np.maximum(0, np.minimum(optimistic * move_cost, 400) / 400 - units[alive, 2:3]),
    ), axis=1)
    return result


class PhysicsState:
    """Persistent public state across five matches and an optional episodic grid."""
    def __init__(self, side, config):
        self.side = side
        self.config = config
        self.terrain = TerrainBelief()
        self.completed = np.zeros((5, 4, SIZE, SIZE), np.float16)
        self.match_index = 0
        self.visits = np.zeros((SIZE, SIZE), np.float32)
        self.last_spatial = None

    def observe(self, obs, frame, reconstructor):
        step = int(frame["step"])
        sensor = frame["sensor"]
        observed = np.where(sensor, np.asarray(obs["map_features"]["tile_type"]), -1)
        self.terrain.observe(step, observed)
        new_match = min(4, step // (self.config["max_steps_in_match"] + 1))
        if new_match != self.match_index:
            # Store only observations obtained before this match boundary.
            self.completed[self.match_index] = self.last_spatial
            self.visits.fill(0)
            self.match_index = new_match
        current = frame["base"]
        boundary = step > 0 and frame["match_step"] == 0
        if not boundary:
            self.visits += current[18]
        probability = current[11]
        support = np.zeros_like(probability)
        for xy in reconstructor.known_relics.values():
            for x, y in (xy, relic_mirror(xy)):
                support[max(0, x-2):x+3, max(0, y-2):y+3] = 1
        energy = EnergyField.public(reconstructor, step)
        terrain = self.terrain.forecast(0)
        terrain3 = self.terrain.forecast(3)
        terrain8 = self.terrain.forecast(8)
        self.last_spatial = np.stack((probability, support, np.minimum(self.visits / 20, 1),
                                      current[12])).astype(np.float16)
        units = np.zeros_like(current[13:17]) if boundary else current[13:17]
        grid = np.stack((sensor, current[1], current[2], terrain == -1, terrain == 1, terrain == 2,
                         terrain3 == 2, terrain3 >= 0, terrain8 == 2, terrain8 >= 0,
                         energy.value, energy.age, probability, current[12], support,
                         *units))
        self.spatial = (grid if self.side == 0 else grid[:, ::-1, ::-1]).astype(np.float16)
        self.canonical_terrain = terrain if self.side == 0 else terrain[::-1, ::-1]

    def critic_view(self, env):
        manager = env.manager
        units = manager.vectors(env.frame)
        alive = env.frame["alive"]
        if self.side == 1:
            units[alive, :2] = 1 - units[alive, :2]
            units[alive & manager.valid, 8:10] = 1 - units[alive & manager.valid, 8:10]
        score = np.asarray(env.obs["team_points"], dtype=np.float32)[[self.side, 1-self.side]] / 1000
        wins = np.asarray(env.obs["team_wins"], dtype=np.float32)[[self.side, 1-self.side]] / 5
        phase = np.asarray([env.steps / 505, env.frame["match_step"] / 100], dtype=np.float32)
        return {"grid": self.spatial, "units": units.astype(np.float16),
                "alive": alive.copy(), "team": np.concatenate((score, wins, phase)).astype(np.float16)}

    def policy_view(self, env, memory=False):
        state = self.critic_view(env)
        state["reach"] = reachability(self.canonical_terrain, state["units"], state["alive"],
                                      self.config["unit_move_cost"], self.spatial[17])
        if memory:
            snapshots = self.completed.copy()
            snapshots[self.match_index] = self.last_spatial
            if self.side == 1:
                snapshots = snapshots[..., ::-1, ::-1].copy()
            state["memory"] = snapshots
            state["memory_count"] = np.array(self.match_index + 1, np.int64)
        return state
