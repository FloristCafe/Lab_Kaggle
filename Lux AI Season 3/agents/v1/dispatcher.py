"""One-shot rule scoring, global assignment, and one-step path execution."""

import heapq

import numpy as np


MOVES = ((0, -1, 1), (1, 0, 2), (0, 1, 3), (-1, 0, 4))


def shortest_paths(start, blocked, unknown, energy_field, energy_known,
                   enemy_risk, move_cost):
    width, height = blocked.shape
    distance = np.full((width, height), np.inf, dtype=np.float32)
    energy_cost = np.full((width, height), np.inf, dtype=np.float32)
    first_move = np.zeros((width, height), dtype=np.int8)
    parent = np.full((width, height, 2), -1, dtype=np.int16)
    distance[start] = 0
    energy_cost[start] = 0
    queue = [(0.0, start[0], start[1])]
    while queue:
        current, x, y = heapq.heappop(queue)
        if current > distance[x, y] + 1e-5:
            continue
        for dx, dy, action in MOVES:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height) or blocked[nx, ny]:
                continue
            tile_gain = energy_field[nx, ny] if energy_known[nx, ny] else 0
            step_energy = max(0, move_cost - tile_gain)
            edge = 1 + 0.05 * step_energy + 0.55 * unknown[nx, ny] + enemy_risk[nx, ny]
            proposal = current + edge
            if proposal + 1e-5 < distance[nx, ny]:
                distance[nx, ny] = proposal
                energy_cost[nx, ny] = energy_cost[x, y] + step_energy
                first_move[nx, ny] = action if (x, y) == start else first_move[x, y]
                parent[nx, ny] = (x, y)
                heapq.heappush(queue, (proposal, nx, ny))
    return distance, energy_cost, first_move, parent


def exploration_map(unseen, radius):
    width, height = unseen.shape
    integral = np.pad(unseen.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    result = np.zeros((width, height), dtype=np.float32)
    for x in range(width):
        left, right = max(0, x - radius), min(width, x + radius + 1)
        for y in range(height):
            low, high = max(0, y - radius), min(height, y + radius + 1)
            result[x, y] = (integral[right, high] - integral[left, high]
                            - integral[right, low] + integral[left, low])
    return result


def marginal_information(unexplored, probability, confidence,
                         previous_occupied, info_weight):
    base = (info_weight * np.sqrt(unexplored)
            + 0.8 * (1 - confidence) * probability * (1 - probability))
    return np.where(previous_occupied, 0, base)


def assign_targets(utility, valid):
    """Assign one distinct physical map coordinate to each unit."""
    count, tile_count = utility.shape
    if count > tile_count:
        raise ValueError("more units than physical map cells")
    values = np.where(valid, utility, -1e9).astype(np.float64)
    try:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(values, maximize=True)
        assigned = np.empty(count, dtype=np.int32)
        assigned[rows] = columns
        return assigned
    except ImportError:
        edges = sorted(
            ((values[row, col], row, col)
             for row in range(count) for col in range(tile_count)
             if valid[row, col]),
            reverse=True,
        )
        assigned = np.full(count, -1, dtype=np.int32)
        used = set()
        for _, row, col in edges:
            if assigned[row] < 0 and col not in used:
                assigned[row] = col
                used.add(col)
        for row in np.flatnonzero(assigned < 0):
            available = np.array([col for col in range(tile_count) if col not in used])
            col = int(available[np.argmax(values[row, available])])
            assigned[row] = col
            used.add(col)
        return assigned


def path_intervals(parent, start):
    """DFS intervals make path-through-target checks constant time."""
    width, height = parent.shape[:2]
    children = [[] for _ in range(width * height)]
    for x in range(width):
        for y in range(height):
            px, py = parent[x, y]
            if px >= 0:
                children[int(px) * height + int(py)].append(x * height + y)
    entry = np.full(width * height, -1, dtype=np.int32)
    exit_time = np.full(width * height, -1, dtype=np.int32)
    clock = 0
    stack = [(start[0] * height + start[1], False)]
    while stack:
        node, leaving = stack.pop()
        if leaving:
            exit_time[node] = clock
        else:
            entry[node] = clock
            clock += 1
            stack.append((node, True))
            stack.extend((child, False) for child in reversed(children[node]))
    return entry.reshape(width, height), exit_time.reshape(width, height)


def diversify_assignment(utility, valid, assigned, units, first_moves,
                         parents, probability, move_cost):
    """Reprice redundant uncertain goals and first-step path congestion."""
    width, height = probability.shape
    uncertainty = 1 - probability
    intervals = [path_intervals(parent, position)
                 for parent, (_, position, _) in zip(parents, units)]

    def next_position(index, goal_index):
        _, position, energy = units[index]
        goal = divmod(int(goal_index), height)
        direction = int(first_moves[index][goal]) if energy >= move_cost else 0
        dx = (direction == 2) - (direction == 4)
        dy = (direction == 3) - (direction == 1)
        return position[0] + dx, position[1] + dy

    def joint_value(choice):
        total = sum(utility[index, col] for index, col in enumerate(choice))
        for index, col in enumerate(choice):
            x, y = divmod(int(col), height)
            for other in range(index):
                ox, oy = divmod(int(choice[other]), height)
                if max(abs(x - ox), abs(y - oy)) <= 1:
                    total -= 8 * uncertainty[x, y] * uncertainty[ox, oy]
                if next_position(index, col) == next_position(other, choice[other]):
                    total -= 6
                for traveler, goal, barrier in (
                    (index, (x, y), (ox, oy)),
                    (other, (ox, oy), (x, y)),
                ):
                    origin = units[traveler][1]
                    entry, exit_time = intervals[traveler]
                    if (barrier != origin and entry[barrier] >= 0
                            and entry[goal] >= entry[barrier]
                            and exit_time[goal] <= exit_time[barrier]):
                        total -= 3
        return total

    best = assigned.copy()
    best_value = joint_value(best)
    seen = {tuple(int(col) for col in assigned)}
    for _ in range(3):
        adjusted = utility.reshape(len(units), width, height).copy()
        next_positions = [next_position(index, col) for index, col in enumerate(assigned)]
        for index, (_, position, energy) in enumerate(units):
            moves = first_moves[index] if energy >= move_cost else np.zeros((width, height), dtype=np.int8)
            next_x = position[0] + np.where(moves == 2, 1, np.where(moves == 4, -1, 0))
            next_y = position[1] + np.where(moves == 3, 1, np.where(moves == 1, -1, 0))
            entry, exit_time = intervals[index]
            for other, chosen in enumerate(assigned):
                if other == index:
                    continue
                ox, oy = divmod(int(chosen), height)
                low_x, high_x = max(0, ox - 1), min(width, ox + 2)
                low_y, high_y = max(0, oy - 1), min(height, oy + 2)
                adjusted[index, low_x:high_x, low_y:high_y] -= (
                    8 * uncertainty[low_x:high_x, low_y:high_y]
                    * uncertainty[ox, oy]
                )
                adjusted[index] -= 6 * (
                    (next_x == next_positions[other][0])
                    & (next_y == next_positions[other][1])
                )
                if (ox, oy) != position and entry[ox, oy] >= 0:
                    adjusted[index] -= 3 * (
                        (entry >= entry[ox, oy])
                        & (exit_time <= exit_time[ox, oy])
                    )
        proposed = assign_targets(adjusted.reshape(len(units), -1), valid)
        value = joint_value(proposed)
        if value > best_value + 1e-5:
            best, best_value = proposed.copy(), value
        key = tuple(int(col) for col in proposed)
        if key in seen:
            break
        seen.add(key)
        assigned = proposed
    return best


class Dispatcher:
    def __init__(self, width, height, max_units, move_cost, sensor_range,
                 max_steps_in_match=100):
        self.width = width
        self.height = height
        self.max_units = max_units
        self.move_cost = move_cost
        self.sensor_range = sensor_range
        self.max_steps_in_match = max_steps_in_match
        self.previous_targets = {}
        self.fixed_target = None

    def plan(self, step, match_step, units, tile_type, last_seen, energy_field,
             energy_seen, enemy_positions, probability, confidence,
             previous_occupied=None):
        width, height = self.width, self.height
        unit_positions = [position for _, position, _ in units]
        unknown = last_seen < 0
        recent = last_seen >= step - 2
        blocked = (tile_type == 2) & recent
        field_known = (energy_seen >= step - 2) & (energy_seen >= 0)
        enemy_risk = np.zeros((width, height), dtype=np.float32)
        for ex, ey, energy in enemy_positions:
            for x in range(max(0, ex - 3), min(width, ex + 4)):
                for y in range(max(0, ey - 3), min(height, ey + 4)):
                    distance = abs(x - ex) + abs(y - ey)
                    if distance <= 3:
                        enemy_risk[x, y] += (4 - distance) * (0.35 + energy / 400)
        unexplored = exploration_map(unknown, self.sensor_range)
        own_count = np.zeros((width, height), dtype=np.float32)
        for x, y in unit_positions:
            own_count[x, y] += 1

        count = len(units)
        actions = np.zeros((self.max_units, 3), dtype=np.int32)
        targets = np.full((self.max_units, 2), -1, dtype=np.int16)
        reachable = np.zeros((self.max_units, width, height), dtype=bool)
        costs = np.full((self.max_units, width, height), -1, dtype=np.float32)
        energy_costs = np.full((self.max_units, width, height), -1, dtype=np.float32)
        if not count:
            return actions, targets, reachable, costs, energy_costs

        utility = np.full((count, width * height), -1e6, dtype=np.float32)
        valid = np.zeros_like(utility, dtype=bool)
        moves = []
        parents = []
        remaining = max(0, self.max_steps_in_match - match_step)
        info_weight = 1.2 if remaining > 40 else 0.45
        score_weight = min(remaining, 25) * 0.9
        if previous_occupied is None:
            previous_occupied = np.zeros((width, height), dtype=bool)
        info_value = marginal_information(
            unexplored, probability, confidence, previous_occupied, info_weight
        )
        for index, (unit_id, position, energy) in enumerate(units):
            distance, energy_cost, first_move, parent = shortest_paths(
                position, blocked, unknown, energy_field, field_known,
                enemy_risk, self.move_cost,
            )
            moves.append(first_move)
            parents.append(parent)
            feasible = np.isfinite(distance) & ~blocked
            reachable[unit_id] = feasible
            costs[unit_id, feasible] = distance[feasible]
            energy_costs[unit_id, feasible] = energy_cost[feasible]
            energy_deficit = np.maximum(0, energy_cost - max(0, energy))
            charge_value = np.where(field_known, np.maximum(0, energy_field), 0)
            charge_weight = max(0, 120 - energy) / 120
            score = (
                score_weight * probability
                + info_value
                + charge_weight * 2.5 * charge_value
            ) / (1 + 0.35 * distance + 0.12 * energy_deficit)
            score -= 0.7 * enemy_risk
            score -= 0.6 * own_count
            previous = self.previous_targets.get(unit_id)
            if previous is not None:
                score[previous] += 0.8
            score[~feasible] = -1e6
            if index == 0 and self.fixed_target is not None and feasible[self.fixed_target]:
                score[self.fixed_target] = 1e5
            utility[index] = score.reshape(-1)
            valid[index] = feasible.reshape(-1)

        assigned = assign_targets(utility, valid)
        assigned = diversify_assignment(
            utility, valid, assigned, units, moves, parents,
            probability, self.move_cost
        )
        self.previous_targets.clear()
        for index, col in enumerate(assigned):
            unit_id, position, energy = units[index]
            target = divmod(int(col), height)
            targets[unit_id] = target
            self.previous_targets[unit_id] = target
            if energy >= self.move_cost:
                actions[unit_id, 0] = moves[index][target]
        return actions, targets, reachable, costs, energy_costs
