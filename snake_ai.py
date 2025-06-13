# snake_ai.py

from typing import List, Tuple, Dict, Optional, Deque, Set, NamedTuple
from collections import deque
import heapq
from functools import lru_cache

# Directions
UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"

class Position(NamedTuple):
    x: int
    y: int

class SmartSnakeAI:
    def __init__(self, grid_width: int, grid_height: int):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.directions = [UP, DOWN, LEFT, RIGHT]
        self.opposites = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
        
        # Dynamic cache size based on grid dimensions
        self.cache_size = max(1024, grid_width * grid_height * 4)
        
        # State tracking
        self.moves_since_food = 0
        self.last_snake_length = 0
        
        # Dynamic patience threshold
        self.base_patience = self.grid_width * self.grid_height * 0.3
        self.max_patience = self.grid_width * self.grid_height * 0.7
        
        # Initialize Hamiltonian cycle
        self._init_hamiltonian_cycle()
        self.recent_history_length = 5

    def _init_hamiltonian_cycle(self):
        """Initialize the Hamiltonian cycle for the grid"""
        self.hamiltonian_cycle = []
        pos_to_index = {}
        
        # Create a basic Hamiltonian cycle that snakes through the grid
        for y in range(self.grid_height):
            row = range(self.grid_width) if y % 2 == 0 else range(self.grid_width - 1, -1, -1)
            for x in row:
                pos = Position(x, y)
                pos_to_index[pos] = len(self.hamiltonian_cycle)
                self.hamiltonian_cycle.append(pos)
        
        self.hamiltonian_indices = pos_to_index

    def _get_valid_moves(self, current_dir: Optional[str]) -> List[str]:
        if not current_dir:
            return self.directions
        return [d for d in self.directions if d != self.opposites[current_dir]]

    @property
    def _move_position_cache_size(self):
        return self.cache_size
    
    @lru_cache(maxsize=None)  # Will use instance property
    def _move_position(self, pos: Position, direction: str) -> Position:
        x, y = pos
        moves = {
            UP: Position(x, y - 1),
            DOWN: Position(x, y + 1),
            LEFT: Position(x - 1, y),
            RIGHT: Position(x + 1, y)
        }
        return moves.get(direction, pos)

    def _is_on_grid(self, pos: Position) -> bool:
        return 0 <= pos.x < self.grid_width and 0 <= pos.y < self.grid_height

    def _is_safe(self, pos: Position, snake_body: Tuple[Position, ...]) -> bool:
        return self._is_on_grid(pos) and pos not in snake_body[:-1]

    @lru_cache(maxsize=1024)
    def _manhattan_distance(self, a: Position, b: Position) -> int:
        return abs(a.x - b.x) + abs(a.y - b.y)

    def _evaluate_future_state(self, pos: Position, snake_body: Tuple[Position, ...], 
                             food_pos: Position, depth: int = 3) -> float:
        """Evaluate future states recursively up to a certain depth"""
        if depth == 0:
            return 0.0
        
        score = 0.0
        next_snake_body = (pos,) + snake_body[:-1]
        
        # Check space available
        space, avg_freedom = self._flood_fill(pos, next_snake_body)
        # Modulate space score by average freedom
        # avg_freedom of 1.5 is neutral. Higher is better, lower is worse.
        # Max avg_freedom is 4. Min for a path is 1 (or 0 if completely boxed).
        freedom_modifier = 1.0 + (avg_freedom - 1.5) * 0.25 # Factor 0.25 is tunable
        effective_space_score = space * freedom_modifier

        score += effective_space_score * (1.0 + depth * 0.5) # Weight space more heavily in early moves
        
        # Evaluate food distance
        dist_to_food = self._manhattan_distance(pos, food_pos)
        score -= dist_to_food * (2.0 - depth * 0.5)  # Food distance matters less in later moves
        
        # Check if move follows Hamiltonian cycle
        if self._follows_hamiltonian(pos, snake_body[0]):
            score += 50 * (depth * 0.5)
        
        # Recursive evaluation of next possible moves
        max_future_score = -float('inf')
        for direction in self.directions:
            next_pos = self._move_position(pos, direction)
            if self._is_safe(next_pos, next_snake_body):
                future_score = self._evaluate_future_state(
                    next_pos, next_snake_body, food_pos, depth - 1
                )
                max_future_score = max(max_future_score, future_score)
        
        return score + (max_future_score * 0.5 if max_future_score > -float('inf') else 0)

    def _follows_hamiltonian(self, pos: Position, prev_pos: Position) -> bool:
        """Check if move follows Hamiltonian cycle direction"""
        try:
            curr_idx = self.hamiltonian_indices[pos]
            prev_idx = self.hamiltonian_indices[prev_pos]
            return (curr_idx == (prev_idx + 1) % len(self.hamiltonian_cycle) or
                   (prev_idx == len(self.hamiltonian_cycle) - 1 and curr_idx == 0))
        except KeyError:
            return False

    def _find_path(self, start: Position, end: Position, 
                   snake_body: Tuple[Position, ...]) -> Optional[List[Position]]:
        obstacles = set(snake_body[1:])
        if end in obstacles:
            obstacles.remove(end)

        open_set = [(self._manhattan_distance(start, end), start)]
        came_from: Dict[Position, Position] = {}
        cost_so_far: Dict[Position, int] = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                temp = end
                while temp in came_from:
                    path.append(temp)
                    temp = came_from[temp]
                return path[::-1]

            for direction in self.directions:
                neighbor = self._move_position(current, direction)
                
                if not self._is_on_grid(neighbor) or neighbor in obstacles:
                    continue

                new_cost = cost_so_far[current] + 1
                    
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._manhattan_distance(neighbor, end)
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current
        
        return None

    def _flood_fill(self, start_pos: Position, snake_body: Tuple[Position, ...]) -> Tuple[int, float]:
        if not self._is_on_grid(start_pos) or start_pos in snake_body:
            return 0, 0.0
        
        q: Deque[Position] = deque([start_pos])
        visited = {start_pos}
        count = 0
        obstacles = set(snake_body)
        total_freedom_score = 0.0

        while q:
            pos = q.popleft()
            count += 1
            current_cell_freedom = 0
            for direction in self.directions:
                next_pos = self._move_position(pos, direction)
                if self._is_on_grid(next_pos) and next_pos not in obstacles:
                    current_cell_freedom += 1
                    if next_pos not in visited:
                        visited.add(next_pos)
                        q.append(next_pos)
            total_freedom_score += current_cell_freedom
        
        average_freedom = (total_freedom_score / count) if count > 0 else 0.0
        return count, average_freedom

    def _get_direction_from_path(self, head: Position, next_pos: Position) -> str:
        dx = next_pos.x - head.x
        dy = next_pos.y - head.y
        if dx == 1: return RIGHT
        if dx == -1: return LEFT
        if dy == 1: return DOWN
        if dy == -1: return UP
        raise ValueError("Positions are not adjacent")

    def _is_path_a_trap(self, path: List[Position], snake_body: Tuple[Position, ...]) -> bool:
        if not path: # Should not happen if path_to_food is valid
            return False # Or True, depending on desired behavior for empty path

        # Simulate the snake's body after moving along the entire path and eating food.
        # The new head will be the end of the path (food position).
        # The snake's length increases by 1.

        # Construct the full new body: path segments in order of new body, then previous body segments.
        # Example: path = [p1, p2, food_pos], snake_body = [s_head, s2, s_tail]
        # new_body_ordered_segments = [food_pos, p2, p1, s_head, s2] (s_tail is dropped)

        temp_new_body = list(reversed(path)) + list(snake_body)
        simulated_full_snake_body = tuple(temp_new_body[:len(snake_body) + 1])

        new_head = simulated_full_snake_body[0] # This is path[-1] (food position)
        new_tail = simulated_full_snake_body[-1] # This is the new tail of the longer snake

        # If the snake is very short (e.g., head and tail are the same or adjacent after growth),
        # it's not really a "trap" in the sense of not being able to reach its tail.
        if len(simulated_full_snake_body) <= 2: # e.g. snake becomes head-tail or just head.
            return False

        # The obstacles for pathfinding are the *inner* segments of this new simulated body.
        # The _find_path method takes a 'snake_body' argument and considers body[1:] as obstacles,
        # removing 'end' (the target, i.e. new_tail) from the obstacles if it's there.
        # This setup works correctly if we pass simulated_full_snake_body.
        path_to_new_tail = self._find_path(new_head, new_tail, simulated_full_snake_body)

        return path_to_new_tail is None

    def _is_dangerous_constriction(self, pos_to_check: Position, entry_direction: str, snake_body_if_moved_to_pos: Tuple[Position, ...]) -> bool:
        # pos_to_check: The cell we're evaluating if it's a constriction.
        # entry_direction: The direction string (UP, DOWN, LEFT, RIGHT) used to enter pos_to_check.
        # snake_body_if_moved_to_pos: The snake's body, assuming its head is now at pos_to_check.

        safe_continuing_moves = 0
        # Iterate through all possible directions the snake could move FROM pos_to_check
        for potential_next_direction in self.directions:
            # Prevent the snake from immediately reversing its direction of entry
            if potential_next_direction == self.opposites[entry_direction]:
                continue

            potential_next_pos = self._move_position(pos_to_check, potential_next_direction)

            # A move is safe if it's on the grid and doesn't collide with its own body
            # (excluding the head segment, which is pos_to_check itself).
            # snake_body_if_moved_to_pos[0] is pos_to_check.
            # So, obstacles are snake_body_if_moved_to_pos[1:].
            if self._is_on_grid(potential_next_pos) and \
               potential_next_pos not in snake_body_if_moved_to_pos[1:]:
                safe_continuing_moves += 1

        # If there's 1 or 0 ways to continue safely, it's a dangerous constriction.
        return safe_continuing_moves <= 1

    def get_best_move(self, snake_body: List[Tuple[int, int]], food_pos: Tuple[int, int],
                      current_direction: Optional[str]) -> str:
        """
        Enhanced version with lookahead, Hamiltonian cycle awareness, and dynamic patience.
        """
        # Convert to Position objects for better caching
        snake_pos = tuple(Position(x, y) for x, y in snake_body)
        food = Position(*food_pos)
        head = snake_pos[0]
        
        # Update snake state
        if len(snake_body) > self.last_snake_length:
            self.moves_since_food = 0
        else:
            self.moves_since_food += 1
        self.last_snake_length = len(snake_body)
        
        # Dynamic patience threshold based on snake length
        snake_ratio = len(snake_body) / (self.grid_width * self.grid_height)
        self.patience_threshold = self.base_patience + (self.max_patience - self.base_patience) * snake_ratio
        
        # Get valid moves
        possible_moves = self._get_valid_moves(current_direction)
        safe_moves = []
        for move in possible_moves:
            next_pos = self._move_position(head, move)
            if self._is_safe(next_pos, snake_pos):
                safe_moves.append((move, next_pos))
        
        if not safe_moves:
            return possible_moves[0] if possible_moves else UP

        # Find path to food
        golden_path = None
        path_to_food = self._find_path(head, food, snake_pos)
        if path_to_food:
            is_trap = self._is_path_a_trap(path_to_food, snake_pos)
            if not is_trap or \
               (is_trap and self.moves_since_food > self.patience_threshold * 0.5) or \
               self.moves_since_food > self.patience_threshold:
                golden_path = path_to_food

        # Score moves with lookahead
        best_move = None
        best_score = -float('inf')

        for move, pos in safe_moves:
            score = self._evaluate_future_state(pos, snake_pos, food)

            # Additional scoring factors
            if golden_path and pos == golden_path[0]:
                score += 10000
            
            if self._follows_hamiltonian(pos, head): # head is snake_pos[0]
                current_hamiltonian_bonus = 50 * snake_ratio

                # Contextual Safety: Check for immediate "breathing room" if following Hamiltonian cycle.
                # This applies more strongly if the snake is long and the cycle path might be the only space left.
                if len(snake_pos) > self.grid_width * self.grid_height * 0.4: # Threshold: snake fills >40% of grid
                    try:
                        pos_idx = self.hamiltonian_indices[pos]
                        # The position snake just came from is 'head' (snake_pos[0])
                        # The position on cycle after 'pos' is 'next_on_cycle'
                        next_on_cycle = self.hamiltonian_cycle[(pos_idx + 1) % len(self.hamiltonian_cycle)]

                        escape_routes = 0
                        for direction in self.directions:
                            neighbor = self._move_position(pos, direction)

                            # Don't count moving back to 'head' as an escape
                            if neighbor == head:
                                continue
                            # Don't count moving along the cycle to 'next_on_cycle' as an escape
                            if neighbor == next_on_cycle:
                                continue

                            # An escape route must be on the grid and not part of the snake's current body
                            if self._is_on_grid(neighbor) and neighbor not in snake_pos:
                                escape_routes += 1

                        if escape_routes == 0: # No immediate way off the cycle path
                            current_hamiltonian_bonus *= 0.25 # Significantly reduce bonus
                        elif escape_routes == 1: # Only one way off
                            current_hamiltonian_bonus *= 0.75 # Slightly reduce bonus

                    except KeyError:
                        # Should not happen if _follows_hamiltonian is true and pos is on cycle.
                        # If pos is somehow not in self.hamiltonian_indices, skip this adjustment.
                        pass

                score += current_hamiltonian_bonus

            # Penalty for recently visited locations (to avoid short loops)
            # snake_pos[0] is current head. snake_pos[1] is its previous location (now neck).
            # We want to penalize if the new head 'pos' lands on a spot where the head was
            # a few steps ago (e.g., snake_pos[2], snake_pos[3], ...).
            # _get_valid_moves already prevents moving to snake_pos[1]'s future location.

            # Determine how deep into the snake's body we check for recent positions.
            # Max depth is self.recent_history_length, but also limited by snake's actual length.
            # We check from index 2 (segment after neck) up to recent_history_length.
            # E.g., if recent_history_length = 5, we check snake_pos[2], snake_pos[3], snake_pos[4].
            history_check_limit = min(self.recent_history_length, len(snake_pos))
            for k in range(2, history_check_limit):
                if pos == snake_pos[k]:
                    score -= 500  # Apply penalty
                    break         # Apply penalty only once

            # Danger Zone Assessment: Penalize moving into constrictions if not on a food path
            if golden_path is None:
                # snake_pos[0] is the current head before this move.
                # 'pos' is the potential next head position.
                # 'move' is the direction string for the current move being evaluated.
                simulated_body_after_this_move = (pos,) + snake_pos[:-1]

                if self._is_dangerous_constriction(pos, move, simulated_body_after_this_move):
                    score -= 1000  # Apply a significant penalty
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else safe_moves[0][0]