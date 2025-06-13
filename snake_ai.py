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

class SnakeAI:
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

        # --- Tunable Parameters ---
        # Evaluation constants
        self.default_lookahead_depth = 3
        self.space_eval_depth_factor_base = 1.0
        self.space_eval_depth_factor_multiplier = 0.5
        self.food_eval_depth_factor_base = 2.0
        self.food_eval_depth_factor_multiplier = 0.5
        self.hamiltonian_depth_bonus_factor = 50.0
        self.future_state_recursion_decay = 0.5
        self.flood_fill_freedom_neutral = 1.5
        self.flood_fill_freedom_scale = 0.25

        # Patience mechanism
        self.base_patience_factor = 0.3
        self.max_patience_factor = 0.7
        self.trapped_path_patience_leniency_factor = 0.5
        
        # Scoring bonuses/penalties
        self.food_path_direct_bonus = 10000.0
        self.hamiltonian_move_bonus_factor = 50.0
        self.hamiltonian_contextual_length_threshold_factor = 0.4
        self.hamiltonian_no_escape_penalty_factor = 0.25
        self.hamiltonian_one_escape_penalty_factor = 0.75
        self.recent_history_length = 5 # Also a form of penalty adjustment
        self.recent_position_penalty = -500.0
        self.danger_zone_penalty = -1000.0
        self.danger_zone_safe_moves_threshold = 1
        # --- End Tunable Parameters ---

        # Dynamic patience threshold (calculated)
        self.base_patience = self.grid_width * self.grid_height * self.base_patience_factor
        self.max_patience = self.grid_width * self.grid_height * self.max_patience_factor
        
        # Initialize Hamiltonian cycle
        self._init_hamiltonian_cycle()
        # self.recent_history_length = 5 # Moved to tunable parameters

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

    def _score_future_space_and_freedom(self, pos_to_eval: Position, snake_body_for_eval: Tuple[Position, ...], current_depth: int) -> float:
        space, avg_freedom = self._flood_fill(pos_to_eval, snake_body_for_eval)
        freedom_modifier = 1.0 + (avg_freedom - self.flood_fill_freedom_neutral) * self.flood_fill_freedom_scale
        effective_space_score = space * freedom_modifier
        return effective_space_score * (self.space_eval_depth_factor_base + current_depth * self.space_eval_depth_factor_multiplier)

    def _score_future_food_distance(self, pos_to_eval: Position, food_position: Position, current_depth: int) -> float:
        dist_to_food = self._manhattan_distance(pos_to_eval, food_position)
        # Returns a negative value or zero, as it's a penalty
        return -dist_to_food * (self.food_eval_depth_factor_base - current_depth * self.food_eval_depth_factor_multiplier)

    def _score_future_hamiltonian_adherence(self, pos_to_eval: Position, prev_head_pos: Position, current_depth: int) -> float:
        if self._follows_hamiltonian(pos_to_eval, prev_head_pos):
            # Using space_eval_depth_factor_multiplier for the (depth * 0.5) part for consistency
            return self.hamiltonian_depth_bonus_factor * (current_depth * self.space_eval_depth_factor_multiplier)
        return 0.0

    def _evaluate_future_state(self, pos: Position, snake_body: Tuple[Position, ...], 
                             food_pos: Position, depth: int = -1) -> float: # Default changed
        """Evaluate future states recursively up to a certain depth"""
        if depth == -1: # Use instance default if not overridden
            depth = self.default_lookahead_depth

        if depth == 0: # Base case for recursion
            return 0.0
        
        current_score = 0.0
        # snake_body[0] is the head position *before* moving to 'pos'.
        # next_snake_body_for_eval is the snake's body configuration *after* head moves to 'pos'.
        next_snake_body_for_eval = (pos,) + snake_body[:-1]

        # Calculate score components using helper methods
        current_score += self._score_future_space_and_freedom(pos, next_snake_body_for_eval, depth)
        current_score += self._score_future_food_distance(pos, food_pos, depth)
        # prev_head_pos for Hamiltonian check is snake_body[0] from the input snake_body
        current_score += self._score_future_hamiltonian_adherence(pos, snake_body[0], depth)

        # Recursive minimax-like evaluation of subsequent states
        max_future_score = -float('inf')

        for direction in self.directions: # self.directions are global or instance [UP, DOWN, LEFT, RIGHT]
            potential_next_step_pos = self._move_position(pos, direction)

            # Safety check for the recursive call:
            # The 'body' for this check is next_snake_body_for_eval.
            if self._is_safe(potential_next_step_pos, next_snake_body_for_eval):
                future_score = self._evaluate_future_state(
                    potential_next_step_pos, next_snake_body_for_eval, food_pos, depth - 1
                )
                # Only update if future_score is actually better (handles -inf)
                if future_score > max_future_score:
                     max_future_score = future_score

        if max_future_score > -float('inf'): # Check if any valid recursive path was found
            current_score += max_future_score * self.future_state_recursion_decay
        # If all future paths from here are terminal (no safe moves or depth ran out and returned 0 from bad spots),
        # max_future_score might remain -float('inf'). In this case, we don't add it.
        # The current_score will then reflect only the heuristic of the current state 'pos'.

        return current_score

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

        # If there's X or fewer ways to continue safely, it's a dangerous constriction.
        return safe_continuing_moves <= self.danger_zone_safe_moves_threshold

    def _calculate_golden_path_bonus(self, pos_to_evaluate: Position, golden_path_list: Optional[List[Position]]) -> float:
        if golden_path_list and pos_to_evaluate == golden_path_list[0]:
            return self.food_path_direct_bonus
        return 0.0

    def _calculate_hamiltonian_bonus(self, pos_to_evaluate: Position, current_head_pos: Position, current_snake_body: Tuple[Position, ...], current_snake_ratio: float) -> float:
        score_adjustment = 0.0
        if self._follows_hamiltonian(pos_to_evaluate, current_head_pos):
            bonus = self.hamiltonian_move_bonus_factor * current_snake_ratio

            if len(current_snake_body) > self.grid_width * self.grid_height * self.hamiltonian_contextual_length_threshold_factor:
                try:
                    pos_idx = self.hamiltonian_indices[pos_to_evaluate]
                    next_on_cycle = self.hamiltonian_cycle[(pos_idx + 1) % len(self.hamiltonian_cycle)]
                    escape_routes = 0
                    for direction in self.directions:
                        neighbor = self._move_position(pos_to_evaluate, direction)
                        if neighbor == current_head_pos or neighbor == next_on_cycle:
                            continue
                        if self._is_on_grid(neighbor) and neighbor not in current_snake_body:
                            escape_routes += 1

                    if escape_routes == 0:
                        bonus *= self.hamiltonian_no_escape_penalty_factor
                    elif escape_routes == 1:
                        bonus *= self.hamiltonian_one_escape_penalty_factor
                except KeyError:
                    pass # pos_to_evaluate not in cycle, _follows_hamiltonian might be more robust
            score_adjustment = bonus
        return score_adjustment

    def _calculate_recent_position_penalty(self, pos_to_evaluate: Position, current_snake_body: Tuple[Position, ...]) -> float:
        history_check_limit = min(self.recent_history_length, len(current_snake_body))
        for k in range(2, history_check_limit): # Check snake_body[2] onwards
            if pos_to_evaluate == current_snake_body[k]:
                return self.recent_position_penalty # This is a negative value
        return 0.0

    def _calculate_danger_zone_penalty(self, pos_to_evaluate: Position, move_direction: str, current_head_pos: Position, current_snake_body: Tuple[Position, ...], golden_path_list: Optional[List[Position]]) -> float:
        if golden_path_list is None:
            # current_snake_body is before the move, head is current_head_pos.
            # simulated_body_after_this_move has pos_to_evaluate as its head.
            simulated_body_after_this_move = (pos_to_evaluate,) + current_snake_body[:-1]
            if self._is_dangerous_constriction(pos_to_evaluate, move_direction, simulated_body_after_this_move):
                return self.danger_zone_penalty # This is a negative value
        return 0.0

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
               (is_trap and self.moves_since_food > self.patience_threshold * self.trapped_path_patience_leniency_factor) or \
               self.moves_since_food > self.patience_threshold:
                golden_path = path_to_food

        # Score moves with lookahead
        best_move = None
        best_score = -float('inf')
        # head = snake_pos[0] # Already defined above
        # snake_ratio = len(snake_pos) / (self.grid_width * self.grid_height) # Already defined above

        for move_str, next_pos in safe_moves: # 'move_str' is direction string, 'next_pos' is next Position
            # Initial score from future state evaluation
            current_score = self._evaluate_future_state(
                next_pos, snake_pos, food, depth=self.default_lookahead_depth
            )

            # Add bonuses and penalties
            current_score += self._calculate_golden_path_bonus(next_pos, golden_path)
            current_score += self._calculate_hamiltonian_bonus(next_pos, head, snake_pos, snake_ratio)
            current_score += self._calculate_recent_position_penalty(next_pos, snake_pos)
            current_score += self._calculate_danger_zone_penalty(next_pos, move_str, head, snake_pos, golden_path)
            
            if current_score > best_score:
                best_score = current_score
                best_move = move_str
        
        return best_move if best_move else safe_moves[0][0] # Fallback