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
        space = self._flood_fill(pos, next_snake_body)
        score += space * (1.0 + depth * 0.5)  # Weight space more heavily in early moves
        
        # Evaluate food distance
        dist_to_food = self._manhattan_distance(pos, food_pos)
        score -= dist_to_food * (2.0 - depth * 0.5)  # Food distance matters less in later moves
        
        # Check if move follows Hamiltonian cycle
        if self._follows_hamiltonian(pos, snake_body[0]):
            score += 200 * (depth * 0.5)
        
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

    def _flood_fill(self, start_pos: Position, snake_body: Tuple[Position, ...]) -> int:
        if not self._is_on_grid(start_pos) or start_pos in snake_body:
            return 0
        
        q: Deque[Position] = deque([start_pos])
        visited = {start_pos}
        count = 0
        obstacles = set(snake_body)

        while q:
            pos = q.popleft()
            count += 1
            for direction in self.directions:
                next_pos = self._move_position(pos, direction)
                if (self._is_on_grid(next_pos) and 
                    next_pos not in obstacles and 
                    next_pos not in visited):
                    visited.add(next_pos)
                    q.append(next_pos)
        
        return count

    def _get_direction_from_path(self, head: Position, next_pos: Position) -> str:
        dx = next_pos.x - head.x
        dy = next_pos.y - head.y
        if dx == 1: return RIGHT
        if dx == -1: return LEFT
        if dy == 1: return DOWN
        if dy == -1: return UP
        raise ValueError("Positions are not adjacent")

    def _is_path_a_trap(self, path: List[Position], snake_body: Tuple[Position, ...]) -> bool:
        simulated_snake = path + list(snake_body)
        simulated_snake = simulated_snake[:len(snake_body) + 1]
        new_head = simulated_snake[0]
        new_tail = simulated_snake[-1]
        path_to_tail = self._find_path(new_head, new_tail, tuple(simulated_snake))
        return path_to_tail is None

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
            if not is_trap or self.moves_since_food > self.patience_threshold:
                golden_path = path_to_food

        # Score moves with lookahead
        best_move = None
        best_score = -float('inf')

        for move, pos in safe_moves:
            score = self._evaluate_future_state(pos, snake_pos, food)

            # Additional scoring factors
            if golden_path and pos == golden_path[0]:
                score += 10000
            
            # Hamiltonian cycle bonus (stronger when snake is longer)
            if self._follows_hamiltonian(pos, head):
                score += 200 * snake_ratio
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move if best_move else safe_moves[0][0]