import random
import pygame
from snake_ai import SnakeAI, Position # Import Position class

# --- Constants ---
GRID_WIDTH = 30
GRID_HEIGHT = 30
CELL_SIZE = 20
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRID_LINE_COLOR = (50, 50, 50)

# --- Helper Functions (Updated for Position) ---
def search(content: int, grid: list[list[int]]) -> tuple[int, int]:
    """
    Finds the (x, y) coordinates of a given content value in the grid.
    Grid is indexed as grid[y][x].
    """
    for y, row in enumerate(grid):
        for x, cell_content in enumerate(row):
            if cell_content == content:
                return (x, y)
    return (-1, -1)

def spawn_food(grid: list[list[int]]):
    """
    Spawns food at a random empty location.
    Grid is indexed as grid[y][x].
    """
    empty_cells = [
        (x, y) for y in range(GRID_HEIGHT) 
        for x in range(GRID_WIDTH) if grid[y][x] == 0
    ]
    if empty_cells:
        x, y = random.choice(empty_cells)
        grid[y][x] = 2

# --- Pygame Initialization ---
pygame.init()
screen = pygame.display.set_mode((GRID_WIDTH * CELL_SIZE, GRID_HEIGHT * CELL_SIZE))
pygame.display.set_caption("Smart Snake AI")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# --- Game State Initialization ---
grid = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]

# Snake starts in the middle
start_pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
snake_body: list[tuple[int, int]] = [start_pos]
grid[start_pos[1]][start_pos[0]] = 1

direction: str | None = None
score = 0
game_over = False
moves_without_food = 0
MAX_MOVES_WITHOUT_FOOD = GRID_WIDTH * GRID_HEIGHT * 2

# Create initial food
spawn_food(grid)

# Initialize snake AI
if __name__ == '__main__':
    snake_ai = SnakeAI(GRID_WIDTH, GRID_HEIGHT)
    fps_counter = 0
    fps_timer = pygame.time.get_ticks()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not game_over:
            # --- AI Control ---
            food_pos = search(2, grid)
            if food_pos != (-1, -1):
                direction = snake_ai.get_best_move(
                    snake_body, 
                    food_pos,
                    direction if direction is not None else "up"
                )

            # --- Game Logic ---
            if direction:
                head = snake_body[0]
                new_x, new_y = head

                if direction == "up":    new_y -= 1
                elif direction == "down":  new_y += 1
                elif direction == "left":  new_x -= 1
                elif direction == "right": new_x += 1

                # Check for wall collisions
                if not (0 <= new_x < GRID_WIDTH and 0 <= new_y < GRID_HEIGHT):
                    game_over = True
                else:
                    new_head = (new_x, new_y)
                    # Check for self-collision
                    if new_head in snake_body[:-1]:
                        game_over = True
                    else:
                        # Handle movement and food
                        found_food = grid[new_y][new_x] == 2
                        
                        if found_food:
                            score += 1
                            moves_without_food = 0
                            # Only remove tail if not eating
                        else:
                            moves_without_food += 1
                            if moves_without_food >= MAX_MOVES_WITHOUT_FOOD:
                                game_over = True
                                print(f"Game over due to timeout! Score: {score}")
                            tail = snake_body.pop()
                            grid[tail[1]][tail[0]] = 0

                        # Update snake and grid
                        snake_body.insert(0, new_head)
                        grid[new_y][new_x] = 1

                        if found_food:
                            spawn_food(grid)

        # --- Drawing ---
        screen.fill(BLACK)

        # Draw grid content
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                cell_content = grid[y][x]

                color = BLACK
                if cell_content == 1:  # Snake body
                    color = GREEN
                elif cell_content == 2:  # Food
                    color = RED

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, GRID_LINE_COLOR, rect, 1)

        # Highlight snake head
        if snake_body:
            head_rect = pygame.Rect(
                snake_body[0][0] * CELL_SIZE, 
                snake_body[0][1] * CELL_SIZE, 
                CELL_SIZE, CELL_SIZE
            )
            pygame.draw.rect(screen, (150, 255, 150), head_rect)

        # Display score and moves without food
        score_text = font.render(f'Score: {score}', True, WHITE)
        moves_text = font.render(f'Moves left: {MAX_MOVES_WITHOUT_FOOD - moves_without_food}', True, WHITE)
        screen.blit(score_text, (10, 10))
        screen.blit(moves_text, (10, 40))

        # Calculate and display FPS
        fps_counter += 1
        current_time = pygame.time.get_ticks()
        if current_time - fps_timer > 1000:  # Every second
            fps = fps_counter
            fps_counter = 0
            fps_timer = current_time
            fps_text = font.render(f'FPS: {fps}', True, WHITE)
            screen.blit(fps_text, (10, 70))

        # Display game over message
        if game_over:
            game_over_text = font.render('Game Over!', True, RED)
            text_rect = game_over_text.get_rect(
                center=(GRID_WIDTH * CELL_SIZE // 2, GRID_HEIGHT * CELL_SIZE // 2)
            )
            screen.blit(game_over_text, text_rect)

        pygame.display.flip()
        clock.tick(144)  # Maximum speed while still being visible

    print(f"Final Score: {score}")
    pygame.quit()