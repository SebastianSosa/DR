#%%
import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt

#%%
##################
### Functions ####
##################
#### Build environment------------------
def create_env(grid_size = 12, num_black_cells = 20):    
    grid = np.zeros((grid_size, grid_size))
    
    # Randomly add black cells
    black_cell_ids = np.random.choice(grid_size * grid_size, size=num_black_cells, replace=False)
    row_indices, col_indices = np.unravel_index(black_cell_ids, (grid_size, grid_size))
    grid[row_indices, col_indices] = 1

    # Add player and AI circles to the grid
    white_cell_ids = [i for i in np.array(range(1, grid_size*grid_size)) if i not in black_cell_ids]
    picks = np.random.choice(white_cell_ids, 2)
    player_position = np.unravel_index(picks[0], (grid_size, grid_size))
    player_position= (player_position[1]+0.5, player_position[0]+0.5)
    ai_position = np.unravel_index(picks[1], (grid_size, grid_size))
    ai_position= (ai_position[1]+0.5, ai_position[0]+0.5)
    
    return grid, black_cell_ids, player_position, ai_position

def draw_env(player_position, ai_position, grid, grid_size = 12):
    # Plot the grid
    plt.imshow(grid, cmap='binary', origin='lower', extent=[0, grid_size, 0, grid_size])
    plt.xticks(np.arange(grid_size))
    plt.yticks(np.arange(grid_size))
    plt.grid(color='black', linewidth=2)
    
    # Add circle annotations
    plt.annotate('P', player_position, color='black', ha='center', va='center', weight='bold')
    plt.annotate('AI', ai_position, color='black', ha='center', va='center', weight='bold')
    
    # Draw a red line between player and AI
    player_row, player_col = player_position
    ai_row, ai_col = ai_position
    plt.plot( [player_row, ai_row], [player_col, ai_col], 'bo', linestyle="--")
    #plt.plot( [player_row+0.5, ai_row+0.5], [player_col+0.5, ai_col+0.5], 'bo', linestyle="--")
    #plt.plot( [player_row-0.5, ai_row-0.5], [player_col-0.5, ai_col-0.5], 'bo', linestyle="--")

    plt.draw()
    plt.pause(0.1)
    plt.clf()

### Moovments --------------------------
def move(current_position, black_cell_ids, grid_size, action):
    # Define the directions based on the action
    directions = {
        0: (-1, 0),       # Up
        1: (1, 0),        # Down
        2: (0, -1),       # Left
        3: (0, 1),        # Right
        4: (-1, -1),      # Top-Left
        5: (-1, 1),       # Top-Right
        6: (1, -1),       # Bottom-Left
        7: (1, 1)         # Bottom-Right
    }

    # Get the selected direction based on the action
    direction = directions[action]

    # Determine the new position based on the selected direction
    row, col = current_position
    new_row = row + direction[0]
    new_col = col + direction[1]

    # Check if the new position is valid and not a black cell
    if (
        new_row >= 0
        and new_row < grid_size
        and new_col >= 0
        and new_col < grid_size
    ):
        block_id = np.ravel_multi_index(np.array([int(new_col), int(new_row)]),(grid_size,grid_size))
        if  block_id not in black_cell_ids:
            return (new_row, new_col)
        else:
            return current_position
    else:
        # If the new position is invalid, the AI remains at its current position
        return current_position

#### Informations for agents------------------ 
def compute_distance(position1, position2):
    # Calculate the Euclidean distance between two positions
    x1, y1 = position1
    x2, y2 = position2
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def black_cell_between_players(ai_position, player_position, black_cell_ids):
    x_min = int(min(ai_position[0], player_position[0]))
    x_max = int(max(ai_position[0], player_position[0]))
    y_min = int(min(ai_position[1], player_position[1]))
    y_max = int(max(ai_position[1], player_position[1]))
    
    black_blocks_coord = np.unravel_index(black_cell_ids, (12, 12))

    blocks_between_players = 0
    for i in range(len(black_blocks_coord[0])):
        if x_min <= black_blocks_coord[1][i] <= x_max and y_min <= black_blocks_coord[0][i] <= y_max:
            blocks_between_players +=1
    return blocks_between_players

def is_ai_adjacent_to_black_cell(ai_position, black_cell_ids, grid_size = 12):
    adjacent_positions = [
        [ai_position[0] - 0.5, ai_position[1] + 0.5], # Up
        [ai_position[0] - 0.5, ai_position[1] - 1.5], # Down
        [ai_position[0] - 1.5, ai_position[1]],       # Left
        [ai_position[0] + 0.5, ai_position[1]],       # Right
        [ai_position[0] - 1.5, ai_position[1] + 0.5],  # Top-left
        [ai_position[0] + 0.5, ai_position[1] + 0.5],  # Top-right
        [ai_position[0] - 1.5, ai_position[1] - 1.5],  # Bottom-left
        [ai_position[0] + 0.5, ai_position[1] - 1.5],  # Bottom-right
    ]
    adjacent_positions = np.clip(adjacent_positions, 0, 11)
    for position in adjacent_positions:
        if position[0] < 0 :
            position[0] = 0
        if position[1] < 0:
            position[1] = 0
        if np.ravel_multi_index(np.array([int(position[1]), int(position[0])]),(grid_size,grid_size)) in black_cell_ids:
            return True

    return False

def cell_in_line_players(ai_position, player_position):
    x1, y1 = ai_position[0]-0.5, ai_position[1]-0.5
    x2, y2 = player_position[0]-0.5, player_position[1]-0.5
    # Line space between player and AI coordinates
    x_coords = np.concatenate((np.ceil(np.linspace(x1, x2, int(abs(x1-x2)+int(abs(y1-y2))))),
                               np.floor(np.linspace(x1, x2, int(abs(x1-x2)+int(abs(y1-y2)))))))
    y_coords = np.concatenate((np.ceil(np.linspace(y1, y2, int(abs(x1-x2)+int(abs(y1-y2))))),
                               np.floor(np.linspace(y1, y2, int(abs(x1-x2)+int(abs(y1-y2)))))))

    blocks_in_line = [(int(x), int(y)) for x, y in zip(x_coords, y_coords)]
    return list(set(blocks_in_line))

def black_cell_in_line_with_players(blocks_in_line, black_cell_ids, grid_size = 12):
    black_cell = []
    for i in range(len(blocks_in_line)):
        if np.ravel_multi_index(np.array([blocks_in_line[i][1], blocks_in_line[i][0]]),(grid_size,grid_size)) in black_cell_ids:
            black_cell.append(blocks_in_line[i])
    return len(black_cell)

#%%
##### testing functions------------------ 
#grid, black_cell_ids, player_position, ai_position = create_env()
##for a in range(50):
##    ai_position = move(ai_position, black_cell_ids, grid_size = 12, action =  random.randrange(8))	
##    draw_env(player_position, ai_position, grid)
##    blocks_in_line = cell_in_line_players(ai_position, player_position)
##    print(black_cell_in_line_with_players(blocks_in_line, black_cell_ids))
#draw_env(player_position, ai_position, grid)
#print("Number of black block(s) between players: " + str(black_cell_between_players(ai_position, player_position, black_cell_ids)))
#print("AI adjacent to black block(s): " + str(is_ai_adjacent_to_black_cell(ai_position, black_cell_ids)))
#print("Cell(s) in line between players: ")
#blocks_in_line = cell_in_line_players(ai_position, player_position)
#print("blocks_in_line:")
#print(blocks_in_line)
#print("black_cell_in_line_with_players:")
#print(black_cell_in_line_with_players(blocks_in_line, black_cell_ids, grid_size = 12))
#print("shortest_path")

#%%
##############
### Class ####
##############
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(8)       
        self.observation_space = spaces.Box(low=-500, high=500, shape=(7,), dtype=np.float64)
        self.state = 38 + random.randint(-3,3)
        self.grid_size = 12
        self.count = 0
        self.ai_position = None
        self.player_position = None
        self.distance_between_players = None
        self.blocks_between_players = None
        self.blocks_in_line_players = None
        self.time_to_find = None
        self.ai_path = None
        self.player_path = None
        self.ai_near_block = False
        self.collision = False
        self.game_over = False
        self.prev_actions = []
        
    def step(self, action):
        self.prev_actions.append(action)
        self.count += 1 
        self.done = False
        if self.count <= 50:
        # Moove AI and recompute information
            self.ai_position = move(self.ai_position, self.black_cell_ids, self.grid_size, action)
            self.distance_between_players = compute_distance(self.ai_position, self.player_position)
            self.blocks_between_players = black_cell_between_players(self.ai_position, self.player_position, self.black_cell_ids)
            self.blocks_in_line_players = cell_in_line_players(self.ai_position, self.player_position)
            self.black_blocks_in_line_players =black_cell_in_line_with_players( self.blocks_in_line_players, self.black_cell_ids)
            self.ai_near_block = is_ai_adjacent_to_black_cell(self.ai_position, self.black_cell_ids)
        
            observation = [self.player_position[0], self.player_position[1],
               self.ai_position[0], self.ai_position[1],
               self.distance_between_players, 
               self.black_blocks_in_line_players,
               self.ai_near_block ] 
            
        else:
        # Moove player
            self.player_position = move(self.player_position, self.black_cell_ids, grid_size = 12, action = random.randrange(0,7))
            self.distance_between_players = compute_distance(self.ai_position, self.player_position)
            self.blocks_between_players = black_cell_between_players(self.ai_position, self.player_position, self.black_cell_ids)
            self.blocks_in_line_players = cell_in_line_players(self.ai_position, self.player_position)
            self.black_blocks_in_line_players =black_cell_in_line_with_players( self.blocks_in_line_players, self.black_cell_ids)
            self.ai_near_block = is_ai_adjacent_to_black_cell(self.ai_position, self.black_cell_ids)
            
            if self.distance_between_players <= 1:
                self.game_over = True
                
            observation = [self.player_position[0], self.player_position[1],
               self.ai_position[0], self.ai_position[1],
               self.distance_between_players,
               self.black_blocks_in_line_players,
               self.ai_near_block ] 
            
        self.ai_reward += 10
        info = {}
        observation = np.array(observation)

        draw_env(self.player_position, self.ai_position, self.grid)
        
        if self.count >= 150:
            self.done = True
            
        return observation, self.ai_reward, self.game_over, self.done, info

    def reset(self, seed=None, options=None):
        self.done = False
        self.game_over = False
        self.count = 0

        # create observation:
        self.grid, self.black_cell_ids, self.player_position, self.ai_position = create_env(grid_size = 12)
        self.distance_between_players = compute_distance(self.ai_position, self.player_position)
        self.blocks_between_players = black_cell_between_players(self.ai_position, self.player_position, self.black_cell_ids)
        self.blocks_in_line_players = cell_in_line_players(self.ai_position, self.player_position)
        self.black_blocks_in_line_players =black_cell_in_line_with_players( self.blocks_in_line_players, self.black_cell_ids)
        self.ai_near_block = is_ai_adjacent_to_black_cell(self.ai_position, self.black_cell_ids)
        self.ai_reward = 0
        self.ai_score = 0  
        self.prev_button_direction = 1
        self.button_direction = 1        
        
        observation = [self.player_position[0], self.player_position[1],
                       self.ai_position[0], self.ai_position[1],
                       self.distance_between_players, 
                       self.black_blocks_in_line_players,
                       self.ai_near_block ]
        observation = np.array(observation, dtype = np.float32)
        

        info = {}
        return observation, info

    def render(self):
        draw_env(self.player_position, self.ai_position, self.grid)
        
    def close(self):
        plt.close('all')
