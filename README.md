# Deep Q-learning Reinforcement Learning for Hide and Seek Game

This project aims to implement Deep Q-learning for a hide and seek game. The AI is dedicated to hiding in a 12x12 grid where black cells are randomly placed. The goal is to train an AI that can efficiently hide on a new map it has never seen during training.

##Solution (environment.py)

I have created the following functions:

     1. create_env: Builds a 12x12 grid world, places random black blocks within the grid, and randomly positions the agents (player and AI) on non-black blocks. It returns the grid, the coordinates of the black cells, and the positions of the player and AI within the grid.

     2. draw_env: Plots the grid, black cells, player, and AI.

     3. move: Defines the neighboring non-black cells.

     4. compute_distance: Computes the Euclidean distance between the player and AI.

     5. black_cell_between_players: Computes the number of blocks between the coordinates of the player and AI.

     6. cell_in_line_players: Computes the number of cells that are in a straight line between the player and AI.

     7. black_cell_in_line_with_players: Computes the black cells' coordinates in order to calculate the number of black cells that are in a straight line between the player and AI.

     8. is_ai_adjacent_to_black_cell: Evaluates if the AI is adjacent to a black cell.
     
I have built the following gymnasium environment for training:

     1. Player search with 8 discrete actions (turn left, right, down, up, and diagonals).

     2. Observations include:
          1. Player coordinates.
          2. AI coordinates.
          3. Player coordinates.
          4. Distance between the player and AI.
          5. Number of black cells in a straight line between the player and AI.
          6. Number of black cells between the player and AI.
          7. Is AI near a black cell?
          
     3. The environment sets up the world using the create_env function and computes observations.

     4. Steps consist of:

          1. The AI randomly moves in the grid for the first 50 rounds.
          2. From rounds 50 to 150, the player randomly moves in the grid.
          3. The game is over when the player is one block away from the AI.
          4. Observations are computed and stored at each step.
          5. The AI's reward is incremented by 10 for each round that the player is not one block away from the AI.

## Deployment

     Conda environment: XXX
     Run the script XXX to train the AI. In the script, you can set the number of training rounds.

## Dependencies

random gymnasium numpy matplotlib.pyploT

##Potential Improvements

1. Instead of a random walk for the player, implement a shortest path algorithm. This could speed up training.
2. Improve the computation of the number of blocks in line with the players.
