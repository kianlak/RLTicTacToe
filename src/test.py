import numpy as np
import random

from enums import CellState as cs
from enums import GameState as gs

class TicTacToe:
	def __init__(self, length = 3):
		self.length = length
		self.grid = np.full((self.length,self.length), cs.EMPTY)
		self.moves = [(i, j) for i in range(self.length) for j in range(self.length)]
		self.game_state = gs.PLAYING
		
	def reset(self):
		self.grid = np.full((self.length,self.length), cs.EMPTY)
		self.moves = [(i, j) for i in range(self.length) for j in range(self.length)]
		self.game_state = gs.PLAYING

	# def check_winner(self):
	# 	winner_found = False

	# 	# Check for rows and columns
	# 	for i in range(self.length):
	# 		if np.all(self.grid[i, :] == cs.O) or np.all(self.grid[:, i] == cs.O):
	# 			self.game_state = cs.O
	# 			winner_found = True
	# 			break
			
	# 		if np.all(self.grid[i, :] == cs.X) or np.all(self.grid[:, i] == cs.X):
	# 			self.game_state = cs.X
	# 			winner_found = True
	# 			break
			
	# 	# Check for diagonals
	# 	if not winner_found:
	# 		for i in range(self.length):
	# 			if np.all(self.grid[i, i] == cs.O) or \
	# 				np.all(self.grid[i, self.length - i - 1] == cs.O):
	# 				self.game_state = cs.O
	# 				break

	# 			if np.all(self.grid[i, i] == cs.X) or \
	# 				np.all(self.grid[i, self.length - i - 1] == cs.X):
	# 				self.game_state = cs.X
	# 				break

	# 	return self.game_state

	def check_winner(self, grid):
		for row in grid:
			if row[0] == row[1] == row[2] and row[0] != 'empty':
				return row[0]  # Return 'x' or 'o'

    # Check columns
		for col in range(3):
			if grid[0][col] == grid[1][col] == grid[2][col] and grid[0][col] != 'empty':
				return grid[0][col]  # Return 'x' or 'o'

    # Check diagonals
		if grid[0][0] == grid[1][1] == grid[2][2] and grid[0][0] != 'empty':
			return grid[0][0]  # Return 'x' or 'o'
		
		if grid[0][2] == grid[1][1] == grid[2][0] and grid[0][2] != 'empty':
			return grid[0][2]  # Return 'x' or 'o'
		
		return gs.PLAYING  # No winner
	

	def step(self,move): #This should change the grid to add o, or x, and should change the moves list.
		#1: Get the initial state. Change the grid based on the move.
		#2: Remove corresponding action from grid.
		#3: If you won, or game ended, return True (for game ending).
		#Both in 3 and maybe in 2 we should decide the rewards.

		initial_grid_value = self.grid[move]
		if self.game_state == gs.PLAYER_WIN:
			return self.grid, self.game_state, x_winner_reward, True

		self.grid[move] = cs.O #Set grid based on move.

		self.game_state = self.check_winner(self.grid) #Made chatGPT make this shitty ass function above.

		o_winner_reward = 100 #Kind of arbitrary and dependent on what we want to do.
		x_winner_reward = -o_winner_reward #Should be the worst possible thing to happen.
		tie_winner_reward = 10 #Should be low, but not too low, since with optimal play, there is a tie.
		center_reward = 5
		corner_reward = 2
		remaining_reward = 0

		#Winning statement. Return grid and winner upon a winner being discovered, or upon a tie. Our "state" is both the grid and the winner.
		if self.game_state == gs.AGENT_WIN:
			return self.grid, self.game_state, o_winner_reward, True
		elif not (cs.EMPTY in self.grid.flatten()): #If self.actions is empty, this will be true.
			return self.grid, gs.TIE, tie_winner_reward, True
		else: #Here should go the rewards for particular actions, like putting in a center square. Maybe we start with no rewards at first?
			if initial_grid_value == cs.O:
				return self.grid, gs.PLAYING, -10000, False
			elif move == (1,1):
				return self.grid, gs.PLAYING, center_reward, False
			elif move == (0,0) or move == (2,0) or move == (0,2) or move == (2,2):
				return self.grid, gs.PLAYING, corner_reward, False
			else:
				return self.grid, gs.PLAYING, remaining_reward, False
		
#That should be the end of the environment itself.

class QLearningAgent:
	def __init__(self, env, learning_rate = 0.1, discount_factor=0.9, exploration_rate=0.1):
		self.env = env #Environment HAHA
		self.q_table = {} #Make dynamic list.
		self.learning_rate = learning_rate  # Alpha: learning rate
		self.discount_factor = discount_factor  # Gamma: discount factor for future rewards
		self.exploration_rate = exploration_rate  # Epsilon: exploration rate for epsilon-greedy
	
	def choose_move(self, state):
		if random.uniform(0, 1) < self.exploration_rate:
			available_moves_indexes = np.where(state.flatten == cs.EMPTY)[0]
			return random.choice(self.env.moves[available_moves_indexes])
		else:
			# Choose the action with the highest Q-value
			return self.best_move(state)
		
	def best_move(self,state):
		q_values = np.array([self.get_q_value(state,move) for move in self.env.moves])
		return np.max(q_values)
	
	def get_q_value(self,state, move):
		self.q_table.get((tuple(state, move)), 0)
		
	def update_q_value(self, state, move, reward, next_state):
		q_value = self.get_q_value(state,move)
		next_q_value = self.best_move(next_state)
		new_q_value = q_value + self.learning_rate*(reward + self.discount_factor*next_q_value - q_value)
		self.q_table[(tuple(state.flatten()), move)] = new_q_value


#Training:
def opponent_moves(grid, moves):
	available_moves_indexes = np.where(grid.flatten() == cs.EMPTY)[0]
	return random.choice(moves[available_moves_indexes])

	
def train_q(episodes = 10000):
	env = TicTacToe()
	agent = QLearningAgent()

	for episode in range(episodes):
		grid, moves, winner = env.reset()
		done = False

		while not done:
			move = agent.choose_move(grid)
			next_grid, reward, winner, done = env.step(move)
			if done:
				agent.update_q_value(grid, move, reward, next_grid)
				break

			opponent_move = opponent_move(next_grid, moves)

			next_grid[opponent_move] = cs.X

			grid = next_grid
		
	if (episode + 1) % 100 == 0:
		print(f"Episode {episode + 1}/{episodes} completed")

	return agent
