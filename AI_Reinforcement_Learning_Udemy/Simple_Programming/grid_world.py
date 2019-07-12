import numpy as np

class Grid: # The environment
	def __init__(self, width, height, start):
		self.width = width
		self.height = height
		# The starting point coordinates (row, column)
		self.i = start[0]
		self.j = start[1]

	def set(self, rewards, actions):
	# rewards = dictionary[i, j] : r(row, column) : rewards
	# actions = dictionary[i, j] : A(row, column) : list of possible actions
		self.rewards = rewards
		self.actions = actions

	def set_state(self, s):
		self.i = s[0]
		self.j = s[1]

	def current_state(self):
		# return the current position (i,j)
		return (self.i, self.j)

	def is_terminal(self, s):
		# return boolean : for if the state is "Terminal"
		return s not in self.actions

	def move(self, action):
		# first check if move is possible from "actions" list
		if action in self.actions[(self.i, self.j)]:
			if action == 'U':
				self.i -= 1
			elif action == 'D':
				self.i += 1
			elif action == 'R':
				self.j += 1
			elif action == 'L':
				self.j -= 1
		# return the reward for this position, from the reward dictionary
		# or 0 if we tried to move away from the envvirinment (grid)
		return self.rewards.get((self.i, self.j), 0)

	def undo_move(self, action):
		if action == 'U':
			self.i += 1
		elif action == 'D':
			self.i -= 1
		elif action == 'R':
			self.j += 1
		elif action == 'L':
			self.j -= 1
		# raise an exception if the current position is not a possible position
		# Possible positions are only the ones with a reward or an action
		# i.e. the keys:(i,j) of the "reward" and the "actions" dictionaries
		assert(self.current_state() in self.all_states())

	def game_over(self):
		# return TRUE if there are no possible actions
		# else return FALSE
		return (self.i, self.j) not in self.actions

	def all_states(self):
		# return all the keys:(i, j) that have an action or reward
		# i.e. The position inside the environment(grid)
		# set-> finds the unique values
		key = {**self.actions, **self.rewards}
		key = key.keys()
		return list(key)


def standard_grid(): # Define the Grid
	# x-> can not go there
	# s-> start position
	# number-> reward

	# . . . 1
	# . x . -1
	# s . . .

	g = Grid(4, 3, (2,0)) # Width, Height, Start_position
	rewards = {(0, 3):1, (1, 3):-1}
	actions = {
	# (0, 3) & (1, 3) are terminals, thus they do not have any action
	(0, 0):('D', 'R'),
	(0, 1):('L', 'R'),
	(0, 2):('L', 'D', 'R'),
	(1, 0):('U', 'D'),
	(1, 2):('U', 'D', 'R'),
	(2, 0):('U', 'R'),
	(2, 1):('L', 'R'),
	(2, 2):('L', 'R', 'U'),
	(2, 3):('L', 'U'),
	}
	g.set(rewards, actions)
	return g

	
	
def play_game(agent, env):
	pass
