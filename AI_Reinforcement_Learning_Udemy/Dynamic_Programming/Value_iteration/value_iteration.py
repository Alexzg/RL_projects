import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

import os

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# this is deterministic
# all P(s',r|s,a) = 1 or 0


if __name__ =='__main__':
	# this grid gives you a reward of -0.1 for every non-terminal state
	# we want to see if this will encourage finding a shorter path to the goal
	grid = negative_grid()

	#print rewards
	print ('rewards:')
	print_values(grid.rewards, grid)

	# state -> action
	# we'll randomly chose an action and update as we learn
	policy = {}
	for s in grid.actions.keys():
		policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

	# initial policy
	print ('initial_policy:')
	print_policy(policy, grid)

	# initial V(s)
	V = {}
	states = grid.all_states()
	for s in states:
		# V[s] = 0
		if s in grid.actions:
			V[s] = np.random.random()
		else:
			#terminal state
			V[s] = 0
	
	# Calculate the Values of the states
	# Repeat until convergence
	# V[s] = max[a]{sum[s',r] {p(s',r|s,a)[r + gamma*V[s']]}}
	while True:
		biggest_change = 0
		
		for s in states:
			old_v = V[s]
			# V(s) only has value if it's not a terminal state
			if s in policy:
				new_v = float('-inf')
				for a in ALL_POSSIBLE_ACTIONS:
					grid.set_state(s)
					r = grid.move(a)
					v = 1 * (r + GAMMA * V[grid.current_state()])
					if v > new_v:
						new_v = v
				V[s] = new_v
				# We want to find the biggest Value change, if we were in the previous state (old_v)
				biggest_change = max(biggest_change, np.abs(old_v - V[s]))
		if biggest_change < SMALL_ENOUGH:
			break
	
	# Find the policy that leads to optimal value function
	# The Values have been previously calculated
		for s in policy.keys():
			best_a = None
			best_value = float('-inf')
			# loop through all possible actions to find best current action, based on the pre-calculated Values of previous step
			for a in ALL_POSSIBLE_ACTIONS:
				grid.set_state(s)
				r = grid.move(a)
				v = 1 * (r + GAMMA + V[grid.current_state()])
				if v > best_value:
					best_value = v
					best_a = a
				policy[s] = best_a
		
	print ('values:')
	print_values(V, grid)
	print ('policy:')
	print_policy(policy, grid)
	
os.system("pause")
