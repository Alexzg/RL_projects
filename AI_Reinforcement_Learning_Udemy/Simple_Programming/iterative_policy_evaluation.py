import numpy as np
import grid_world

SMALL_ENOUGH = 10e-4 # threshold for convergence

# Visualize values
# V:values dictionary, g:Grid class
def print_values(V, g):
	for i in range(g.height):
		print ('-------------------------')
		for j in range(g.width):
			v = V.get((i,j), 0)
			if v >= 0:
				print ((' %.2f') % v, end='|')
			else:
				print (('%.2f') % v, end='|')
		print ('')

# Visualize policies
# (only for deterministic policies) for each state only one policy can be printed
# V:policy dictionary, g:Grid class
def print_policy(P, g):
	for i in range(g.height):
		print ('-------------------------')
		for j in range(g.width):
			a = P.get((i,j), ' ') # policy_dict = (i,j):action
			print ((' %s ') % a, end='|')
		print ('')

if __name__=='__main__':
	# iterative policy evaluation
	# given a policy, find it's values function V(s)
	# this will be done for a) uniform random policy, b) fixed policy
	# Note:
	# there are 2 sources of randomness
	# p(a|s) - deciding what action to take given the state
	# p(s',r|s,a) - the next action and reward given your action-state pair
	# here it is only modeling p(s|a)=uniform
	grid = grid_world.standard_grid()

	# states are positions (i,j)
	states = grid.all_states()

	### uniformly random actions ###
	# initialize V(s) = 0
	V = {}
	for s in states:
		V[s] = 0
	gamma = 1.0 # discount factor

	# repeat until convergence
	while True:
		biggest_change = 0
		for s in states:
			old_v = V[s]
			# V(s) only has value if it's not a terminal state
			if s in grid.actions:

				new_v = 0 # we will accumulate the answer
				p_a = 1.0 / len(grid.actions[s]) # each action has equal probability
				for a in grid.actions[s]:
					grid.set_state(s)
					r = grid.move(a)
					new_v += p_a * (r + gamma * V[grid.current_state()])
				V[s] = new_v
				biggest_change = max(biggest_change, np.abs(old_v - V[s]))

		if biggest_change < SMALL_ENOUGH:
			break
	print("values for uniformly random actions:")
	print_values(V, grid)
	print("\n\n")

	  ### fixed policy ###
	policy = {
		(2, 0): 'U',
		(1, 0): 'U',
		(0, 0): 'R',
		(0, 1): 'R',
		(0, 2): 'R',
		(1, 2): 'R',
		(2, 1): 'R',
		(2, 2): 'R',
		(2, 3): 'U',
	}
	print_policy(policy, grid)

	# initialize V(s) = 0
	V = {}
	for s in states:
		V[s] = 0

	# let's see how V(s) changes as we get further away from the reward
	gamma = 0.9 # discount factor

	# repeat until convergence
	while True:
		biggest_change = 0
		for s in states:
			old_v = V[s]

			# V(s) only has value if it's not a terminal state
			if s in policy:
				a = policy[s]
				grid.set_state(s)
				r = grid.move(a)
				V[s] = r + gamma * V[grid.current_state()]
				biggest_change = max(biggest_change, np.abs(old_v - V[s]))

		if biggest_change < SMALL_ENOUGH:
			break
	print("values for fixed policy:")
	print_values(V, grid)
	print("\n\n")

	print("Coordinates:")
	  ### fixed policy ###
	coordinates = {
		(2, 0): '(2, 0)',
		(1, 0): '(1, 0)',
		(0, 0): '(0, 0)',
		(0, 1): '(0, 1)',
		(0, 2): '(0, 2)',
		(1, 2): '(1, 2)',
		(2, 1): '(2, 1)',
		(2, 2): '(2, 2)',
		(2, 3): '(2, 3)',
		(1, 1): '  X   ',
		(0, 3): '  1   ',
		(1, 3): ' -1   ',
	}
	print_policy(coordinates, grid)

input('Exit?')
