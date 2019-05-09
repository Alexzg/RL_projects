import gym
import numpy as np
from matplotlib import pyplot as plt

def run_episode(env, parameters):  
	observation = env.reset()
	totalreward = 0
	for _ in range(200):
		action = 0 if np.matmul(parameters,observation) < 0 else 1
		observation, reward, done, info = env.step(action)
		totalreward += reward
		if done:
			break
	return totalreward

bestparams = None  
bestreward = 0
total_reward_list = []
env = gym.make('CartPole-v0')

for ep in range(2000):  
	parameters = np.random.rand(4)
	reward = run_episode(env,parameters)
	total_reward_list.append(reward)
	if reward > bestreward:
		bestreward = reward
		bestparams = parameters
		# considered solved if the agent lasts 200 timesteps 

plt.title('Total rewards')
plt.plot(total_reward_list)
print('Plot is displayed')
plt.show()
