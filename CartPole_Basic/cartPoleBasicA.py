import gym
import numpy as np
from matplotlib import pyplot as plt

env = gym.make('CartPole-v0')

def runEpisode(env,optimal_policy, parameters, episode):
	PARAMETER_USE_CHANCE = 0.95
	ACTION_0_CHANCE = 0.5
	RENDER_APPLICATION = False
	RENDER_APPLICATION_EPISODE_STEP = 20
	TIME_STEPS = 1000
	total_reward = 0
	policy = []
	observation = env.reset()
	for timestep in range(TIME_STEPS): 
		if RENDER_APPLICATION and (episode%RENDER_APPLICATION_EPISODE_STEP) == 0:
			#env.render() #Uncomment for visualization
			pass
		if np.random.rand()<PARAMETER_USE_CHANCE:
			action = 0 if np.matmul(parameters,observation) < 0 else 1
		else:
			action = 0 if np.random.rand() < ACTION_0_CHANCE else 1
		policy.append(action)
		observation, reward, done, info = env.step(action)
		total_reward += reward
		if total_reward == TIME_STEPS:
			print("Episode succeed after {} timesteps".format(timestep+1))
			break
		if reward==0 and timestep==(TIME_STEPS*0.3):
			print("Episode failed after {} timesteps".format(timestep+1))
			break
	return total_reward, policy, observation

def addDisturbanceManually(episodeNow, episodesWithDisturbance, parameters):
	for episode in episodesWithDisturbance:
		if episodeNow==episode:
			print('Manual Disturbance here ------')
			return np.random.rand(4)
	return parameters
		

EPISODES = 1000
EPISODES_UNTIL_RE_INITIALIZATION = 10
EPISODES_WITH_DISTURBANCE = [50, 100, 150, 450, 500, 520, 530, 700]
biggest_reward = 0
optimal_policy = []
total_reward_list = []
parameters = np.random.rand(4)
for i_episode in range(EPISODES):
	print('------- episode:', i_episode, ' ------')
	observation = env.reset()
	total_reward, policy, observation = runEpisode(env,optimal_policy, parameters, i_episode)
	parameters = addDisturbanceManually(i_episode, EPISODES_WITH_DISTURBANCE, parameters)
	total_reward_list.append(total_reward)
	if total_reward >= biggest_reward:
		biggest_reward = total_reward
		optimal_policy = policy
		parameters = parameters
		episodeCounter = 0
	episodeCounter += 1
	if episodeCounter>=EPISODES_UNTIL_RE_INITIALIZATION:
		parameters = np.random.rand(4)#Initialize parameters again
env.close()

print('Average reward', np.mean(total_reward_list))

plt.title('Total rewards')
plt.plot(total_reward_list)
print('Plot is displayed')
plt.show()
