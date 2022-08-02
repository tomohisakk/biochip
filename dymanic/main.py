import gym
import numpy as np
from agent import DQNAgent
from env02 import MEDAEnv
import matplotlib.pyplot as plt
from random import randint


if __name__ == '__main__':
	env = MEDAEnv(w=8, l=8)
	best_score = -np.inf
	load_checkpoint = False
	n_games = 50000
	agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
					 input_dims=env.observation_space.shape,
					 n_actions=4, mem_size=50000, eps_min=0.1,
					 batch_size=64, replace=1000, eps_dec=1e-6,
					 chkpt_dir='models/', env_name='batchsize_August1')

	if load_checkpoint:
		agent.load_models()

	fname = agent.env_name + '_' + str(n_games) + 'games'
	figure_file = 'plots/' + fname + '.png'

	n_steps = 0
	scores, eps_history, steps_array = [], [], []
	n_modules = 0

	for i in range(n_games):
		#print(i)
		n_modules = randint(0,8)
		#print(n_modules)
		done = False
		score = 0
		observation = env.reset(n_modules)
		#print(observation)

		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, _ = env.step(action)
			score += reward

			if not load_checkpoint:
				agent.store_transition(observation, action, reward, observation_, done)
				#print(observation)
				agent.learn()
			observation = observation_
			n_steps += 1
		scores.append(score)
		steps_array.append(n_steps)

		avg_score = np.mean(scores)
		if (i % 100 == 0):
			print('epsode ', i, 'average score %.1f best score %.1f epsilon %.2f' % (avg_score, best_score, agent.epsilon))

		if avg_score > best_score:
			if not load_checkpoint:
				agent.save_models()
			best_score = avg_score

		eps_history.append(agent.epsilon)

	fig, ax1 = plt.subplots()
	ax1.plot(scores, color = 'red')
	ax1.set_xlabel("Training Steps")
	ax1.set_ylabel("Rewards", color = 'red')

	ax2 = ax1.twinx()
	ax2.plot(eps_history, color = 'blue')
	ax2.set_ylabel("epsilon", color = 'blue')
	plt.savefig(figure_file)
