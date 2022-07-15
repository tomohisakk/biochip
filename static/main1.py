import numpy as np
from agent_1dim import DQNAgent
from env_1dim import MEDAEnv
import matplotlib.pyplot as plt

if __name__ == '__main__':
	env = MEDAEnv()
	best_score = -np.inf
	load_checkpoint = False
	n_games = 10000
	agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001, n_actions=env.action_space.n,
					 input_dims=env.shape, mem_size=50000, batch_size=32,
					 eps_min=0.01, replace=10000, eps_dec=1e-5, file_name='tmp', chkpt_dir='models')

	fname = agent.file_name + '_' + str(n_games) + 'games'
	figure_file = 'plots/' + fname + '.png'

	if load_checkpoint:
		agent.load_models()

	n_steps = 0
	scores, eps_history, steps_array = [], [], []

	for i in range(n_games):
		done = False
		score = 0
		observation = env.reset()
		img = plt.imshow(env.render())

		while not done:
			action = agent.choose_action(observation)
			observation, reward, observation_, done = env.step(action)
			img.set_data(env.render())
			plt.draw()
			plt.pause(0.0000001)
			score += reward

			if not load_checkpoint:
				agent.store_transition(observation, action, reward, observation_, int(done))	
				agent.learn()
			observation = observation_
			n_steps += 1
		scores.append(score)
		steps_array.append(n_steps)

		avg_score = np.mean(scores[-100:])
		if (i % 100 == 0):
			print('epsode ', i, 'score %.2f average score %.1f best score %.1f epsilon %.2f' %
				  (score, avg_score, best_score, agent.epsilon), 'steps ', n_steps)
	
		if avg_score > best_score:
			if not load_checkpoint:
				agent.save_models()
			best_score = avg_score
		eps_history.append(agent.epsilon)


	fig, ax1 = plt.subplots()
	ax1.plot(scores, color = 'red')
	ax2 = ax1.twinx()
	ax2.plot(eps_history, color = 'blue')
	plt.savefig(figure_file)
