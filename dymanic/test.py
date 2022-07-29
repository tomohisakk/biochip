import numpy as np
from agent import DQNAgent
from env02 import MEDAEnv
import matplotlib.pyplot as plt

if __name__ == '__main__':
	env = MEDAEnv(w=8, l=8)
	best_score = -np.inf
	load_checkpoint = True
	n_games = 5
	agent = DQNAgent(gamma=0.99, epsilon=0, lr=0.0001,
					 input_dims=env.observation_space.shape,
					 n_actions=4, mem_size=50000, eps_min=0,
					 batch_size=32, replace=10000, eps_dec=5e-8,
					 chkpt_dir='models/', env_name='ngames_batchsize_umove_reward_July29')

	agent.load_models()
	n_modules = 2

	for i in range(n_games):
		done = False
		score = 0
		observation = env.reset(n_modules)
		s_file = 'demos/start_observation' + str(i)
		f_file = 'demos/final_observation' + str(i)
		plt.imshow(env.render())
		plt.savefig(s_file)
		print()
		print('---------start game', i, '-----------')
		while not done:
			action = agent.choose_action(observation)
			print(action)
			observation_, reward, done, _ = env.step(action)
			score += reward
			observation = observation_
			print(observation)
			plt.imshow(env.render())
			plt.savefig(f_file)
		