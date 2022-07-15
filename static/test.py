import numpy as np
from agent_1dim import DQNAgent
from env_1dim import MEDAEnv
import matplotlib.pyplot as plt


if __name__ == '__main__':
	env = MEDAEnv()
	best_score = -np.inf
	load_checkpoint = True
	agent = DQNAgent(gamma=0.99, epsilon=0, lr=0.0001, n_actions=env.action_space.n,
					 input_dims=env.shape, mem_size=50000, batch_size=32,
					 eps_min=0, replace=10000, eps_dec=1e-5, file_name='1dim_env', chkpt_dir='models')

	if load_checkpoint:
		agent.load_models()

	done = False
	observation = env.reset()
	
	img = plt.imshow(env.render())
	plt.savefig('start_observation')
	while not done:
		action = agent.choose_action(observation)
		observation, reward, observation_, done = env.step(action)
		print("demos/action:", action)
		print(observation)
		#img.set_data(env.render())
		img = plt.imshow(env.render())
	plt.savefig('demos/goal_observation')
