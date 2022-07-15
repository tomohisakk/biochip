from pickletools import uint8
from gym import spaces
import numpy as np
import math
import copy

MAPS = {
	"4*4": np.array([
		[1, 0, 0, 0],
		[0, 0, 2, 0],
		[0, 2, 0, 0],
		[0, 0, 0, 3]
	]),
	"8*8": np.array([
		[1, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 2, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 2, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 0],
		[2, 0, 0, 0, 0, 0, 0, 0],
		[0, 0, 0, 0, 0, 0, 0, 3],
	])
}

class MEDAEnv:
	def __init__(self):
		self.action_meaning = {
			0: "N",
			1: "E",
			2: "S",
			3: "W",
			4: "NE",
			5: "SE",
			6: "SW",
			7: "NW",
		}
		self.action_space = spaces.Discrete(len(self.action_meaning))
		self.observation = copy.copy(MAPS["8*8"])
		self.goal = (self.width - 1, self.height - 1)
		self.start = (0, 0)
		self.state = self.start
		self.unmove_count = 0
		self.shape = (1, self.width, self.height)
	@property
	def height(self):
		return len(self.observation)

	@property
	def width(self):
		return len(self.observation[0])

	def actions(self):
		return self.action_space

	def states(self):
		for h in range(self.height):
			for w in range(self.width):
				yield (h, w)

	def next_state(self, state, action):
		action_move_map = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
		move = action_move_map[action]
		state_ = (state[0] + move[0], state[1] + move[1])
		ny, nx = state_

		if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
			state_ = state
			
		elif self.observation[state_] == 2:
			state_ = state

		elif state[0] != state_[0] and state[1] != state_[1] and ((self.observation[state[0], state_[1]] == (2 or 3)) or\
					self.observation[state_[0], state[1]] == (2 or 3)):
			state_ = state

		self.observation[state] = 2
		if action >= 4:
			self.observation[state[0]][state_[1]] = 2
			self.observation[state_[0]][state[1]] = 2
		self.observation[state_] = 1

		return state_

	def compute_distance(self, state):
		delta_x = state[0] - self.goal[0]
		delta_y = state[1] - self.goal[1]
		return math.sqrt(delta_x**2 + delta_y**2)

	def reward(self, state, action, next_state):
		if (next_state == self.goal):
			return (1)

		reward = 0
		state_distance = self.compute_distance(state)
		next_state_distance = self.compute_distance(next_state)
		diagnal_param = 3 if action >= 4 else 1
		if (state_distance == next_state_distance):
			reward = -0.3
		elif (state_distance > next_state_distance):
			reward = 0.5 / diagnal_param
		elif (state_distance < next_state_distance):
			reward = -0.8

		return reward

	def reset(self):
		self.state = (0, 0)
		self.observation = copy.copy(MAPS["8*8"])
		return (self.observation)

	def step(self, action):
		state = self.state
		observation = self.observation
		state_ = self.next_state(state, action)
		reward = self.reward(state, action, state_)
		done = (state_ == self.goal or self.unmove_count == 10)
		if self.state == state_:
			self.unmove_count += 1
		if self.state != state_:
			self.unmove_count = 0
		self.state = state_
		observation_ = self.observation
		return observation, reward, observation_, done

	def render(self):
		img = self.observation
		img = np.zeros(shape = (self.height, self.width, 3), dtype=int)

		for y in range(self.width):
			for x in range(self.height):
				if np.array_equal(self.observation[y][x], 3): #red
					img[y][x] = [255, 0, 0]
				elif np.array_equal(self.observation[y][x],2): #gre
					img[y][x] = [0, 255, 0]
				elif np.array_equal(self.observation[y][x],1): #blu
					img[y][x] = [0, 0, 255]
				else: # grey
					img[y][x] = [255, 255, 255]
		return img

	def close(self):
		pass