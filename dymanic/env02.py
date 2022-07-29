import copy
import queue
import random
import numpy as np
from enum import IntEnum
from gym import spaces
import random

class Actions(IntEnum):
	N = 0
	E = 1
	S = 2
	W = 3
	NE = 4
	SE = 5
	SW = 6
	NW = 7

class Module:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def is_point_inside(self, point):
		if point[1] == self.y and point[0] == self.x:
			return True
		else:
			return False

	def _is_lines_overlap(self, xa_1, xa_2, xb_1, xb_2):
		if xa_1 > xb_2:
			return False
		elif xb_1 > xa_2:
			return False
		else:
			return True

	def is_module_overlap(self, m):
		if self.x == m.x and self.y == m.y:
			return True
		else:
			return False

class MEDAEnv():
	def __init__(self, w, l, b_random = False, n_modules = 5):
		super(MEDAEnv, self).__init__()
		assert w > 0 and l > 0
		self.width = w
		self.length = l
		self.actions = Actions
		self.action_space = spaces.Discrete(len(self.actions))
		#self.action_space = 4
		self.observation_space = spaces.Box(low = 0, high = 1, shape = (1, w, l), dtype = 'uint8')
		self.b_random = b_random
		self.unmove_count = 0
		self.step_count = 0
		self.max_step = 5
		self.m_usage = np.zeros((l, w))
		self.m_health = np.zeros((l, w))

		if b_random:
			self.state, self.goal = self._random_sart_n_end()
		else:
			self.state = (0, 0)
			self.goal = (w-1, l-1)
		self.start = copy.deepcopy(self.state)
		self.modules = self._gen_random_modules(n_modules)
		self.m_distance = self._compute_dist()
		self.obs = self._get_obs()

	def step(self, action):
		done = False
		self.step_count += 1
		prev_dist = self._get_dist()
		self._update_position(action)
		curr_dist = self._get_dist()
		#print("prev_dist:", prev_dist)
		#print("curent_dist:", curr_dist)
		obs = self._get_obs()
		diagnal_param = 3 if action >= 4 else 1
		if self._is_complete():
			reward = 1.0 / diagnal_param
			done = True
		elif self.unmove_count > self.max_step:
			reward = -0.8
			self.unmove_count = 0
			done = True
		elif prev_dist > curr_dist:
			reward = 0.5 / diagnal_param
			self.unmove_count = 0
		elif prev_dist == curr_dist:
			reward = -0.3 #change 0726
			self.unmove_count += 1
		else:
			self.unmove_count = 0
			reward = -0.3 #change 0722
		#print(self.max_step)
		#print(self.step_count)
		return obs, reward, done, {}

	def reset(self, n_modules):
		self.step_count = 0
		if self.b_random is True:
			self.state, self.goal = self._random_sart_n_end()
		else:
			self.state, self.goal = self._get_next_sart_n_end()
		self.start = copy.deepcopy(self.state)
		self.m_health = np.zeros((self.length, self.width))
		self.m_usage = np.zeros((self.length, self.width))
		self.modules = self._gen_random_modules(n_modules)
		self.m_distance = self._compute_dist()
		obs = self._get_obs()
		return obs

	def _gen_random_modules(self, n_modules):
		""" Generate reandom modules up to n_modules"""
		if self.width < 5 or self.length < 5:
			return []
		modules = []
		for i in range(n_modules):
			x = random.randrange(0, self.width - 1)
			y = random.randrange(0, self.length - 1)
			m = Module(x, y)
			while m.is_point_inside(self.state) or m.is_point_inside(self.goal) or \
					self._is_module_overlap(m, modules):
				x = random.randrange(0, self.length - 1)
				y = random.randrange(0, self.width - 1)
				m = Module(x, y)
			modules.append(m)
		return modules

	def _is_module_overlap(self, m, modules):
		for mdl in modules:
			if mdl.is_module_overlap(m):
				return True
		return False

	def _compute_dist(self):
		m_dist = np.zeros(
				shape = (self.length, self.width),
				dtype = np.uint8)
		q = queue.Queue()
		q.put(self.goal)
		m_dist[self.goal[1]][self.goal[0]] = 1
		self._set_modules_with_value(m_dist, np.iinfo(np.uint8).max)
		while not q.empty():
			q, m_dist = self._update_queue(q, m_dist)
		return m_dist

	def _set_modules_with_value(self, m_dist, v):
		for m in self.modules:
			m_dist[m.x][m.y] = v
		return

	def _update_queue(self, q, m_dist):
		head = q.get()
		dist = m_dist[head[1]][head[0]]
		neighbors = self._get_neighbors(head)
		for n in neighbors:
			if m_dist[n[1]][n[0]] == 0:
				q.put(n)
				m_dist[n[1]][n[0]] = dist + 1
		return q, m_dist

	def _get_neighbors(self, p):
		neighbors = [
				(p[0] - 1, p[1]),
				(p[0] + 1, p[1]),
				(p[0], p[1] - 1),
				(p[0], p[1] + 1)]
		return [n for n in neighbors if self._is_point_inside(n)]

	def _random_sart_n_end(self):
		x = random.randrange(0, self.length)
		y = random.randrange(0, self.width)
		start = (y, x)
		repeat = random.randrange(0, self.length * self.width)
		for i in range(repeat):
			x = random.randrange(0, self.length)
			y = random.randrange(0, self.width)
		end = (y, x)
		while end == start:
			x = random.randrange(0, self.length)
			y = random.randrange(0, self.width)
			end = (y, x)
		return start, end

	def _get_next_sart_n_end(self):
		start = self.start
		end = self.goal
		return start, end

	def _get_dist(self):
		y = self.state[0]
		x = self.state[1]
		#print(self.m_distance)
		return self.m_distance[y][x]

	def _update_position(self, action):
		next_p = list(self.state)

		if action == Actions.N:
			next_p[1] -= 1
		elif action == Actions.E:
			next_p[0] += 1
		elif action == Actions.S:
			next_p[1] += 1
		elif action == Actions.W:
			next_p[0] -= 1
		elif action == Actions.NE:
			next_p[1] -= 1
			next_p[0] += 1
		elif action == Actions.SE:
			next_p[1] += 1
			next_p[0] += 1
		elif action == Actions.SW:
			next_p[1] += 1
			next_p[0] -= 1
		else:
			next_p[1] -= 1
			next_p[0] -= 1
		if not self._is_point_inside(next_p):
			return
		elif self._is_touching_module(next_p, action):
			return 
		elif random.randint(1, 10) == 5:
			self.m_usage[next_p[1]][next_p[0]] += 1
			return
		else:
			if action >= 4:
				self.m_usage[self.state[1]][next_p[0]] += 1
				self.m_usage[next_p[1]][self.state[0]] += 1
			self.state = tuple(next_p)
			self.m_usage[next_p[1]][next_p[0]] += 1
		return

	def _is_point_inside(self, point):
		if point[1] < 0 or point[1] >= self.length:
			return False
		if point[0] < 0 or point[0] >= self.width:
			return False
		return True

	def _is_touching_module(self, point, action):
		if self.m_health[point[1]][point[0]]:
			#print("touch direct")
			return True
		if action >= 4:
			#print("state: ",self.state)
			#print("point:", point)
			#print(self.state[1], point[0])
			#print(self.m_health[self.state[1]][point[0]])
			#print(point[1],self.state[0])
			#print(self.m_health[point[1]][self.state[0]])
			if self.m_health[self.state[1]][point[0]] or \
				self.m_health[point[1]][self.state[0]]:
				#print("touch diagnal")
				return True
		return False

	def _is_complete(self):
		if self.state == self.goal:
			return True
		else:
			return False

	def _get_obs(self):
		"""
		none: 0
		degrade: 1
		state: 3
		goal: 2
		"""

		obs = np.zeros((self.length, self.width))
		obs = self._add_modules_in_obs(obs)

		index = self.m_usage > 0
		self.m_health = np.add(index.astype(int), obs)
		self.m_health[self.start[1]][self.start[0]] = 1
		obs = np.copy(self.m_health)
		#print(m_health)
		#obs[self.m_health[1].astype(int)][self.m_health[0].astype(int)] = 1
		obs[self.goal[1]][self.goal[0]] = 2
		obs[self.state[1]][self.state[0]] = 3

		return obs

	def _add_modules_in_obs(self, obs):
		for m in self.modules:
			obs[m.y][m.x] = 1
		return obs

	def render(self):
		obs = self._get_obs()
		img = np.zeros(shape = (self.width, self.length, 3), dtype=int)
		for y in range(self.width):
			for x in range(self.length):
				if np.array_equal(obs[y][x], 1):
					img[y][x] = [0, 0, 255]
				elif np.array_equal(obs[y][x], 2):
					img[y][x] = [255, 0, 0]
				elif  np.array_equal(obs[y][x], 3):
					img[y][x] = [0, 255, 0]
				
				else: # grey
					img[y][x] = [255, 255, 255]

		return img

"""
if __name__ == '__main__':
	env = MEDAEnv(w=8, l=8)
	done = 0
	while(done==0):
		action = np.random.randint(0,4)
		obs, reward, done, _ = env.step(action)
		#env.reset(n_modules=5)
		#print("action and obs")
		print(action)
		print(obs)
		print(reward)
	print("--------------SECOND RAUND----------")
	env.reset(n_modules=5)
	done = 0
	while(done==0):
		action = np.random.randint(0,4)
		obs, reward, done, _ = env.step(action)
		#env.reset(n_modules=5)
		#print("action and obs")
		print(action)
		print(obs)
		print(reward)
"""