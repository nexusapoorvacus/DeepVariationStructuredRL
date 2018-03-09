from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		self.minimum_number_samples = 500

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.capacity:
			self.memory.append(None)
			self.memory[self.position] = Transition(*args)
			self.position = (self.position + 1) % self.capacity
			return True
		else:
			return False

	def can_sample(self):
		return len(self.memory) > self.minimum_number_samples 

	def sample(self, batch_size):
		if self.can_sample():
			return random.sample(self.memory, batch_size)
		return False

	def __len__(self):
		return len(self.memory)
