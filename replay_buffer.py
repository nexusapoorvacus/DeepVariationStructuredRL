from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'next_state', 'attribute_actions', 'predicate_actions', 
					'next_object_actions', 'attribute_reward', 'predicate_reward', 
					'next_object_reward', 'next_state_attribute_actions', 
					'next_state_predicate_actions', 'next_state_next_object_actions', 'done'))

class ReplayMemory(object):

	def __init__(self, capacity, minimum_number_of_samples):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		self.minimum_number_samples = minimum_number_of_samples

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
