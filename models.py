import torch
import torch.nn as nn
import torchvision.models as models

class VGG16(nn.Module):
	def __init__(self):
		super(VGG16, self).__init__()
		resnet = models.resnet50(pretrained=True)
		modules = list(resnet.children())[:-1] # delete the last fc layer.
    		self.model = nn.Sequential(*modules)
    		for param in self.model.parameters():
			param.requires_grad = False
    
	def forward(self, x):
        	return self.model(x)

class DQN(nn.Module):
	"""
	Adapted from:
	http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
	"""

	def __init__(self):
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)
		self.output = nn.Linear(448, 1)

	def forward(self, x):
		x = nn.functional.relu(self.bn1(self.conv1(x)))
		x = nn.functional.relu(self.bn2(self.conv2(x)))
		x = nn.functional.relu(self.bn3(self.conv3(x)))
		return self.output(x.view(x.size(0), -1))
