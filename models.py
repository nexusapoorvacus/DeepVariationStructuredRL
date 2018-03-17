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

class DQN_MLP(nn.Module):

	def __init__(self, input_size, output_size, hidden_size=300, num_hidden_layers=5):
		super(DQN_MLP, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_hidden_layers = num_hidden_layers
		
		layers = []
		layers.append(nn.Linear(input_size, hidden_size))
		layers.append(nn.ReLU())
		for i in range(num_hidden_layers):
			layers.append(nn.Linear(hidden_size, hidden_size))
			layers.append(nn.ReLU())
		layers.append(nn.Linear(hidden_size, output_size))
		self.layers = nn.Sequential(*layers)
		self.softmax = nn.Softmax()

	def forward(self, x):
		return self.softmax(self.layers(x))




