import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedVGG16(nn.Module):

	def __init__(self):
		super(PretrainedVGG16, self).__init__()
		vgg16 = models.vgg16(pretrained=True)
		 
