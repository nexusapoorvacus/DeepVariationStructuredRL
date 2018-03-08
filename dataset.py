from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class VGDataset(Dataset):

	def __init__(self, image_data, image_dir):
		self.image_data = image_data
		self.image_dir = image_dir
		self.transform_vgg = transforms.Compose([
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		self.transform_frcnn = transforms.Compose([
			transforms.Resize((600, 600)),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	def __len__(self):
		return len(self.image_data)

	def __getitem__(self, index):
		image_dict = self.image_data[index]
		image_name = image_dict["image_name"]
		image = Image.open(self.image_dir + image_name)
		image = self.transform_vgg(image).unsqueeze(0)
		return image, image_dict
