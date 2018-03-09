from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataset import VGDataset
from models import VGG16, DQN
from operator import itemgetter
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from image_state import ImageState

import torch
import torch.nn as nn
import argparse
import json
import pickle

def train(model_vgg, model_frcnn, model_entities, model_predicate, model_attributes, data_loader, criterion):
	print("CUDA Available: " + str(torch.cuda.is_available()))
	# make model CUDA
	if torch.cuda.is_available():
		model = model.cuda()
		model_entites_
	# dictionary to keep track of image states
	image_states = {}

	for progress, (image, gt_scene_graph) in enumerate(data_loader):
		if torch.cuda.is_available():
			image = image.cuda()
		images = model_vgg(image)
		# iterate through images in batch
		num_images_in_batch = image_state.size(0)	
		for idx in range(num_images_in_batch):
			# initializing image state
			gt_sg = gt_scene_graph[idx]
			image_feature = images[idx]
			entity_bboxes, entity_scores, entity_classes = model_frcnn.detect(image, 0.7)
			entity_features = []
			for box in entity_boxes:
				cropped_entity = crop_box(image_name, box), 
				box_feature = VGG16(cropped_entity, conv5_3_layer=True)
				box_feature.append(entity_features)
			im_state = ImageState(gt_sg["image_name"], gt_sg)
			im_state.add_entities(entity_proposals)
			
			state_vector = create_state_vector(image, entity_features, entity_scores)
			# perform variation structured traveral scheme
					

			entity_action_vectors = torch.from_numpy(np.identity())

def create_state_vector(image_state, image_feature, entity_features, entity_scores, curr_subject=None):
	# find subject to start with if curr_subject is None
	if curr_subject == None:
		indices, sorted_scores = zip(*sorted(enumerate(entity_scores), key=iitemgetter(1)))
		for idx in indices:
			if idx not in image_state.explored_entities:
				curr_subject = torch.from_numpy(entity_features[idx])
		if curr_subject == None:
			return False
	else:
		curr_subject = torch.from_numpy(entity_features[curr_subject])
	return torch.cat([image_feature, curr_subject], 1)


if __name__=='__main__':
	# flags
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_data", type=str,
				default="data/data_samples/train_data.json",
				help='Location of the file containing train data samples')
	parser.add_argument("--images_dir", type=str,
				default="data/images/",
				help="Location of Visual Genome images")
	parser.add_argument("--train", help="trains model", action="store_true")
	parser.add_argument("--test", help="evaluates model", action="store_true")
	parser.add_argument("--batch_size", type=int, default=64, help="batch size to use")
	parser.add_argument("--num_workers", type=int, default=4, help="number of threads")
	args = parser.parse_args()

	# create semantic action graph
	semantic_action_graph = pickle.load(open("graph.pickle", "rb"))
	
	# create VGG model for state featurization
	model_vgg = VGG16()

	# create Faster-RCNN model for state featurization
	model_file = 'VGGnet_fast_rcnn_iter_70000.h5'
	model_frcnn = FasterRCNN()
	network.load_net(model_file, model_frcnn)
	model_frcnn.cuda()
	model_frcnn.eval()

	# create DQN's for the next object, predicates, and attributes
	DQN_next_object = DQN()
	DQN_predicate = DQN()
	DQN_attribute = DQN()

	# load train data samples
	if args.train:
		train_data_samples = json.load(open(args.train_data))
		train_dataset = VGDataset(train_data_samples, args.images_dir)
		train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
									shuffle=True, num_workers=args.num_workers)
		train_images_state = train(model_vgg, model_frcnn, DQN_next_object, , DQN_predicate, DQN_attribute, data_loader=train_data_loader)
