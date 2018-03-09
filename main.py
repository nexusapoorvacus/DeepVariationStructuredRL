from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataset import VGDataset
from models import VGG16, DQN
from operator import itemgetter
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from image_state import ImageState
from replay_buffer import ReplayMemory
from utils.vg_utils import entity_to_alias, predicate_to_alias, find_object_neighbors

import torch
import torch.nn as nn
import argparse
import json
import pickle

def train(model_vgg, model_frcnn, model_next_object_main, model_next_object_target, model_attribute_main, 
		model_attribute_target, model_predicate_main, model_predicate_target, replay_buffer, data_loader,
		optimizer_next_object, optimizer_attribute, optimizer_predicate, target_update_frequency, 
		semantic_action_graph):
	print("CUDA Available: " + str(torch.cuda.is_available()))
	# make model CUDA
	if torch.cuda.is_available():
		model = model.cuda()
		model_entites = model_next_object.cuda()
		model_predicate = model_predicate.cuda()
		model_attribute = model_attribute.cuda()

	# keeps track of current scene graphs for images
	image_states = {}
	
	for progress, (image, gt_scene_graph) in enumerate(data_loader):
		if torch.cuda.is_available():
			image = image.cuda()

		# get image features from VGG16
		images = model_vgg(image)

		# iterate through images in batch
		num_images_in_batch = image_state.size(0)	
		for idx in range(num_images_in_batch):
			# initializing image state if necessary
			image_name = gt_scene_graph[idx]["image_name"]
			if image_name not in image_states:	
				gt_sg = gt_scene_graph[idx]
				image_feature = images[idx]
				entity_proposals, entity_scores, entity_classes = model_frcnn.detect(image, 0.7)
				entity_features = []
				for box in entity_boxes:
					cropped_entity = crop_box(image_name, box), 
					box_feature = VGG16(cropped_entity, conv5_3_layer=True)
					box_feature.append(entity_features)
				im_state = ImageState(gt_sg["image_name"], gt_sg, image_feature, entity_features,
										entity_proposals, entity_classes, entity_scores)
				im_state.add_entities(entity_proposals, entity_classes, entity_scores)
				image_states[image_name] = im_state

			# get the image state object for image
			im_state = image_states[image_name]

			# compute state vector of image
			state_vector, subject_id, object_id = create_state_vector(im_state)

			# perform variation structured traveral scheme to get adaptive actions
			subject_name = entity_to_aliases(im_state.entity_classes[subject_id])
			object_name = entity_to_aliases(im_state.entity_classes[object_id])
			subject_bbox = im_state.entity[proposals[subject_id]]
			previously_mined_attributes = im_state.previously_mined_attributes[subject_name]
			previously_mined_next_objects = im_state.previously_mined_next_objects[subject_name]
			predicate_adaptive_actions, attribute_adaptive_actions = semantic_action_graph.variation_based_traveral(subject_name, 
																						object_name, previously_mined_attributes)
			next_object_adaptive_actions = find_object_neighbors(subject_bbox, im_state.entity_proposals, previously_mined_next_objects)
					

			entity_action_vectors = torch.from_numpy(np.identity())

def create_state_vector(image_state):
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

def choose_action_epsilon_greedy(state, adaptive_action_set, model, epsilon):
	sample = random.random()
	if sample > epsilon: # exploit
		return adaptive_action_set[model(state).data.max(1)]
	else: # explore
		return random.choice(adaptive_action_set)

def update_target(main_model, target_model):
	target_model.load_state_dict(main_model.state_dict())	

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
	DQN_next_object_main = DQN()
	DQN_next_object_target = DQN()
	DQN_predicate_main = DQN()
	DQN_predicate_target = DQN()
	DQN_attribute_main = DQN()
	DQN_attribute_target = DQN()

	# create shared optimizer
	# TODO: The paper says this optimizer is shared. Right now
	# the optimizers are not shared. Need to implement version
	# where it is shared
	optimizer_next_object = optim.RMSprop(DQN_next_object_main.parameters())
	optimizer_predicate = optim.RMSprop(DQN_predicate_main.parameters())
	optimizer_attribute = optim.RMSprop(DQN_attribute_main.parameters())

	# create replay buffer
	replay_buffer = ReplayMemory()

	# target update frequency
	target_update_frequency = 10000

	# load train data samples
	if args.train:
		train_data_samples = json.load(open(args.train_data))
		train_dataset = VGDataset(train_data_samples, args.images_dir)
		train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
									shuffle=True, num_workers=args.num_workers)
		train_images_state = train(model_vgg, model_frcnn, DQN_next_object_main, DQN_next_object_target,
						DQN_attribute_main, DQN_attribute_target, DQN_predicate_main, DQN_predicate_target,
						replay_buffer, data_loader=train_data_loader, optimizer_next_object,
						optimizer_attribute, optimizer_predicate, target_update_frequency)
