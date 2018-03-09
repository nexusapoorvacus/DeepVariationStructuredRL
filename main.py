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
					
			# creating state + action vectors to feed in DQN
			attribute_state_vectors = torch.concat([state_vector, torch.from_numpy(np.identity(len(attribute_adaptive_actions)))], 1)
			predicate_state_vectors = torch.concat([state_vector, torch.from_numpy(np.identity(len(predicate_adaptive_actions)))], 1)
			next_object_state_vectors = torch.concat([state_vector, torch.from_numpy(np.identity(len(next_object_adaptive_actions)))], 1)

			# choose action using epsilon greedy
			attribute_action = choose_action_epsilon_greedy(attribute_state_vectors, attribute_adaptive_actions, model_attribute_main,
															epsilon, training=replay_buffer.can_sample())
			predicate_action = choose_action_epsilon_greedy(predicate_state_vectors, predicate_adaptive_actions, model_predicate_main,
															epsilon, training=replay_buffer.can_sample())
			next_object_action = choose_action_epsilon_greedy(next_object_state_vectors, next_object_adaptive_actions, model_next_object_main,
															epsilon, training=replay_buffer.can_sample())

			# step image_state
			attribute_reward, predicate_reward, next_object_reward, done = image_state.step(attribute_action, predicate_action, next_object_action)
			next_state = create_state_vector(im_state)		
	
			# decay epsilon
			epsilon = epsilon * epsilon_decay

			# add transition tuple to replay buffer
			replay_buffer.push(state_vector, attribute_adaptive_actions, predicate_adaptive_actions, next_object_adaptive_actions, next_state, attribute_reward, 
								predicate_reward, next_object_reward, done)

			# sample minibatch if replay_buffer has enough samples
			if replay_buffer.can_sample():
				minibatch_transitions = replay_buffer.sample(minibatch_size)
				main_q_attribute_list, main_q_predicate_list, main_q_next_object_list = [], [], []
				target_q_attribute_list, target_q_predicate_list, target_q_next_object_list = [], [], []
				for transition in minibatch_transitions:
					target_q_attribute, target_q_predicate, target_q_next_object = None, None, None
					if transition.done:
						target_q_attribute = transition.attribute_reward
						target_q_predicate = transition.predicate_reward
						target_q_next_object = transition.target_q_next_object
					else:
						next_state_attribute = torch.concat([transition.next_state, 
														torch.from_numpy(np.identity(len(transition.next_state_attribute_actions)))], 1)
						next_state_predicate = torch.concat([transition.next_state, 
														torch.from_numpy(np.identity(len(transition.next_state_predicate_actions)))], 1)
						next_state_next_object = torch.concat([transition.next_state, 
														torch.from_numpy(np.identity(len(transition.next_state_next_object_actions)))], 1)
						target_q_attribute = transition.attribute_reward + gamma * torch.max(model_attribute_target(next_state_attribute))
						target_q_predicate = transition.predicate_reward + gamma * torch.max(model_predicate_target(next_state_predicate))
						target_q_next_object = transition.next_object_reward + gamma * torch.max(model_next_object_target(next_state_next_object))
					# compute loss
					main_state_attribute = torch.concat([transition.state, 
														torch.from_numpy(np.identity(len(transition.attribute_actions)))], 1)
					main_state_predicate = torch.concat([transition.state, 
														torch.from_numpy(np.identity(len(transition.predicate_actions)))], 1)
					main_state_next_object = torch.concat([transition.state, 
														torch.from_numpy(np.identity(len(transition.next_object_actions)))], 1)
					main_q_attribute = transition.attribute_reward + gamma * torch.max(model_attribute(main_state_attribute))
					main_q_predicate = transition.predicate_reward + gamma * torch.max(model_predicate(main_state_predicate))
					main_q_next_object = transition.next_object_reward + gamma * torch.max(model_next_object(main_state_next_object))
					
					# add to q value lists
					target_q_attribute_list.append(target_q_attribute)
					target_q_predicate_list.append(target_q_predicate)
					target_q_next_object_list.append(target_q_next_object)

					main_q_attribute_list.append(main_q_attribute)
					main_q_predicate_list.append(main_q_predicate)
					main_q_next_object_list.append(main_q_next_object)

				# calculate loss and optimize model
				loss_attribute = loss_fn_attribute(torch.FloatTensor(main_q_attribute_list), torch.FloatTensor(target_q_attribute_list))
				loss_predicate = loss_fn_predicate(torch.FloatTensor(main_q_predicate_list), torch.FloatTensor(target_q_predicate_list))
				loss_next_object = loss_fn_next_object(torch.FloatTensor(main_q_next_object), torch.FloatTensor(target_q_next_object))
				
				optimizer_attribute.zero_grad()	
				optimizer_predicate.zero_grad()
				optimizer_next_object.zero_grad()

				loss_attribute.backward()
				loss_predicate.backward()
				loss_next_object.backward()

				for param in model_attribute.parameters:
					param.grad.data.clamp_(-1, 1)
				for param in model_predicate.parameters:
					param.grad.data.clamp_(-1, 1)
				for param in model_next_object.parameters:
					param.grad.data.clamp_(-1, 1)

				optimizer_attibute.step()
				optimizer_predicate.step()
				optimizer_next_object.step()




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

def choose_action_epsilon_greedy(state, adaptive_action_set, model, epsilon, training=False):
	sample = random.random()
	if sample > epsilon and training: # exploit
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
