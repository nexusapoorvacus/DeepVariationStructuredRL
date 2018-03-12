from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataset import VGDataset, collate
from models import VGG16, DQN, DQN_MLP
from operator import itemgetter
from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from image_state import ImageState
from replay_buffer import ReplayMemory
from utils.vg_utils import entity_to_aliases, predicate_to_aliases, find_object_neighbors, crop_box
from PIL import Image

import torch
import torch.nn as nn
import argparse
import json
import dill
import pickle
import numpy as np
import random

def train(parameters):
	print("CUDA Available: " + str(torch.cuda.is_available()))
	# make model CUDA
	if torch.cuda.is_available():
		model_VGG = model_vgg.cuda()
		model_FRCNN = model_frcnn.cuda()
		model_next_object_main = DQN_next_object_main.cuda()
		model_next_object_target = DQN_next_object_target.cuda()	
		model_attribute_main = DQN_attribute_main.cuda()
		model_attribute_target = DQN_attribute_target.cuda()
		model_predicate_main = DQN_predicate_main.cuda()
		model_predicate_target = DQN_predicate_target.cuda()

	# keeps track of current scene graphs for images
	image_states = {}
	total_number_timesteps_taken = 0

	for epoch in range(num_epochs):
		print("Epoch: ", epoch)
		for progress, (images, images_orig, gt_scene_graph) in enumerate(train_data_loader):
			images = torch.autograd.Variable(torch.squeeze(images, 1))
			if torch.cuda.is_available():
				images = images.cuda()
			# get image features from VGG16
			images = model_VGG(images)
	
			# iterate through images in batch
			for idx in range(images.size(0)):
				# initializing image state if necessary
				image_name = gt_scene_graph[idx]["image_name"]
				if image_name not in image_states:
					gt_sg = gt_scene_graph[idx]
					image_feature = images[idx]
					entity_proposals, entity_scores, entity_classes = model_FRCNN.detect(images_orig[idx], object_detection_threshold)
					# 
					entity_proposals = entity_proposals[:maximum_num_entities_per_image]
					entity_scores = entity_scores[:maximum_num_entities_per_image]
					entity_classes = entity_classes[:maximum_num_entities_per_image]
					if len(entity_scores) < 2:
						continue

					entity_features = []
					for box in entity_proposals:
						cropped_entity = crop_box(images_orig[idx], box)
						cropped_entity = torch.autograd.Variable(cropped_entity)
						if torch.cuda.is_available():
							cropped_entity = cropped_entity.cuda()
						box_feature = model_VGG(cropped_entity)
						entity_features.append(box_feature)
					im_state = ImageState(gt_sg["image_name"], gt_sg, image_feature, entity_features,
											entity_proposals, entity_classes, entity_scores, semantic_action_graph)
					im_state.initialize_entities(entity_proposals, entity_classes, entity_scores)
					image_states[image_name] = im_state
				else:
					# reset image state from last epoch
					image_states[image_name].reset()
				im_state = image_states[image_name]	
				while not im_state.is_done():
					print("Iter for image " + str(image_name))
					# get the image state object for image
					im_state = image_states[image_name]
	
					print("Computing state vector")
					# compute state vector of image
					state_vector = create_state_vector(im_state)
					subject_id = im_state.current_subject
					object_id = im_state.current_object
					if type(state_vector) == type(None):
						if im_state.current_subject == None:
							break
						else:
							im_state.explored_entities.append(im_state.current_subject)
							im_state.current_subject = None
							im_state.current_object = None
							continue

					# perform variation structured traveral scheme to get adaptive actions
					print("Creating adaptive action sets...")
					subject_name = entity_to_aliases(im_state.entity_classes[subject_id])
					object_name = entity_to_aliases(im_state.entity_classes[object_id])
					subject_bbox = im_state.entity_proposals[subject_id]
					previously_mined_attributes = im_state.current_scene_graph["objects"][subject_id]["attributes"]
					previously_mined_next_objects = im_state.objects_explored_per_subject[subject_id]
					attribute_adaptive_actions, predicate_adaptive_actions = semantic_action_graph.variation_based_traversal(subject_name, object_name, previously_mined_attributes)
					next_object_adaptive_actions = find_object_neighbors(subject_bbox, im_state.entity_proposals, previously_mined_next_objects)
						
					# creating state + action vectors to feed in DQN
					print("Creating state + action vectors to pass into DQN...")
					attribute_state_vectors = create_state_action_vector(state_vector, attribute_adaptive_actions, len(semantic_action_graph.attribute_nodes))
					predicate_state_vectors = create_state_action_vector(state_vector, predicate_adaptive_actions, len(semantic_action_graph.predicate_nodes))
					next_object_state_vectors = create_state_action_vector(state_vector, next_object_adaptive_actions, parameters["maximum_num_entities_per_image"])
		
					# choose action using epsilon greedy
					print("Choose action using epsilon greedy...")
					attribute_action, predicate_action, next_object_action = None, None, None
					if type(attribute_state_vectors) != type(None):
						attribute_action = choose_action_epsilon_greedy(attribute_state_vectors, attribute_adaptive_actions, model_attribute_main, parameters["epsilon"], training=replay_buffer.can_sample())
					if type(predicate_state_vectors) != type(None):
						predicate_action = choose_action_epsilon_greedy(predicate_state_vectors, predicate_adaptive_actions, model_predicate_main, parameters["epsilon"], training=replay_buffer.can_sample())
					if type(next_object_state_vectors) != type(None):
						next_object_action = choose_action_epsilon_greedy(next_object_state_vectors, next_object_adaptive_actions, model_next_object_main, parameters["epsilon"], training=replay_buffer.can_sample())
					# step image_state
					print("Step state environment using action...")
					attribute_reward, predicate_reward, next_object_reward, done = im_state.step(attribute_action, predicate_action, next_object_action)
					print("Rewards(A,P,O)", attribute_reward, predicate_reward, next_object_reward)
					next_state = create_state_vector(im_state)		
					im_state = image_states[image_name]
					# decay epsilon
					if parameters["epsilon"] > parameters["epsilon_end"]:
						parameters["epsilon"] = parameters["epsilon"] * parameters["epsilon_anneal_rate"]
	
					# add transition tuple to replay buffer
					print("Adding transition tuple to replay buffer...")
					subject_name_1 =  entity_to_aliases(im_state.entity_classes[im_state.current_subject])
					object_name_1 = entity_to_aliases(im_state.entity_classes[im_state.current_object])
					previously_mined_attributes_1 = im_state.current_scene_graph["objects"][im_state.current_subject]["attributes"]
					previously_mined_next_objects_1 = im_state.objects_explored_per_subject[im_state.current_subject]
					attribute_adaptive_actions_1, predicate_adaptive_actions_1 = semantic_action_graph.variation_based_traversal(subject_name_1, object_name_1, previously_mined_attributes)
					next_object_adaptive_actions_1 = find_object_neighbors(im_state.entity_proposals[im_state.current_subject], im_state.entity_proposals, previously_mined_next_objects)

					replay_buffer.push(state_vector, next_state, attribute_adaptive_actions, predicate_adaptive_actions, next_object_adaptive_actions, attribute_reward, predicate_reward, next_object_reward, attribute_adaptive_actions_1, predicate_adaptive_actions_1, next_object_adaptive_actions_1, done)
	
					# sample minibatch if replay_buffer has enough samples
					if replay_buffer.can_sample():
						print("Sample minibatch of transitions...")
						minibatch_transitions = replay_buffer.sample(parameters["batch_size"])
						main_q_attribute_list, main_q_predicate_list, main_q_next_object_list = [], [], []
						target_q_attribute_list, target_q_predicate_list, target_q_next_object_list = [], [], []
						for transition in minibatch_transitions:
							target_q_attribute, target_q_predicate, target_q_next_object = None, None, None
							if transition.done:
								target_q_attribute = transition.attribute_reward
								target_q_predicate = transition.predicate_reward
								target_q_next_object = transition.target_q_next_object
							else:
								next_state_attribute = create_state_action_vector(transition.next_state, transition.next_state_attribute_actions, len(semantic_action_graph.attribute_nodes))
								next_state_predicate = create_state_action_vector(transition.next_state, transition.next_state_predicate_actions, len(semantic_action_graph.predicate_nodes))
								next_state_next_object = create_state_action_vector(transition.next_state, transition.next_state_next_object_actions, parameters["maximum_num_entities_per_image"])
								if type(next_state_attribute) != type(None):
									next_state_attribute.volatile = True
									output = torch.max(model_attribute_target(next_state_attribute))
									target_q_attribute = transition.attribute_reward + parameters["discount_factor"] * output
								if type(next_state_predicate) != type(None):
									next_state_predicate.volatile = True
									target_q_predicate = transition.predicate_reward + parameters["discount_factor"] * torch.max(model_predicate_target(next_state_predicate))
								if type(next_state_next_object) != type(None):
									next_state_next_object.volatile = True
									target_q_next_object = transition.next_object_reward + parameters["discount_factor"] * torch.max(model_next_object_target(next_state_next_object))
							# compute loss
							main_state_attribute = create_state_action_vector(transition.state, transition.attribute_actions, len(semantic_action_graph.attribute_nodes))
							main_state_predicate = create_state_action_vector(transition.state, transition.predicate_actions, len(semantic_action_graph.predicate_nodes))
							main_state_next_object = create_state_action_vector(transition.state, transition.next_object_actions, parameters["maximum_num_entities_per_image"])

							main_q_attribute, main_q_predicate, main_q_next_object = None, None, None
							if type(main_state_attribute) != type(None) and type(target_q_attribute) != type(None):	
								main_q_attribute = transition.attribute_reward + parameters["discount_factor"] * torch.max(model_attribute_main(main_state_attribute))
								loss_attribute = loss_fn_attribute(main_q_attribute, target_q_attribute)
								optimizer_attribute.zero_grad()	
								loss_attribute.backward()
								for param in model_attribute_main.parameters():
									param.grad.data.clamp_(-1, 1)
								optimizer_attribute.step()

							if type(main_state_predicate) != type(None) and type(target_q_predicate) != type(None):
								main_q_predicate = transition.predicate_reward + parameters["discount_factor"] * torch.max(model_predicate_main(main_state_predicate))
								loss_predicate = loss_fn_predicate(main_q_predicate, target_q_predicate)
								optimizer_predicate.zero_grad()
								loss_predicate.backward()
								for param in model_predicate_main.parameters():
									param.grad.data.clamp_(-1, 1)
								optimizer_predicate.step()

							if type(main_state_next_object) != type(None) and type(target_q_next_object) != type(None):
								main_q_next_object = transition.next_object_reward + parameters["discount_factor"] * torch.max(model_next_object_main(main_state_next_object))
								loss_next_object = loss_fn_next_object(main_q_next_object, target_q_next_object)
								optimizer_next_object.zero_grad()
								loss_next_object.backward()
								for param in model_next_object_main.parameters():
									param.grad.data.clamp_(-1, 1)
								optimizer_next_object.step()

					# update target weights if it has been tao steps
					if total_number_timesteps_taken % target_update_frequency == 0:
						update_target(model_attribute_main, model_attribute_target)
						update_target(model_predicate_main, model_predicate_target)
						update_target(model_next_object_main, model_next_object_target)
			
		# evaluate statistics on validation set
		evaluate(validation_data_loader)
	
	with open("image_states.pickle", "wb") as handle:
		pickle.dump(image_states, handle)


def evaluate(data_loader):
	pass

def create_state_vector(image_state):
	# find subject to start with if curr_subject is None
	if image_state.current_subject == None or len(image_state.objects_explored_per_subject[image_state.current_subject]) >= image_state.max_objects_to_explore:
		curr_subject_feature = None
		for idx in range(len(image_state.entity_scores)):
			if idx not in image_state.explored_entities:
				curr_subject_feature = image_state.entity_features[idx]
				image_state.explored_entities.append(idx)
				image_state.current_subject = idx
				break
		if type(curr_subject_feature) == type(None):
			return None
	else:	
		curr_subject_feature = image_state.entity_features[image_state.current_subject]
	# find object for this state if object is none
	if image_state.current_object == None:
		curr_object_id =  len(image_state.objects_explored_per_subject[image_state.current_subject])
		if curr_object_id == image_state.current_subject:
			curr_object_id += 1
		#curr_object_id = find_object_neighbors(image_state.entity_proposals[image_state.current_subject], image_state.entity_proposals, image_state.objects_explored_per_subject[image_state.current_subject])
		if curr_object_id >= len(image_state.entity_scores):
			return None
		image_state.current_object = curr_object_id
		#image_state.current_object = curr_object_id[0]
	curr_object_feature = image_state.entity_features[image_state.current_object]
	return torch.cat([torch.squeeze(image_state.image_feature), torch.squeeze(curr_subject_feature), torch.squeeze(curr_object_feature)])

def create_state_action_vector(state_vector, action_set, total_set_size):
	len_action_set = len(action_set)
	if len_action_set == 0:
		return None
	else:
		onehot = torch.FloatTensor(len_action_set, total_set_size)
		onehot.zero_()
		onehot.scatter_(1, torch.LongTensor(action_set).view(-1, 1), 1)
		identity = torch.autograd.Variable(onehot.float())
		if torch.cuda.is_available():
			identity = identity.cuda()
		model_input = torch.cat([state_vector.repeat(len_action_set, 1), identity], 1)
		return model_input.view(model_input.size(0), 1, model_input.size(1))

def choose_action_epsilon_greedy(state, adaptive_action_set, model, epsilon, training=False):
	sample = random.random()
	if sample > epsilon and training: # exploit
		return adaptive_action_set[int(torch.squeeze(model(state)).max(0)[1].data.cpu().numpy())]
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
	parser.add_argument("--validation_data", type=str,
				default="data/data_samples/validation_data.json",
				help='Location of the file containing validation data samples')
	parser.add_argument("--test_data", type=str,
				default="data/data_samples/test_data.json",
				help='Location of the file containing test data samples')
	parser.add_argument("--images_dir", type=str,
				default="data/VG_100K/",
				help="Location of Visual Genome images")
	parser.add_argument("--train", help="trains model", action="store_true")
	parser.add_argument("--test", help="evaluates model", action="store_true")
	parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs to train on")
	parser.add_argument("--batch_size", type=int, default=4, help="batch size to use")
	parser.add_argument("--discount_factor", type=float, default=0.9, help="discount factor")
	parser.add_argument("--learning_rate", type=float, default=0.0007, help="learning rate")
	parser.add_argument("--epsilon", type=float, default=1, help="epsilon starting value (used in epsilon greedy)")
	parser.add_argument("--epsilon_anneal_rate", type=float, default=0.045, help="factor to anneal epsilon by")
	parser.add_argument("--epsilon_end", type=float, default=0.1, help="minimum value of epsilon (when we can stop annealing)")
	parser.add_argument("--target_update_frequency", type=int, default=10000, help="how often to update the target")
	parser.add_argument("--replay_buffer_capacity", type=int, default=20000, help="maximum size of the replay buffer")
	parser.add_argument("--replay_buffer_minimum_number_samples", type=int, default=8, help="Minimum replay buffer size before we can sample")
	parser.add_argument("--object_detection_threshold", type=float, default=0.005, help="threshold for Faster RCNN module when detecting objects")
	parser.add_argument("--maximum_num_entities_per_image", type=int, default=10, help="maximum number of entities to explore per image")
	parser.add_argument("--maximum_adaptive_action_space_size", type=int, default=20, help="maximum size of adaptive_action space")
	parser.add_argument("--num_workers", type=int, default=4, help="number of threads")
	args = parser.parse_args()

	# saving parameters in variables
	num_epochs = args.num_epochs
	batch_size = args.batch_size
	discount_factor = args.discount_factor
	learning_rate = args.learning_rate
	epsilon = args.epsilon
	epsilon_anneal_rate = args.epsilon_anneal_rate
	epsilon_end = args.epsilon_end
	target_update_frequency = args.target_update_frequency
	replay_buffer_capacity = args.replay_buffer_capacity
	replay_buffer_minimum_number_samples = args.replay_buffer_minimum_number_samples	
	object_detection_threshold = args.object_detection_threshold
	maximum_num_entities_per_image = args.maximum_num_entities_per_image
	maximum_adaptive_action_space_size = args.maximum_adaptive_action_space_size

	parameters = {"num_epochs": num_epochs, "batch_size": batch_size, "discount_factor": discount_factor,
			"learning_rate": learning_rate, "epsilon": epsilon, 
			"epsilon_anneal_rate": epsilon_anneal_rate, "epsilon_end": epsilon_end, 
			"target_update_frequency":target_update_frequency, 
			"replay_buffer_capacity":replay_buffer_capacity,
			"replay_buffer_minimum_number_samples": replay_buffer_minimum_number_samples,
			"object_detection_threshold": object_detection_threshold,
			"maximum_num_entities_per_image": maximum_num_entities_per_image,
			"maximum_adaptive_action_space_size": maximum_adaptive_action_space_size}

	# create semantic action graph
	print("Loading graph.pickle...")
	semantic_action_graph = pickle.load(open("graph.pickle", "rb"))
	print("Done!")	

	# create VGG model for state featurization
	print("Loading VGG model...")
	model_vgg = VGG16()
	print("Done!")

	# create Faster-RCNN model for state featurization
	print("Loading Fast-RCNN...")
	model_file = 'VGGnet_fast_rcnn_iter_70000.h5'
	model_frcnn = FasterRCNN()
	network.load_net(model_file, model_frcnn)
	model_frcnn.cuda()
	model_frcnn.eval()
	print("Done!")

	# create DQN's for the next object, predicates, and attributes
	print("Creating DQN models...")
	DQN_next_object_main = DQN_MLP(2048*3 + parameters["maximum_num_entities_per_image"], 1)
	DQN_next_object_target = DQN_MLP(2048*3 + parameters["maximum_num_entities_per_image"], 1)
	DQN_predicate_main = DQN_MLP(2048*3 + len(semantic_action_graph.predicate_nodes), 1)
	DQN_predicate_target = DQN_MLP(2048*3 + len(semantic_action_graph.predicate_nodes), 1)
	DQN_attribute_main = DQN_MLP(2048*3 + len(semantic_action_graph.attribute_nodes), 1)
	DQN_attribute_target = DQN_MLP(2048*3 + len(semantic_action_graph.attribute_nodes), 1)
	print("Done!")

	# create shared optimizer
	# TODO: The paper says this optimizer is shared. Right now
	# the optimizers are not shared. Need to implement version
	# where it is shared
	print("Creating optimizers...")
	optimizer_next_object = torch.optim.RMSprop(DQN_next_object_main.parameters())
	optimizer_predicate = torch.optim.RMSprop(DQN_predicate_main.parameters())
	optimizer_attribute = torch.optim.RMSprop(DQN_attribute_main.parameters())
	print("Done!")

	# define loss functions
	loss_fn_attribute = nn.MSELoss()
	loss_fn_predicate = nn.MSELoss()
	loss_fn_next_object = nn.MSELoss()	

	# create replay buffer
	print("Creating replay buffer...")
	replay_buffer = ReplayMemory(replay_buffer_capacity, replay_buffer_minimum_number_samples)
	print("Done!")

	# load train data samples
	if args.train:
		train_data_samples = json.load(open(args.train_data))
		train_dataset = VGDataset(train_data_samples, args.images_dir)
		train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
						shuffle=True, num_workers=args.num_workers,
						collate_fn=collate)
		validation_data_samples = json.load(open(args.validation_data))
		validation_dataset = VGDataset(validation_data_samples, args.images_dir)
		validation_data_loader = DataLoader(dataset=validation_dataset, batch_size=args.batch_size,
									shuffle=True, num_workers=args.num_workers)
		train_images_state = train(parameters)
	if args.test:
		test_data_samples = json.load(open(args.test_data))
		test_dataset = VGDataset(test_data_samples, args.images_dir)
		test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
									shuffle=True, num_workers=args.num_workers)
		test_images_state = evaluate(test_data_loader)

