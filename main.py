from train import train
from dataset import VGDataset, collate
from models import ResNet50, DQN, DQN_MLP
from faster_rcnn.faster_rcnn import FasterRCNN
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from replay_buffer import ReplayMemory
from skip_thoughts import skipthoughts
from faster_rcnn import network

import torch
import torch.nn as nn
import argparse
import json
import pickle

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
        parser.add_argument("--image_embedding_model_type", type=str, 
                                default="resnet",
                                help="resnet or vgg (used to create image embedding)")
	parser.add_argument("--train", help="trains model", action="store_true")
	parser.add_argument("--test", help="evaluates model", action="store_true")
	parser.add_argument("--num_epochs", type=int, default=5, help="number of epochs to train on")
	parser.add_argument("--batch_size", type=int, default=4, help="batch size to use")
	parser.add_argument("--discount_factor", type=float, default=0.9, help="discount factor")
	parser.add_argument("--learning_rate", type=float, default=0.0007, help="learning rate")
	parser.add_argument("--epsilon", type=float, default=1, help="epsilon starting value (used in epsilon greedy)")
	parser.add_argument("--epsilon_anneal_rate", type=float, default=0.045, help="factor to anneal epsilon by")
	parser.add_argument("--epsilon_end", type=float, default=0.1, help="minimum value of epsilon (when we can stop annealing)")
	parser.add_argument("--target_update_frequency", type=int, default=30, help="how often to update the target")
	parser.add_argument("--replay_buffer_capacity", type=int, default=20000, help="maximum size of the replay buffer")
	parser.add_argument("--replay_buffer_minimum_number_samples", type=int, default=8, help="Minimum replay buffer size before we can sample")
	parser.add_argument("--object_detection_threshold", type=float, default=0.005, help="threshold for Faster RCNN module when detecting objects")
	parser.add_argument("--maximum_num_entities_per_image", type=int, default=500, help="maximum number of entities to explore per image")
	parser.add_argument("--maximum_adaptive_action_space_size", type=int, default=20, help="maximum size of adaptive_action space")
	parser.add_argument("--num_workers", type=int, default=4, help="number of threads")
	parser.add_argument("--use_adaptive_action_sets", help="Whether to use adaptive actions sets or not", action="store_true")
	parser.add_argument("--use_skip_thought", help="Whether to use skip thought history embeddings in the state", action="store_true")
	parser.add_argument("--use_fastrcnn_proposals", help="Whether to use fast rcnn generated proposals (vs ground truth)", action="store_true")
	parser.add_argument("--use_exact_match", help="Whether to consider similar matches as positive rewards (as opposed to exact match)", action="store_true")
	parser.add_argument("--positive_reward", type=int, default=1, help="Amount of positive reward when correct prediction made")
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
	print("Loading image embedding model...")
        if args.image_embedding_model_type == "resnet":
	    im_emb_model = ResNet50()
        elif args.image_embedding_model_type == "vgg":
            im_emb_model = VGG16()
        else:
            print("--image_embedding_model_type must be either resnet or vgg")
            sys.exit(0)
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
	DQN_next_object_main = DQN_MLP(2048*3+9600 + parameters["maximum_num_entities_per_image"], 1)
	DQN_next_object_target = DQN_MLP(2048*3+9600 + parameters["maximum_num_entities_per_image"], 1)
	DQN_predicate_main = DQN_MLP(2048*3+9600 + len(semantic_action_graph.predicate_nodes), 1)
	DQN_predicate_target = DQN_MLP(2048*3+9600 + len(semantic_action_graph.predicate_nodes), 1)
	DQN_attribute_main = DQN_MLP(2048*3+9600 + len(semantic_action_graph.attribute_nodes), 1)
	DQN_attribute_target = DQN_MLP(2048*3+9600 + len(semantic_action_graph.attribute_nodes), 1)
	print("Done!")

	# create shared optimizer
	# The paper says this optimizer is shared. Right now
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

	# load skip thought model
	skip_thought_model = skipthoughts.load_model()
	skip_thought_encoder = skipthoughts.Encoder(skip_thought_model)

	models = {"im_emb_model": im_emb_model, "model_frcnn": model_frcnn, "DQN_next_object_main": DQN_next_object_main,
              "DQN_next_object_target": DQN_next_object_target, "DQN_attribute_main": DQN_attribute_main,
              "DQN_attribute_target": DQN_attribute_target, "DQN_predicate_main": DQN_predicate_main,
              "DQN_predicate_target": DQN_predicate_target, "skip_thought_model": skip_thought_model, 
              "skip_thought_encoder": skip_thought_encoder}
	optimizers = {"next_object": optimizer_next_object, "predicate": optimizer_predicate, "attribute": optimizer_attribute}
	loss_functions = {"next_object": loss_fn_next_object, "predicate":loss_fn_predicate, "attribute":loss_fn_attribute}
	flags = {"skip_thought": args.use_skip_thought, "frcnn": args.use_fastrcnn_proposals, "adaptive_action_sets": args.use_adaptive_action_sets}

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
		dataloaders = {"val": validation_data_loader, "train": train_data_loader}
		train_images_state = train(semantic_action_graph, parameters, flags, models, dataloaders, optimizers, loss_functions, replay_buffer)
	if args.test:
		test_data_samples = json.load(open(args.test_data))
		test_dataset = VGDataset(test_data_samples, args.images_dir)
		test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
									shuffle=True, num_workers=args.num_workers)
		test_images_state = evaluate(test_data_loader)
