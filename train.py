from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#from dataset import VGDataset, collate
from models import ResNet50, DQN, DQN_MLP
from operator import itemgetter
from faster_rcnn.faster_rcnn import FasterRCNN
from image_state import ImageState
from replay_buffer import ReplayMemory
from utils.vg_utils import entity_to_aliases, predicate_to_aliases, find_object_neighbors, crop_box
from PIL import Image
from skip_thoughts import skipthoughts
from collections import defaultdict

import torch
import torch.nn as nn
import pickle
import numpy as np
import random


def train(semantic_action_graph, parameters, flags, models, dataloaders, optimizers, loss_functions, replay_buffer):
        print("CUDA Available: " + str(torch.cuda.is_available()))
        # make model CUDA
        if torch.cuda.is_available():
                model_IM_EMB = models["im_emb_model"].cuda()
                model_FRCNN = models["model_frcnn"].cuda()
                model_next_object_main = models["DQN_next_object_main"].cuda()
                model_next_object_target = models["DQN_next_object_target"].cuda()
                model_attribute_main = models["DQN_attribute_main"].cuda()
                model_attribute_target = models["DQN_attribute_target"].cuda()
                model_predicate_main = models["DQN_predicate_main"].cuda()
                model_predicate_target = models["DQN_predicate_target"].cuda()

        # keeps track of current scene graphs for images
        image_states = {}
        total_number_timesteps_taken = 0
        data_loader_val = dataloaders["val"]
        number_of_epochs = parameters["num_epochs"]
        data_loader = dataloaders["train"]

        # dictionary for skip-though
        skip_thought_dict = defaultdict(lambda:[])

        for epoch in range(number_of_epochs):
                print("Epoch: ", epoch)
                num = -1
                for progress, (images, images_orig, gt_scene_graph) in enumerate(data_loader):
                        images = torch.autograd.Variable(torch.squeeze(images, 1))
                        if torch.cuda.is_available():
                                images = images.cuda()
                        # get image features from VGG16
                        images = model_IM_EMB(images)

                        # iterate through images in batch
                        for idx in range(images.size(0)):
                                num += 1
                                print("Image number " + str(num))
                                # initializing image state if necessary
                                image_name = gt_scene_graph[idx]["image_name"]
                                if image_name not in image_states:
                                        gt_sg = gt_scene_graph[idx]
                                        image_feature = images[idx]
                                        entity_proposals, entity_scores, entity_classes = [], [], []
                                        for obj in gt_scene_graph[idx]["labels"]["objects"]:
                                                entity_proposals.append([obj["x"], obj["y"], obj["x"] + obj["w"], obj["y"] + obj["h"]])
                                                entity_scores.append(1)
                                                if "name" in obj:
                                                        entity_classes.append(obj["name"])
                                                else:
                                                        entity_classes.append(obj["names"][0])
                                        entity_proposals = np.array(entity_proposals)
                                        entity_scores = np.array(entity_scores)
                                        entity_classes = np.array(entity_classes)
                                        #entity_proposals, entity_scores, entity_classes = models["model_FRCNN"].detect(images_orig[idx], object_detection_threshold)
                                        
                                        entity_proposals = entity_proposals[:parameters["maximum_num_entities_per_image"]]
                                        entity_scores = entity_scores[:parameters["maximum_num_entities_per_image"]]
                                        entity_classes = entity_classes[:parameters["maximum_num_entities_per_image"]]
                                        if len(entity_scores) < 2:
                                                continue

                                        entity_features = []
                                        for box in entity_proposals:
                                                cropped_entity = crop_box(images_orig[idx], box)
                                                cropped_entity = torch.autograd.Variable(cropped_entity)
                                                if torch.cuda.is_available():
                                                        cropped_entity = cropped_entity.cuda()
                                                box_feature = model_IM_EMB(cropped_entity)
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

                                        #print("Iter for image " + str(image_name))

                                        # get the image state object for image
                                        im_state = image_states[image_name]

                                        #print("Computing state vector")
                                        # compute state vector of image
                                        state_vector = create_state_vector(im_state, skip_thought_dict, models["skip_thought_encoder"], 
                                                                           semantic_action_graph, use_skip_thought=flags["skip_thought"])
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
                                        #print("Creating adaptive action sets...")
                                        subject_name = entity_to_aliases(im_state.entity_classes[subject_id])
                                        object_name = entity_to_aliases(im_state.entity_classes[object_id])
                                        subject_bbox = im_state.entity_proposals[subject_id]
                                        previously_mined_attributes = im_state.current_scene_graph["objects"][subject_id]["attributes"]
                                        previously_mined_next_objects = im_state.objects_explored_per_subject[subject_id]

                                        if flags["adaptive_action_sets"]:
                                                attribute_adaptive_actions, predicate_adaptive_actions = semantic_action_graph.variation_based_traversal(subject_name, object_name, previously_mined_attributes)
                                                next_object_adaptive_actions = find_object_neighbors(subject_bbox, im_state.entity_proposals, previously_mined_next_objects)
                                        else:
                                                attribute_adaptive_actions = range(len(semantic_action_graph.attribute_nodes))
                                                predicate_adaptive_actions = range(len(semantic_action_graph.predicate_nodes))
                                                next_object_adaptive_actions = range(len(im_state.entity_proposals)-1)

                                        # creating state + action vectors to feed in DQN
                                        #print("Creating state + action vectors to pass into DQN...")
                                        attribute_state_vectors = create_state_action_vector(state_vector, attribute_adaptive_actions, len(semantic_action_graph.attribute_nodes))
                                        predicate_state_vectors = create_state_action_vector(state_vector, predicate_adaptive_actions, len(semantic_action_graph.predicate_nodes))
                                        next_object_state_vectors = create_state_action_vector(state_vector, next_object_adaptive_actions, parameters["maximum_num_entities_per_image"])

                                        # choose action using epsilon greedy
                                        #print("Choose action using epsilon greedy...")
                                        attribute_action, predicate_action, next_object_action = None, None, None
                                        if type(attribute_state_vectors) != type(None):
                                                attribute_action = choose_action_epsilon_greedy(attribute_state_vectors, attribute_adaptive_actions, model_attribute_main, parameters["epsilon"], training=replay_buffer.can_sample())
                                        if type(predicate_state_vectors) != type(None):
                                                predicate_action = choose_action_epsilon_greedy(predicate_state_vectors, predicate_adaptive_actions, model_predicate_main, parameters["epsilon"], training=replay_buffer.can_sample())

                                        # update skip thought vector
                                        if predicate_action != None and flags["skip_thought"]:
                                                skip_thought_dict[(im_state.current_subject, im_state.current_object)].append(predicate_action)
                                        if len(skip_thought_dict[(im_state.current_subject, im_state.current_object)]) > 2:
                                                skip_thought_dict[(im_state.current_subject, im_state.current_object)].pop(0)

                                        if type(next_object_state_vectors) != type(None):
                                                next_object_action = choose_action_epsilon_greedy(next_object_state_vectors, next_object_adaptive_actions, model_next_object_main, parameters["epsilon"], training=replay_buffer.can_sample())
                                        # step image_state
                                        #print("Step state environment using action...")
                                        attribute_reward, predicate_reward, next_object_reward, done = im_state.step(attribute_action, predicate_action, next_object_action)
                                        #print("Rewards(A,P,O)", attribute_reward, predicate_reward, next_object_reward)                                        
                                        next_state = create_state_vector(im_state, skip_thought_dict, models["skip_thought_encoder"], semantic_action_graph,
                                                                         use_skip_thought=flags["skip_thought"])
                                        im_state = image_states[image_name]
                                        # decay epsilon
                                        if parameters["epsilon"] > parameters["epsilon_end"]:
                                                parameters["epsilon"] = parameters["epsilon"] * parameters["epsilon_anneal_rate"]
                                                #print("NEW EPSILON", parameters["epsilon"])
                                        # add transition tuple to replay buffer
                                        #print("Adding transition tuple to replay buffer...")
                                        subject_name_1 =  entity_to_aliases(im_state.entity_classes[im_state.current_subject])
                                        object_name_1 = entity_to_aliases(im_state.entity_classes[im_state.current_object])
                                        previously_mined_attributes_1 = im_state.current_scene_graph["objects"][im_state.current_subject]["attributes"]
                                        previously_mined_next_objects_1 = im_state.objects_explored_per_subject[im_state.current_subject]
                                        attribute_adaptive_actions_1, predicate_adaptive_actions_1 = semantic_action_graph.variation_based_traversal(subject_name_1, object_name_1, previously_mined_attributes)
                                        next_object_adaptive_actions_1 = find_object_neighbors(im_state.entity_proposals[im_state.current_subject], im_state.entity_proposals, previously_mined_next_objects)

                                        replay_buffer.push(state_vector, next_state, attribute_adaptive_actions, predicate_adaptive_actions, next_object_adaptive_actions, attribute_reward, predicate_reward, next_object_reward, attribute_adaptive_actions_1, predicate_adaptive_actions_1, next_object_adaptive_actions_1, done)

                                        # sample minibatch if replay_buffer has enough samples
                                        if replay_buffer.can_sample():
                                                #print("Sample minibatch of transitions...")
                                                minibatch_transitions = replay_buffer.sample(parameters["batch_size"])
                                                main_q_attribute_list, main_q_predicate_list, main_q_next_object_list = [], [], []
                                                target_q_attribute_list, target_q_predicate_list, target_q_next_object_list = [], [], []
                                                for transition in minibatch_transitions:
                                                        total_number_timesteps_taken += 1
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
                                                                        output = torch.max(model_attribute_target(next_state_attribute))[0]
                                                                        #print("output of target model attributes", output)
                                                                        target_q_attribute = transition.attribute_reward + parameters["discount_factor"] * output
                                                                if type(next_state_predicate) != type(None):
                                                                        next_state_predicate.volatile = True
                                                                        target_q_predicate = transition.predicate_reward + parameters["discount_factor"] * torch.max(model_predicate_target(next_state_predicate))[0]
                                                                if type(next_state_next_object) != type(None):
                                                                        next_state_next_object.volatile = True
                                                                        target_q_next_object = transition.next_object_reward + parameters["discount_factor"] * torch.max(model_next_object_target(next_state_next_object))[0]
                                                        # compute loss
                                                        main_state_attribute = create_state_action_vector(transition.state, transition.attribute_actions, len(semantic_action_graph.attribute_nodes))
                                                        main_state_predicate = create_state_action_vector(transition.state, transition.predicate_actions, len(semantic_action_graph.predicate_nodes))
                                                        main_state_next_object = create_state_action_vector(transition.state, transition.next_object_actions, parameters["maximum_num_entities_per_image"])

                                                        main_q_attribute, main_q_predicate, main_q_next_object = None, None, None
                                                        if type(main_state_attribute) != type(None) and type(target_q_attribute) != type(None):
                                                                main_q_attribute = transition.attribute_reward + parameters["discount_factor"] * torch.max(model_attribute_main(main_state_attribute))
                                                                #print("main & target preds", main_q_attribute, target_q_attribute)
                                                                loss_attribute = loss_functions["attribute"](main_q_attribute, target_q_attribute)
                                                                #print("Loss attribute: " + str(loss_attribute.data[0]))
                                                                optimizers["attribute"].zero_grad()
                                                                loss_attribute.backward()
                                                                for param in model_attribute_main.parameters():
                                                                        param.grad.data.clamp_(-1, 1)
                                                                optimizers["attribute"].step()

                                                        if type(main_state_predicate) != type(None) and type(target_q_predicate) != type(None):
                                                                main_q_predicate = transition.predicate_reward + parameters["discount_factor"] * torch.max(model_predicate_main(main_state_predicate))
                                                                loss_predicate = loss_functions["predicate"](main_q_predicate, target_q_predicate)
                                                                optimizers["predicate"].zero_grad()
                                                                #print("Loss predicate: " + str(loss_predicate.data[0]))
                                                                loss_predicate.backward()
                                                                for param in model_predicate_main.parameters():
                                                                        param.grad.data.clamp_(-1, 1)
                                                                optimizers["predicate"].step()

                                                        if type(main_state_next_object) != type(None) and type(target_q_next_object) != type(None):
                                                                main_q_next_object = transition.next_object_reward + parameters["discount_factor"] * torch.max(model_next_object_main(main_state_next_object))
                                                                loss_next_object = loss_functions["next_object"](main_q_next_object, target_q_next_object)
                                                                optimizers["next_object"].zero_grad()
                                                                #print("Loss next object: " + str(loss_next_object.data[0]))
                                                                loss_next_object.backward()
                                                                for param in model_next_object_main.parameters():
                                                                        param.grad.data.clamp_(-1, 1)
                                                                optimizers["next_object"].step()

                                        # update target weights if it has been tao steps
                                        if total_number_timesteps_taken % parameters["target_update_frequency"] == 0:
                                                #print("UPDATING TARGET NOW")
                                                update_target(model_attribute_main, model_attribute_target)
                                                update_target(model_predicate_main, model_predicate_target)
                                                update_target(model_next_object_main, model_next_object_target)

        gt_graphs = []
        our_graphs = []
        for ims in image_states.values():
                gt_graphs.append(ims.gt_scene_graph)
                our_graphs.append(ims.current_scene_graph)
        with open("image_states.pickle", "wb") as handle:
                pickle.dump({"gt": gt_graphs, "curr": our_graphs}, handle)


def create_state_vector(image_state, skip_thought_dict, skip_thought_encoder, semantic_action_graph, use_skip_thought=False):
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
                curr_object_id = len(image_state.objects_explored_per_subject[image_state.current_subject])
                if curr_object_id == image_state.current_subject:
                        curr_object_id += 1
                #curr_object_id = find_object_neighbors(image_state.entity_proposals[image_state.current_subject], image_state.entity_proposals, image_state.objects_explored_per_subject[image_state.current_subject])
                if curr_object_id >= len(image_state.entity_scores):
                        return None
                image_state.current_object = curr_object_id
                #image_state.current_object = curr_object_id[0]
        subject_name = image_state.entity_classes[image_state.current_subject]
        object_name = image_state.entity_classes[image_state.current_object]
        # get skip thought encoding
        if len(skip_thought_dict[(image_state.current_subject, image_state.current_object)]) > 0 and use_skip_thought:
                relationships = skip_thought_dict[(image_state.current_subject, image_state.current_object)]
                rel = semantic_action_graph.predicate_nodes[relationships[0]].name
                if type(rel) == tuple:
                        rel = rel[0]
                st_encoding = torch.from_numpy(skip_thought_encoder.encode([subject_name +" "+ rel + " " + object_name]))
                if  len(relationships) == 2:
                        rel2 = semantic_action_graph.predicate_nodes[relationships[1]].name
                        if type(rel2) == tuple:
                                rel2 = rel2[0]
                        st_encoding2 = torch.from_numpy(skip_thought_encoder.encode([subject_name +" "+ rel2 + " " + object_name]))
                        st_encoding = torch.cat([torch.squeeze(st_encoding), torch.squeeze(st_encoding2)])
                else:
                        st_encoding = torch.cat([torch.squeeze(st_encoding), torch.zeros(4800)])
        else:
                st_encoding = torch.zeros(9600)

        curr_object_feature = image_state.entity_features[image_state.current_object]
        return torch.cat([torch.squeeze(image_state.image_feature), torch.squeeze(curr_subject_feature), torch.squeeze(curr_object_feature), torch.autograd.Variable(st_encoding).cuda()])

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
        if sample > epsilon and training: # exploiti
                return adaptive_action_set[int(torch.squeeze(model(state)).max(0)[1].data.cpu().numpy())]
        else: # explore
                return random.choice(adaptive_action_set)

def update_target(main_model, target_model):
        target_model.load_state_dict(main_model.state_dict())
