import json
import pickle
import os
import collections
import random

RELATIONSHIPS_FILE = "relationships.json"
OUTPUT_FILE_TRAIN = open("data_train.json", "wb")
OUTPUT_FILE_TEST = open("data_test.json", "wb")
OUTPUT_FILE_VALIDATE = open("data_validate.json", "wb")
NUM_IMAGES = 50
NUM_PAIRS_PER_IMAGE = 2
PERCENT_TRAIN = 0.7
PERCENT_VALIDATE = 0.1

if (os.path.isfile("rel_pairs.pickle")):
    positive_pairs = pickle.load(open("rel_pairs.pickle", 'rb'))
else:
    relationships = json.load(open(RELATIONSHIPS_FILE))
    # coordinates of box stored as [x, y, w, h]
    positive_pairs = {}

    for i,im_rels in enumerate(relationships):
        im_name = str(im_rels["image_id"]) + ".jpg"
        positive_pairs[im_name] = []
        for rel in im_rels["relationships"]:
            subj_tuple = (rel["subject"]["y"], rel["subject"]["y"] + rel["subject"]["h"],
                          rel["subject"]["x"], rel["subject"]["x"] + rel["subject"]["w"])
            obj_tuple = (rel["object"]["y"], rel["object"]["y"] + rel["object"]["h"],
                          rel["object"]["x"], rel["object"]["x"] + rel["object"]["w"])
            positive_pairs[im_name].append((subj_tuple, obj_tuple))     
    with open('rel_pairs.pickle', 'wb') as handle:
        pickle.dump(positive_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
data_samples = []
# CREATE DATA FILE
for im_name, pairs in positive_pairs.items()[:NUM_IMAGES]:
    connections = collections.defaultdict(lambda: set())
    for item in pairs:
        connections[item[0]].add(item[1])
        connections[item[1]].add(item[0])
    for i in range(min(NUM_PAIRS_PER_IMAGE, len(pairs))):
        subj_p = pairs[i][0]
        obj_p = pairs[i][1]
        subj_n = None
        obj_n = None
        obj_n_fail = True
        subj_n_fail = True
        if random.random() < 0.5:
            # subject not connected to another object
            for conn in connections:
                if conn != subj_p:
                    if conn not in connections[subj_p]:
                        obj_n = conn
                        obj_n_fail = False
                        break
        if obj_n_fail:
            # object not connected to another subject
            for conn in connections:
                if conn != obj_p:
                    if conn not in connections[obj_p]:
                        subj_n = conn
                        subj_n_fail = False
                        break
        if subj_n_fail and obj_n_fail:
            continue
        data_samples.append((im_name, subj_p, obj_p, 1))
        if obj_n == None:
            data_samples.append((im_name, subj_n, obj_p, 0))
        else:
            data_samples.append((im_name, subj_p, obj_n, 0))

NUM_TRAINING_SAMPLES = int(len(data_samples)*PERCENT_TRAIN)
NUM_VALIDATE_SAMPLES = int(len(data_samples)*PERCENT_VALIDATE)
OUTPUT_FILE_TRAIN.write(str(data_samples[:NUM_TRAINING_SAMPLES]))
OUTPUT_FILE_VALIDATE.write(str(data_samples[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VALIDATE_SAMPLES]))
OUTPUT_FILE_TEST.write(str(data_samples[NUM_TRAINING_SAMPLES + NUM_VALIDATE_SAMPLES:]))
