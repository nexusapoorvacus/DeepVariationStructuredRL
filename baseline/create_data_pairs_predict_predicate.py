import json
import pickle
import os
import collections
import random

RELATIONSHIPS_FILE = "relationships.json"
RELATIONSHIP_ALIAS_FILE = "relationship_alias.txt"
OUTPUT_FILE_TRAIN = open("data_train_all_pred_small.json", "wb")
OUTPUT_FILE_TEST = open("data_test_all_pred_small.json", "wb")
OUTPUT_FILE_VALIDATE = open("data_validate_all_pred_small.json", "wb")
NUM_IMAGES = 10
NUM_PAIRS_PER_IMAGE = 1
PERCENT_TRAIN = 0.7
PERCENT_VALIDATE = 0.1

relationship_labels = {("No Relationship",): 0}

def read_rel_aliases(filename):
    alias_file = open(filename, "r")
    rel_aliases = {}
    for line in alias_file.readlines():
        rels = line.replace("\n", "").split(",")
        for rr in rels:
            rel_aliases[rr] = tuple(rels)
    return rel_aliases

# load alias file
rel_aliases = read_rel_aliases(RELATIONSHIP_ALIAS_FILE)

if not os.path.exists('rel_pairs.pickle'):
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
            if rel["predicate"] in rel_aliases:
                predicate = rel_aliases[rel["predicate"]]
            else:
                predicate = (rel["predicate"],)
            if predicate not in relationship_labels:
                relationship_labels[predicate] = max(relationship_labels.values()) + 1
            positive_pairs[im_name].append((subj_tuple, obj_tuple, predicate))
    with open('rel_pairs.pickle', 'wb') as handle:
        pickle.dump(positive_pairs, handle)
    # save relationship labels dictionary
    with open('relationship_labels.pickle', 'wb') as handle:
        pickle.dump(relationship_labels, handle)
else:
    positive_pairs = pickle.load(open('rel_pairs.pickle', "rb"))
 
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
        pred = pairs[i][2]
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
        data_samples.append((im_name, subj_p, obj_p, pred))
        if obj_n == None:
            data_samples.append((im_name, subj_n, obj_p, ("No Relationship",)))
        else:
            data_samples.append((im_name, subj_p, obj_n, ("No Relationship",)))

NUM_TRAINING_SAMPLES = int(len(data_samples)*PERCENT_TRAIN)
NUM_VALIDATE_SAMPLES = int(len(data_samples)*PERCENT_VALIDATE)
OUTPUT_FILE_TRAIN.write(str(data_samples[:NUM_TRAINING_SAMPLES]))
OUTPUT_FILE_VALIDATE.write(str(data_samples[NUM_TRAINING_SAMPLES:NUM_TRAINING_SAMPLES + NUM_VALIDATE_SAMPLES]))
OUTPUT_FILE_TEST.write(str(data_samples[NUM_TRAINING_SAMPLES + NUM_VALIDATE_SAMPLES:]))
