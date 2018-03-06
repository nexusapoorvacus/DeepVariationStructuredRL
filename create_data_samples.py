import json
import argparse
import os
import sys

IMAGE_DIR = "data/images/"
OUTPUT_DATA_SAMPLE_FILE = "data/data_samples.json"
SCENE_GRAPH_DATA  = "data/scene_graphs.json"
ATTRIBUTE_DATA = "data/attributes.json"
NUM_IMAGES = 10

def create_data_sample_file():
    file_to_write = open(OUTPUT_DATA_SAMPLE_FILE, "w")
    images = []
    for im_name in os.listdir(IMAGE_DIR)[:NUM_IMAGES]:
        im_number = int(im_name[:-4])
        labels = scene_graph_information[image_to_id_mapping_SG[im_number]]
        for o, obj in enumerate(labels["objects"]):
            for obj_a in attribute_information[image_to_id_mapping_attr[im_number]]["attributes"]:
                if obj["object_id"] == obj_a["object_id"]:
                    labels["objects"][o]["attributes"] = obj_a["attributes"]
                    break
        im_data = {"image_name": im_name, "labels": labels}
        images.append(im_data)
    return images

def add_attributes_to_labels():
    new_data = []
    for im in data:
        im_name = im["image_name"]
        im_number = int(im["image_name"][:-4])
        labels = im["labels"]
        for o, obj in enumerate(labels["objects"]):
            for obj_a in attribute_information[image_to_id_mapping_attr[im_number]]["attributes"]:
                if obj["object_id"] == obj_a["object_id"]:
                    if "attributes" in obj_a:
                        labels["objects"][o]["attributes"] = obj_a["attributes"]
                    else:
                        labels["objects"][o]["attributes"] = []
                    break
        im_data = {"image_name": im_name, "labels": labels}
        new_data.append(im_data)
    return new_data

def create_image_to_id_mapping_SG():
    image_to_id_mapping = {}
    for i, image in enumerate(scene_graph_information):
        image_to_id_mapping[image["image_id"]] = i
    return image_to_id_mapping

def create_image_to_id_mapping_attr():
    image_to_id_mapping_attr = {}
    for i, image in enumerate(attribute_information):
        image_to_id_mapping_attr[image["image_id"]] = i
    return image_to_id_mapping_attr
    
if __name__ == "__main__":
    # flags
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--add_scene_graph_labels", help="adds scene graph labels to data samples",
                    action="store_true")
    parser.add_argument("-a", "--add_attribute_labels", help="adds attribute labels to the data samples",
                    action="store_true")
    args = parser.parse_args()
    
    if args.add_scene_graph_labels:
        print("Loading scene graph file")
        scene_graph_information = json.load(open(SCENE_GRAPH_DATA))
        print("Done!")
        image_to_id_mapping_SG = create_image_to_id_mapping_SG()
        data = create_data_sample_file()
    elif args.add_attribute_labels:
        # make sure data_samples.pickle file is available
        if not os.path.isfile("data_samples.json"):
            print("You must run this file with the --add_scene_graph_labels flag before running with --add_attribute_labels flag")
            sys.exit(0)
        with open('data_samples.json') as data_file:
            data = json.load(data_file)
        print("Loading attributes file")
        attribute_information = attribute_information = json.load(open(ATTRIBUTE_DATA))
        print("Done!")
        image_to_id_mapping_attr = create_image_to_id_mapping_attr()
        data = add_attributes_to_labels(data)
    else:
        print("Run with either the -s or -a flag. Do not use both flags at the same time.")
    
    # save as pickle object
    with open('data_samples.json', 'w') as outfile:
        json.dump(data, outfile)