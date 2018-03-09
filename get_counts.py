from utils.vg_utils import entity_to_aliases, predicate_to_aliases

import json

# file to read
SCENE_GRAPH_FILE = "data/raw_data/scene_graphs.json"
ATTRIBUTES_FILE = "data/raw_data/attributes.json"

entity_counts = {}
predicate_counts = {}
attribute_counts = {}

def get_entity_predicate_counts(scene_graph_data):
	for image in scene_graph_data:
		for entity in image["objects"]:
			entity_name = entity["name"] if "name" in entity else entity["names"]
			entity_name = entity_to_aliases(entity_name)
			if str(entity_name) in entity_counts:
				entity_counts[str(entity_name)] += 1
			else:
				entity_counts[str(entity_name)] = 1
		for predicate in image["relationships"]:
			predicate_name = predicate["predicate"]
			predicate_name = predicate_to_aliases(predicate_name)
			if str(predicate_name) in predicate_counts:
				predicate_counts[str(predicate_name)] += 1
			else:
				predicate_counts[str(predicate_name)] = 1

def get_attribute_counts(attributes_data):
	for image in attributes_data:
		for entity in image["attributes"]:
			if "attributes" in entity:
				for attribute in entity["attributes"]:
					if attribute in attribute_counts:
						attribute_counts[attribute] += 1
					else:
						attribute_counts[attribute] = 1

def main():
	print("Loading scene graph data...")
	f = open(SCENE_GRAPH_FILE, "rb")
	scene_graph_data = json.load(f)
	f.close()
	print("Done!")
	get_entity_predicate_counts(scene_graph_data)

	print("Saving entity counts...")
	with open("entity_counts.json", "w") as outfile:
		json.dump(entity_counts, outfile)
	print("Done!")

	print("Saving predicate counts...")
	with open("predicate_counts.json", "w") as outfile:
		json.dump(predicate_counts, outfile)
	print("Done!")

	print("Loading attributes file...")
	f = open(ATTRIBUTES_FILE, "rb")
	attributes_data = json.load(f)
	f.close()
	print("Done!")
	get_attribute_counts(attributes_data)	
	print("Saving attribute counts...")
	with open("attribute_counts.json", "w") as outfile:
		json.dump(attribute_counts, outfile)
	print("Done!")


if __name__ == "__main__":
	main()
