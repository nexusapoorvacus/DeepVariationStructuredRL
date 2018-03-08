from graph import Semantic_Action_Graph, Entity_Node, Predicate_Node, Attribute_Node, Predicate_Edge, Attribute_Edge
from utils.vg_utils import entity_to_aliases, predicate_to_aliases

import json
import pickle
import argparse
import os
import sys

# files to read
OBJECTS_FILE = "data/objects.json"
PREDICATES_FILE = "data/relationships.json"
ATTRIBUTES_FILE = "data/attributes.json"

def add_objects(graph, object_data, entity_counts, min_occurrences):
	for image in object_data:
		for entity in image["objects"]:
			object_name = entity_to_aliases(entity["name"]) if "name" in entity else entity_to_aliases(entity["names"])
			if entity_counts[str(object_name)] >= min_occurrences:
				if graph.get_entity_id(object_name) == None:
					# add node to graph
					Entity_Node(object_name, graph)	

def add_predicates(graph, predicate_data, entity_counts, predicate_counts, min_occurrences):
	for image in predicate_data:
		relationships_info = image["relationships"]
		for relationship_info in relationships_info:
			predicate_name = predicate_to_aliases(relationship_info["predicate"])
			if predicate_counts[str(predicate_name)] >= min_occurrences:
				if graph.get_predicate_id(predicate_name) == None:
					# add node to graph
					pred_node = Predicate_Node(predicate_name, graph)
			else:
				continue
			subject_name = relationship_info["subject"]["name"] if "name" in relationship_info["subject"] else relationship_info["subject"]["names"][0]
			object_name = relationship_info["object"]["name"] if "name" in relationship_info["object"] else relationship_info["object"]["names"][0]
			subject_name = entity_to_aliases(subject_name)
			object_name = entity_to_aliases(object_name)
			subject_node = graph.get_entity_by_name(subject_name)
			object_id = graph.get_entity_id(object_name)
			if subject_node == None or object_id == None or entity_counts[str(subject_name)] < min_occurrences or entity_counts[str(object_name)] < min_occurrences:
				continue
			if subject_node == None:
				subject_node = Entity_Node(subject_name, graph)
			if object_id == None:
				object_node = Entity_Node(object_name, graph)
				object_id = object_node.ID
			predicate_id = graph.get_predicate_id(predicate_name)

			if subject_node.get_predicate_edge(predicate_id, object_id) == None:
				# creating and adding edge
				new_edge = Predicate_Edge(subject_node.ID, predicate_id, object_id)
				subject_node.add_predicate_edge(new_edge)
		
def add_attributes(graph, attribute_data, entity_counts, attribute_counts, min_occurrences):
	for image in attribute_data:
		for entity in image["attributes"]:
			if "attributes" in entity:
				attributes = entity["attributes"]
			else:
				continue
			for attribute_name in attributes:
				if attribute_counts[attribute_name] < min_occurrences:
					continue
				if graph.get_attribute_id(attribute_name) == None:
					# add node to graph
					attribute_node = Attribute_Node(attribute_name, graph)
				subject_name = entity["name"] if "name" in entity else entity["names"]
				subject_name = entity_to_aliases(subject_name)
				if entity_counts[str(subject_name)] < min_occurrences:
					continue
				subject_node = graph.get_entity_by_name(subject_name)
				if subject_node == None:
					subject_node = Entity_Node(subject_name, graph)
				attribute_id = graph.get_attribute_id(attribute_name)
				if subject_node.get_attribute_edge(attribute_id) == None:
					# creating and adding edge
					new_edge = Attribute_Edge(subject_node.ID, attribute_id)
					subject_node.add_attribute_edge(new_edge)
		
def main():
	# flags
	parser = argparse.ArgumentParser()
	parser.add_argument("-e", "--add_entities", help="adds entity nodes to the graph",
                    action="store_true")
	parser.add_argument("-p", "--add_predicates", help="adds predicate nodes to the graph",
                    action="store_true")
	parser.add_argument("-a", "--add_attributes", help="adds attribute nodes to the graph",
                    action="store_true")
	parser.add_argument("-m", "--minimum_occurrances", type=int, default=200, help="minimum number of occurrances an entity, predicate, and attribute must have in VG to be added to the graph")
	args = parser.parse_args()
	MIN_OCCURRENCES = args.minimum_occurrances

	# load graph if exists
	if os.path.isfile("graph.pickle"):
		print("Loading graph from file...")
		loaded_graph_file = open("graph.pickle", "rb")
		graph = pickle.load(loaded_graph_file)
		loaded_graph_file.close()
		print("Done!")
	else: 	
		print("Creating Graph...")
		graph = Semantic_Action_Graph()

	print("Loading entity counts file...")
	entity_counts = json.load(open("entity_counts.json", "rb"))
	print("Done!")
	
	# adding nodes/edges	
	if args.add_entities:
		print("Loading entity data...")
		object_data = json.load(open(OBJECTS_FILE))
		print("Done!")
		
		print("Adding entity nodes...")
		add_objects(graph, object_data, entity_counts, MIN_OCCURRENCES)
	elif args.add_predicates:
		# make sure entity nodes have been added
		if len(graph.entity_nodes) == 0:
			print("Add entity nodes first by using the --add_entities flag")
			sys.exit(0)
		print("Loading predicate data...")
		predicate_data = json.load(open(PREDICATES_FILE))
		print("Done!")
		print("Loading predicate counts file...")
		predicate_counts = json.load(open("predicate_counts.json", "rb"))
		print("Done!")

		print("Adding predicate nodes and predicate edges...")
		add_predicates(graph, predicate_data, entity_counts, predicate_counts, MIN_OCCURRENCES)
	elif args.add_attributes:
		# make sure entity nodes have been added
		if len(graph.entity_nodes) == 0:
			print("Add entity nodes first by using the --add_entities flag")
			sys.exit(0)
		print("Loading attribute data...")
		attribute_data = json.load(open(ATTRIBUTES_FILE))
		print("Done!")
		print("Loading attribute counts file...")
		attribute_counts = json.load(open("attribute_counts.json", "rb"))
		print("Done!")

		print("Adding attribute nodes and attribute edges...")
		add_attributes(graph, attribute_data, entity_counts, attribute_counts, MIN_OCCURRENCES)
	else:
		print("You must use only one of --add_entities, -add_predicates, or -add_attributes")
		sys.exit(0)

	print("Done adding nodes/edges to graph. Saved in graph.pickle")
	with open("graph.pickle", "wb") as handle:
		pickle.dump(graph, handle)

	#print("Renaming saved_graph.pickle as graph.pickle...")
	#os.rename("saved_graph.pickle", "graph.pickle") 
	
if __name__ == "__main__":
	main()
