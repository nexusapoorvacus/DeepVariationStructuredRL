from collections import defaultdict

class ImageState():

	def __init__(self, image_name, gt_scene_graph, image_feature, entity_features, entity_proposals, entity_classes, entity_scores, graph):
		self.image_name = image_name
		self.gt_scene_graph = gt_scene_graph
		self.current_scene_graph = {"relationships": [], "objects": []}
		self.graph = graph

		# number of objects to explore for a subject
		self.max_objects_to_explore = 5
		self.objects_explored_per_subject = defaultdict(list)

		# entities chosen to explore so far
		self.explored_entities = []
	
		# entity currently being explored
		self.current_subject = None
		self.current_object = None
		
		# image feature
		self.image_feature = image_feature
		
		# add object features
		self.entity_features = entity_features
		self.entity_proposals = entity_proposals
		self.entity_classes = entity_classes
		self.entity_scores = entity_scores

		# previously mined 
		# per subject for attributes and next object
		self.previously_mined_attributes = defaultdict(lambda: [])
		self.previously_mined_next_objects = defaultdict(lambda: [])

	def initialize_entities(self, entity_proposals, entity_classes, entity_scores):
		for i in range(len(entity_classes)):
			object_dict = {"object_id": 0, "x": 0, "y": 0, "w": 0, "h": 0, "name": "", "attributes":[], "score": 0}
			object_dict["object_id"] = len(self.current_scene_graph["objects"])
			points = entity_proposals[i]
			object_dict["x"] = int(points[0])
			object_dict["y"] = int(points[1])
			object_dict["w"] = int(points[2] - points[0])
			object_dict["h"] = int(points[3] - points[1])
			object_dict["name"] = entity_classes[i]
			object_dict["score"] = entity_scores[i]
			object_dict["attributes"] = []
			self.current_scene_graph["objects"].append(object_dict)

	def add_attribute(self, object_id, attribute_id):	
		# NOTE: object_id is the id for this particular image
		#	attribute_id is the id from the SAG
		attribute_name = self.graph.attribute_nodes[attribute_id].name
		for object_dict in self.current_scene_graph["objects"]:
			if object_dict["object_id"] == object_id:
				object_dict["attributes"].append(attribute_name)
				break

	def add_predicate(self, subject_id, predicate_name, object_id):
		# NOTE: subject_id and object_id are ids for this particular image
		#	predicate_id is the id from the SAG

		relationship_dict = {"relationship_id":0, "predicate": "", "subject_id": 0, "object_id": 0} 
		relationship_dict["relationship_id"] = len(self.current_scene_graph["relationships"])
		relationship_dict["predicate"] = predicate_name
		relationship_dict["subject_id"] = subject_id
		relationship_dict["object_id"] = object_id
		self.current_scene_graph["relationships"].append(relationship_dict)

	def is_done(self):
		# returns true if we are done building a scene graph for this image
		if len(self.entity_classes) <= 1:
			return True
		if len(self.explored_entities) == len(self.entity_classes):
			return True
		return False

	def reset(self):
		self.current_scene_graph = {"relationships": [], "objects": []}
		self.explored_entities = []
		self.object_counts_per_subject = {}
		self.current_subject = None
		self.current_object = None
		self.previously_mined_attributes = defaultdict(lambda: [])
		self.previously_mined_next_objects = defaultdict(lambda: [])

	def step(self, attribute_action, predicate_action, next_object_action):
		# should return reward_attribute, reward_predicate, and 
		# reward_next_object, and boolean indicating whether done
		reward_attribute, reward_predicate, reward_next_object = -1, -1, -1 
		pred_attribute_name = self.graph.attribute_nodes[attribute_action].name if attribute_action != None else None
		pred_predicate_name = self.graph.predicate_nodes[predicate_action].name if predicate_action != None else None
		if attribute_action != None:
			self.add_attribute(self.current_subject, attribute_action)
		if pred_predicate_name != None:
			self.add_predicate(self.current_subject, pred_predicate_name, self.current_object)
		self.objects_explored_per_subject[self.current_subject].append(self.current_object)
               
		gt_subject_index = self.overlaps(self.current_subject)
		if gt_subject_index != -1: #overlap
			if "attributes" in self.gt_scene_graph["labels"]["objects"][gt_subject_index] and pred_attribute_name in self.gt_scene_graph["labels"]["objects"][gt_subject_index]["attributes"]:
				reward_attribute = 1
			gt_object_index = self.overlaps(self.current_object)
			if gt_object_index != -1:
				for relationship_dict in self.gt_scene_graph["labels"]["relationships"]:
					if pred_predicate_name == relationship_dict["predicate"] and \
						gt_subject_index == relationship_dict["subject_id"] and \
						gt_object_index == relationship_dict["object_id"]:
						reward_predicate = 1
						break

		if next_object_action != None and next_object_action < len(self.entity_proposals):
			gt_new_object_index = self.overlaps(next_object_action)
			#self.explored_entities.append(new_object_index)
			if gt_new_object_index != -1:
				if gt_new_object_index not in self.explored_entities:
					reward_next_object = 5
			self.current_object = next_object_action
		else:
			self.current_subject = None
		return reward_attribute, reward_predicate, reward_next_object, self.is_done()

	def overlaps(self, entity_id):
		entity = self.current_scene_graph["objects"][entity_id]
		gt_index = -1
		for index, obj_dict in enumerate(self.gt_scene_graph["labels"]["objects"]):
			name1 = obj_dict["name"] if "name" in obj_dict else obj_dict["names"][0]
			name2 = entity["name"] if "name" in entity else entity["names"][0]
			if name1 == name2:
				if self.bbox_overlap([entity["x"], entity["y"], entity["w"], entity["h"]],\
					[obj_dict["x"], obj_dict["y"], obj_dict["w"], obj_dict["h"]]) > 0.5:
					gt_index = index
				break
		return index

	'''
	Intersection over union code from https://gist.github.com/vierja/38f93bb8c463dce5500c0adf8648d371
	'''
	def bbox_overlap(self, box1, box2):
		bx1,by1,bw1,bh1 = box1
		bx2,by2,bw2,bh2 = box2
		# determine the (x, y)-coordinates of the intersection rectangle
		x1 = max(bx1, bx2)
		y1 = max(by1, by2)
		x2 = min(bx1+bw1, bx2+bw2)
		y2 = min(by1+bh1, by2+bh2)

		# compute the area of intersection rectangle
		interArea = (x2 - x1)*(y2 - y1)

		# compute the area of both the prediction and ground-truth
		# rectangles
		boxAArea = bw1*bh1
		boxBArea = bw2*bh2

		# compute the intersection over union by taking the intersection
		# area and dividing it by the sum of prediction + ground-truth
		# areas - the interesection area
		iou = interArea / float(boxAArea + boxBArea - interArea)

		# return the intersection over union value
		return iou
