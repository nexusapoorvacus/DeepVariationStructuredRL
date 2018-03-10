from collections import defaultdict

class ImageState():

	def __init__(self, image_name, gt_scene_graph, image_feature, entity_features, entity_proposals, entity_classes, entity_scores):
		self.image_name = image_name
		self.gt_scene_graph = gt_scene_graph
		self.current_scene_graph = {"relationships": dict(), "objects": dict()}
                self.current_scene_graph["objects"] = {"object_id": [], "x" = [], "y" = [], "w": [], "h": [], "name": [], "attributes"= [], "score": []}
                self.current_scene_graph["relationships"] = {"relationship_id":[], "predicate" :[], "subject_id": [], "object_id": []}
		# number of objects to explore for a subject
		self.max_objects_to_explore = 5
		self.objects_explored_per_subject = {}		

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
                    object_dict = self.current_scene_graph["objects"]
                    for i in range(len(entity_classes)):
                        object_dict["object_id"].append(0) #TODO:get id from class
                        points = entity_proposals[i]
                        object_dict["x"].append(int(points[0]))
                        object_dict["y"].append(int(points[1]))
                        object_dict["w"].append(int(points[3] - points[1]))
                        object_dict["h"].append(int(points[4] - points[2]))
                        object_dict["name"].append(entity_classes[i])
                        object_dict["score"].append(entity_scores[i])
                        object_dict["attributes"].append([])
		def add_attribute(self, subject_id, attribute_id):
			# NOTE: subject_id is the id for this particular image
			#	attribute_id is the id from the SAG
		        object_dict = self.current_scene_graph["objects"]
                        index = object_dict["object_id"].index(subject_id)
                        object_dict["attributes"][index].append(attribute_id)

		def add_predicate(self, subject_id, predicate_id, object_id):
			# NOTE: subject_id and object_id are ids for this particular image
			#	predicate_id is the id from the SAG
		        relationship_dict = self.current_scene_graph["relationships"]
                        relationship_dict["relationship_id"].append(predicate_id)
                        relationship_dict["predicate"].append("")
                        relationship_dict["subject_id"].append(subject_id)
                        relationship_dict["object_id"].append(object_id)

		def is_done(self):
			# returns true if we are done building a scene graph for this image
			pass		

		def reset(self):
			self.current_scene_graph = {"relationships": dict(), "objects":dict()}
			self.current_scene_graph["objects"] = {"object_id": [], "x" = [], "y" = [], "w": [], "h": [], "name": [], "attributes"= [], "score": []}
                        self.current_scene_graph["relationships"] = {"relationship_id":[], "predicate" :[], "subject_id": [], "object_id": []}
                        self.explored_entities = []
			self.object_counts_per_subject = {}
			self.current_subject = None
			self.current_object = None
			self.previously_mined_attributes = defaultdict(lambda: [])
			self.previously_mined_next_objects = defaultdict(lambda: [])

		def step(self, attribute_action, predicate_action, next_object_action):
			# should return reward_attribute, reward_predicate, and 
			# reward_next_object, and boolean indicating whether done
                        object_dict = self.current_scene_graph["objects"]
                        relationship_dict = self.current_scene_graph["relationships"]
                        reward_attribute, reward_predicate, reward_next_object = -1, -1, -1 
                        subject_overlap, subject_index = self.overlaps(self.current_subject)
                        if subject_overlap:
                            if attribute_action in self.gt_scene_graph["objects"]["attributes"][index]:
                                reward_attribute = 1
                            object_overlap, object_index = self.overlaps(self.current_object)
                            if object_overlap:
                                object_indices = self.current_scene_graph["relationships"]["subject_id"].index(subject_id)
                                if object_index in object_indices and self.current_scene_graph["relationships"]["predicate"][object_index] == predicate_action:
                                    reward_predicate = 1
                        new_object_overlap, new_object_index = self.overlaps(self.next_object_action)
                        if new_object_overlap:
                            if new_object_index not in self.explored_entities:
                                reward_next_object = 5

                        return reward_attribute, reward_predicate, reward_next_object
                
                def overlaps(entity_id):
                    index = self.current_scene_graph["objects"]["object_id"].index(entity_id)
                    entity_class = self.current_scene_graph["objects"]
                    return 
