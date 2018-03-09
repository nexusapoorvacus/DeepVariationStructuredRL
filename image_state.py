from collections import defaultdict

class ImageState():

	def __init__(self, image_name, gt_scene_graph, image_feature, entity_features, entity_proposals, entity_classes, entity_scores):
		self.image_name = image_name
		self.gt_scene_graph = gt_scene_graph
		self.current_scene_graph = {"relationships": [], "entities": []}
		
		# number of objects to explore for a subject
		self.max_objects_to_explore = 5
		
		# entities chosen to explore so far
		self.explored_entities = []
		
		# next object to explore
		self.next_object = None
		
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
			pass
	
		def add_attribute(self, subject_id, attribute_id):
			# NOTE: subject_id is the id for this particular image
			#	attribute_id is the id from the SAG
			pass

		def add_predicate(self, subject_id, predicate_id, object_id):
			# NOTE: subject_id and object_id are ids for this particular image
			#	predicate_id is the id from the SAG
			pass

		def is_done(self):
			# returns true if we are done building a scene graph for this image
		
		def reset(self):
			self.current_scene_graph = {"relationships": [], "entities":[]}
			self.explored_entities = []
			self.next_object = None
			self.previously_mined_attributes = defaultdict(lambda: [])
			self.previously_mined_next_objects = defaultdict(lambda: [])

		def step(self, attribute_action, predicate_action, next_object_action):
			# should return reward_attribute, reward_predicate, and 
			# reward_next_object, and boolean indicating whether done
			pass

