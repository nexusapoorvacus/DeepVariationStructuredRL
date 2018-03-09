class ImageState():

	def __init__(self, image_name, gt_scene_graph):
		self.image_name = image_name
		self.gt_scene_graph = gt_scene_graph
		self.current_scene_graph = {"relationships": [], "entities": []}
		# number of objects to explore for a subject
		self.max_objects_to_explore = 5
		# entities chosen to explore so far
		self.explored_entities = []
		# next object to explore
		self.next_object = None

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

