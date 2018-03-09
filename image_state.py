class ImageState():

	def __init__(self, image_name, gt_scene_graph):
		self.image_name = image_name
		self.gt_scene_graph = gt_scene_graph
		self.choose_new_subject = False
		self.current_scene_graph = {"relationships": [], "entities": []}
		self.explored_entities = []

		def add_entities(self, proposals):
			pass 
