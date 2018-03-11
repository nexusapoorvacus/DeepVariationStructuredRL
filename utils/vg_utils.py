from PIL import Image

OBJECT_ALIAS_FILE = "data/raw_data/object_alias.txt"
PREDICATE_ALIAS_FILE = "data/raw_data/relationship_alias.txt"

def entity_to_aliases(entity):
	if type(entity) == list:
		entity = entity[0]
	if entity in OBJECT_ALIASES:
		return OBJECT_ALIASES[entity]
	return (entity,)

def predicate_to_aliases(predicate):
	if type(predicate) == list:
		predicate = predicate[0]
	if predicate in PREDICATE_ALIASES:
		return PREDICATE_ALIASES[predicate]
	return (predicate,)

def find_object_neighbors(subject_bbox, entity_proposals, previously_mined_objects=[]):
	neighboring_objects = []
	x, y, w, h = subject_bbox
	for o, obj in enumerate(entity_proposals):
		if o not in previously_mined_objects:
			if subject_bbox != (obj[0], obj[1], obj[2], obj[3]):
				if abs(obj[0] - x) < 0.5 * (obj[2] + w) and abs(obj[1] - y) < 0.5 * (obj[3] + h):
					neighboring_objects.append(o)
	return neighboring_objects

def crop_box(image_dir, image_name, box):
	image = Image.open(image_dir + image_name)
	image = np.asarray(image)

	crop_box = image[box[1]:box[3], box[0]:box[2], :]
	crop_box = Image.fromarray(crop_box)
	return crop_box

# HELPER FUNCTIONS

def read_aliases(filename):
	alias_file = open(filename, "r")
	all_aliases = {}
	for line in alias_file.readlines():
		aliases = line.replace("\n", "").split(",")
		for a in aliases:
			all_aliases[a] = tuple(aliases)
	return all_aliases

##########
## MAIN ##
##########

# load predicate alias file
PREDICATE_ALIASES = read_aliases(PREDICATE_ALIAS_FILE)
# load object alias file
OBJECT_ALIASES = read_aliases(OBJECT_ALIAS_FILE)
