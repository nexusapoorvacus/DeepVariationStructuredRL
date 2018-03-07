OBJECT_ALIAS_FILE = "data/object_alias.txt"
PREDICATE_ALIAS_FILE = "data/relationship_alias.txt"

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

