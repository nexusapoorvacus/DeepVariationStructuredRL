OBJECT_ALIAS_FILE = "data/object_alias.txt"
PREDICATE_ALIAS_FILE = "data/relationship_alias.txt"

def entity_to_aliases(entity):
	if type(entity) == list:
		e = entity[0]
	else:
		e = entity
	if e in OBJECT_ALIASES:
		return OBJECT_ALIASES[e]
	return tuple(entity)

def predicate_to_aliases(predicate):
	if predicate in PREDICATE_ALIASES:
		return PREDICATE_ALIASES[predicate]
	return tuple(predicate)

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

