from abc import ABCMeta, abstractmethod

# GRAPH
class Semantic_Action_Graph():

	def __init__(self):
		# NODES
		# maps node id: node instance
		self.entity_nodes = {}
		self.predicate_nodes = {}
		self.attribute_nodes = {}
		
		self.entity_name_to_id = {}
		self.predicate_name_to_id = {}
		self.attribute_name_to_id = {}

	def get_entity_id(self, entity):
		if entity in self.entity_name_to_id:
			return self.entity_name_to_id[entity]
		return None

	def get_predicate_id(self, predicate):
		if predicate in self.predicate_name_to_id:
			return self.predicate_name_to_id[predicate]
		return None

	def get_attribute_id(self, attribute):
		if attribute in self.attribute_name_to_id:
			return self.attribute_name_to_id[attribute]
		return None
	
	def add_node(self, node, node_id):
		if isinstance(node, Entity_Node):
			self.entity_nodes[node_id] = node
		elif isinstance(node, Predicate_Node):
			self.predicate_nodes[node_id] = node
		elif isinstance(node, Attribute_Node):
			self.attribute_nodes[node_id] = node
		else:
			raise Exception("Trying to Add Invalid Node Type!")

	def get_entity_by_name(self, entity):
		if entity in self.entity_name_to_id:
			return self.entity_nodes[self.entity_name_to_id[entity]]
		return None

	def variation_based_traversal(self, subject_name, object_name, previously_mined_attributes=[], max_num_to_return=-1):
		subject_node = self.get_entity_by_name(subject_name)
		object_node = self.get_entity_by_name(object_name)
		if subject_node == None:
			return [], []
		attributes_to_return = {}
		for a_edge in subject_node.attribute_edges:
			if self.attribute_nodes[subject_node.attribute_edges[a_edge].attribute_id].ID not in previously_mined_attributes:
				attributes_to_return[subject_node.attribute_edges[a_edge].attribute_id] = subject_node.attribute_edges[a_edge].multiplicity
		predicates_to_return = {}
		if object_node != None:
			for p_edge in subject_node.predicate_edges:
				subject_node.attribute_edges[a_edge].multiplicity
				if subject_node.predicate_edges[p_edge].object_id == object_node.ID:
					predicates_to_return[subject_node.predicate_edges[p_edge].predicate_id] = subject_node.predicate_edges[p_edge].multiplicity
	
		attributes_to_return = sorted(attributes_to_return.items(), key=lambda x: -x[1])
		predicates_to_return = sorted(predicates_to_return.items(), key=lambda x: -x[1])
		attributes_to_return = [a[0] for a in attributes_to_return]
		predicates_to_return = [p[0] for p in predicates_to_return]

		if max_num_to_return == -1:
			return (attributes_to_return, predicates_to_return)
		return (attributes_to_return[:max_num_to_return], predicates_to_return[:max_num_to_return]) 

# NODES
class Node():
	
	__metaclass__ = ABCMeta

	def __init__(self, name, ID, graph):
		self.name = name
		self.ID = ID
		self.graph = graph

class Entity_Node(Node):

	def __init__(self, name, graph):
		ID = max(graph.entity_name_to_id.values())+1 if len(graph.entity_name_to_id) > 0 else 0
		Node.__init__(self, name, ID, graph)
		graph.entity_name_to_id[name] = ID
		self.graph.add_node(self, ID)

		# Dictionary of Edge Objects
		# key is a tuple of ids, value is Edge object
		self.predicate_edges = {}
		self.attribute_edges = {}

	def add_predicate_edge(self, edge):
		assert edge.subject_id == self.ID
		if (edge.subject_id, edge.predicate_id, edge.object_id) not in self.predicate_edges:
			self.predicate_edges[(edge.subject_id, edge.predicate_id, edge.object_id)] = edge
		edge.multiplicity += 1

	def get_predicate_edge(self, predicate_id, object_id):
		if (self.ID, predicate_id, object_id) in self.predicate_edges:
			return self.predicate_edges[(self.ID, predicate_id, object_id)]
		return None

	def add_attribute_edge(self, edge):
		assert edge.subject_id == self.ID
		if (edge.subject_id, edge.attribute_id) not in self.attribute_edges:
			self.attribute_edges[(edge.subject_id, edge.attribute_id)] = edge
		edge.multiplicity += 1

	def get_attribute_edge(self, attribute_id):
		if (self.ID, attribute_id) in self.attribute_edges:
			return self.attribute_edges[(self.ID, attribute_id)]
		return None 

class Predicate_Node(Node):

	def __init__(self, name, graph):
		ID = max(graph.predicate_name_to_id.values())+1 if len(graph.predicate_name_to_id) > 0 else 0
		Node.__init__(self, name, ID, graph)
		graph.predicate_name_to_id[name] = ID
	
		self.graph.add_node(self, ID)

class Attribute_Node(Node):

	def __init__(self, name, graph):
		ID = max(graph.attribute_name_to_id.values())+1 if len(graph.attribute_name_to_id) > 0 else 0
		Node.__init__(self, name, ID, graph)
		graph.attribute_name_to_id[name] = ID

		self.graph.add_node(self, ID)	

# EDGES
class Edge():

	__metaclass__ = ABCMeta

	def __init__(self):
		pass

class Predicate_Edge(Edge):

	def __init__(self, subject_id, predicate_id, object_id):
		self.subject_id = subject_id
		self.predicate_id = predicate_id
		self.object_id = object_id
		self.multiplicity = 0
		
class Attribute_Edge(Edge):

	def __init__(self, subject_id, attribute_id):
		self.subject_id = subject_id
		self.attribute_id = attribute_id
		self.multiplicity = 0
