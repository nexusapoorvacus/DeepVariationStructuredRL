import pickle
import json

img = pickle.load(open('image_states.pickle', 'rb'))

current = img['curr']
gt = img['gt']


def create_dict(graph, graph_name):
    old_to_new_index= {}
    curr_dict = {"objects": [], "attributes":[], "relationships":[]}
    for obj_dict in graph['objects']:
        name = obj_dict["name"] if "name" in obj_dict else obj_dict["names"][0]
        obj_id = obj_dict['object_id']
        old_to_new_index[obj_id] = len(curr_dict['objects'])
        curr_dict['objects'].append({"name":name})
        if 'attributes' in obj_dict.keys():
            for attr in obj_dict['attributes']:
                curr_dict['attributes'].append({"attribute": attr, "object": old_to_new_index[obj_id]})
                break
    for rel_dict in graph["relationships"]:
        object_id =  old_to_new_index[rel_dict["object_id"]]
        subject_id = old_to_new_index[rel_dict["subject_id"]]
        if object_id != subject_id:
            predicate =rel_dict['predicate'][0] if graph_name == "curr" else rel_dict['predicate']
            curr_dict['relationships'].append({"predicate": predicate, "object": object_id, "subject": subject_id})

    return curr_dict

for i in range(len(current)):
    curr_graph = current[i]
    gt_graph = gt[i]['labels']
    curr_dict = create_dict(curr_graph, "curr")
    gt_dict = create_dict(gt_graph, "gt")
    with open('curr_graph_%d.json'%i, 'w') as outfile:
        json.dump(curr_dict, outfile)
    with open('gt_graph_%d.json'%i, 'w') as outfile:
        json.dump(gt_dict, outfile)

