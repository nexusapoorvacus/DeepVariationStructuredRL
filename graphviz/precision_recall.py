import json

total_rel = 0
pred_rel = 0
total_attr = 0
pred_attr = 0
curr_rel = 0
curr_attr = 0

for i in range(10):
    curr = json.load(open('curr_graph_%i.json'%i))
    gt = json.load(open('gt_graph_%i.json'%i))

    for gt_rel in gt['relationships']:
        if gt_rel in curr['relationships']:
            pred_attr += 1
        total_attr += 1
    curr_rel += len(curr['relationships'])
    for gt_attr in gt['attributes']:
        if gt_attr in curr['attributes']:
            pred_rel += 1
        total_rel += 1
    curr_attr += len(curr['attributes'])

print (total_rel, pred_rel, curr_rel)
print ("Recall rel: ", pred_rel*1.0/total_rel)
print ("precision rel: ", pred_rel*1.0/curr_rel)

print (total_attr, pred_attr, curr_attr)
print ("Recall attr: ", pred_attr*1.0/total_attr)
print ("precision attr: ", pred_attr*1.0/curr_attr)

total = total_attr+total_rel
pred = pred_attr+pred_rel
curr_total = curr_rel + curr_attr
print (total, pred, curr_total)
print ("Recall total: ", pred*1.0/total)
print ("Precision total: ", pred*1.0/curr_total)
        


