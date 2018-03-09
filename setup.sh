#!/usr/bin/env bash

# create json files with counts
python get_counts.py

# create semantic action graph
python create_semantic_action_graph.py -e
python create_semantic_action_graph.py -p
python create_semantic_action_graph.py -a
