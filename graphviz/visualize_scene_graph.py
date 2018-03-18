"""Visualizes a scene graph stored in a json in a webpage.
"""

import argparse
import json
import os
import webbrowser


def generate_graph_js(graph, js_file):
    """Converts a json into a readable js object.

    Args:
        graph: A scene graph object.
        js_file: The javascript file to write to.
    """
    f = open(js_file, 'w')
    f.write('var graph = ' + json.dumps(graph))
    f.close()


def visualize_scene_graph(graph, js_file):
    """Creates an html visualization of the scene graph.

    Args:
        graph: A scene graph object.
        js_file: The javascript file to write to.
    """
    scene_graph = {'objects': [], 'attributes': [], 'relationships': []}
    for obj in graph['objects']:
        name = ''
        if 'name' in obj:
            name = obj['name']
        elif 'names' in obj and len(obj['names']) > 0:
            name = obj['names'][0]
        scene_graph['objects'].append({'name': name})
    scene_graph['attributes'] = graph['attributes']
    scene_graph['relationships'] = graph['relationships']
    generate_graph_js(scene_graph, js_file)
    webbrowser.open('file://' + os.path.realpath('graphviz.html'))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str,
                        default='example_graph.json',
                        help='Location of scene graph file to visualize.')
    parser.add_argument('--js-file', type=str,
                        default='scene_graph.js',
                        help='Temporary file generated to enable visualization.')
    args = parser.parse_args()
    graph = json.load(open(args.graph))
    visualize_scene_graph(graph, args.js_file)
