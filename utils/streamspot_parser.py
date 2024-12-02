import networkx as nx
from tqdm import tqdm
import json

# Streamspot dataset raw path
raw_path = '../data/streamspot/'

NUM_GRAPHS = 600

# Predefined node, edge types
node_type_dict = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
edge_type_dict = ['i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                  'q', 't', 'u', 'v', 'w', 'y', 'z', 'A', 'C', 'D', 'E', 'G']

# Make respective dictionaries
node_type_set = set(node_type_dict)
edge_type_set = set(edge_type_dict)

count_graph = 0

# Open the tab separated file
with open(raw_path + 'all.tsv', 'r', encoding='utf-8') as f:

    # Read lines from file
    lines = f.readlines()

    # Make a new directed graph, empty node map
    g = nx.DiGraph()
    node_map = {}
    count_node = 0

    # For every line 
    for line in tqdm(lines):

        # Tokenize the the data
        src, src_type, dst, dst_type, etype, graph_id = line.strip('\n').split('\t')
        graph_id = int(graph_id)

        # A node has unknown type, skip
        if src_type not in node_type_set or dst_type not in node_type_set:
            continue
        
        # An edge has unknown type, skip
        if etype not in edge_type_set:
            continue

        # If graph id is different than graph count
        if graph_id != count_graph:
            count_graph += 1

            # For every node in the graph, set the type to its first index
            for n in g.nodes():
                g.nodes[n]['type'] = node_type_dict.index(g.nodes[n]['type'])

            # For every edge in the graph, set the type to its first index
            for e in g.edges():
                g.edges[e]['type'] = edge_type_dict.index(g.edges[e]['type'])

            # Open a json file for writing
            f1 = open(raw_path + str(count_graph) + '.json', 'w', encoding='utf-8')

            # Dump graph data to file
            json.dump(nx.node_link_data(g), f1)
            assert graph_id == count_graph

            # Reset the graph and node count
            g = nx.DiGraph()
            count_node = 0

        # Source not in node map, add it to graph, add to source map increment node count
        if src not in node_map:
            node_map[src] = count_node
            g.add_node(count_node, type=src_type)
            count_node += 1

        # Destination not in node map, add it to graph, add to destination map, increment node count
        if dst not in node_map:
            node_map[dst] = count_node
            g.add_node(count_node, type=dst_type)
            count_node += 1

        # Edge is not present in the graph, add it to graph
        if not g.has_edge(node_map[src], node_map[dst]):
            g.add_edge(node_map[src], node_map[dst], type=etype)

    # Increment graph count
    count_graph += 1

    # For every node in the graph, set the type to its first index
    for n in g.nodes():
        g.nodes[n]['type'] = node_type_dict.index(g.nodes[n]['type'])

    # For every edge in the graph, set the type to its first index
    for e in g.edges():
        g.edges[e]['type'] = edge_type_dict.index(g.edges[e]['type'])

    # Open a new json file for writing
    f1 = open(raw_path + str(count_graph) + '.json', 'w', encoding='utf-8')

    # Write to json file
    json.dump(nx.node_link_data(g), f1)
