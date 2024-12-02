import argparse
import json
import os
import random
import re

from tqdm import tqdm
import networkx as nx
import pickle as pkl


node_type_dict = {}
edge_type_dict = {}
node_type_cnt = 0
edge_type_cnt = 0

# Metadata for 3 Darpa data sets
metadata = {
    'trace':{
        'train': ['ta1-trace-e3-official-1.json', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3'],
        'test': ['ta1-trace-e3-official-1.json', 'ta1-trace-e3-official-1.json.1', 'ta1-trace-e3-official-1.json.2', 'ta1-trace-e3-official-1.json.3', 'ta1-trace-e3-official-1.json.4']
    },
    'theia':{
            'train': ['ta1-theia-e3-official-6r.json', 'ta1-theia-e3-official-6r.json.1', 'ta1-theia-e3-official-6r.json.2', 'ta1-theia-e3-official-6r.json.3'],
            'test': ['ta1-theia-e3-official-6r.json.8']
    },
    'cadets':{
            'train': ['ta1-cadets-e3-official.json','ta1-cadets-e3-official.json.1', 'ta1-cadets-e3-official.json.2', 'ta1-cadets-e3-official-2.json.1'],
            'test': ['ta1-cadets-e3-official-2.json']
    }
}

# Regular expressions to match various patterns in the dataset
pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_type = re.compile(r'type\":\"(.*?)\"')
pattern_time = re.compile(r'timestampNanos\":(.*?),')
pattern_file_name = re.compile(r'map\":\{\"path\":\"(.*?)\"')
pattern_process_name = re.compile(r'map\":\{\"name\":\"(.*?)\"')
pattern_netflow_object_name = re.compile(r'remoteAddress\":\"(.*?)\"')


def read_single_graph(dataset, malicious, path, test=False):

    # Global counters for node and edge types
    global node_type_cnt, edge_type_cnt

    # Initialize a directed graph
    g = nx.DiGraph()

    # Data file path setup
    print('converting {} ...'.format(path))
    path = '../data/{}/'.format(dataset) + path + '.txt'

    # Open the data file for reading
    f = open(path, 'r')
    lines = []

    # For every line from the file
    for l in f.readlines():

        # Get the tokens
        split_line = l.split('\t')
        src, src_type, dst, dst_type, edge_type, ts = split_line
        ts = int(ts)

        # For training mode
        if not test:

            # If source or destination is malicious
            if src in malicious or dst in malicious:

                if src in malicious and src_type != 'MemoryObject':
                    continue
                if dst in malicious and dst_type != 'MemoryObject':
                    continue

        # Source node type not in dictionary
        if src_type not in node_type_dict:
            node_type_dict[src_type] = node_type_cnt
            node_type_cnt += 1

        # Destination node type not in dictionary
        if dst_type not in node_type_dict:
            node_type_dict[dst_type] = node_type_cnt
            node_type_cnt += 1

        # Edge type not in dictionary
        if edge_type not in edge_type_dict:
            edge_type_dict[edge_type] = edge_type_cnt
            edge_type_cnt += 1

        # For edge types READ, RECV, LOAD, we add a line in the line list
        if 'READ' in edge_type or 'RECV' in edge_type or 'LOAD' in edge_type:
            lines.append([dst, src, dst_type, src_type, edge_type, ts])
        else:
            lines.append([src, dst, src_type, dst_type, edge_type, ts])

    # Sort lines based on timestamp
    lines.sort(key=lambda l: l[5])

    # Dictionaries for node mapping
    node_map = {}
    node_type_map = {}
    node_cnt = 0
    node_list = []

    # For every line from our line list
    for l in lines:

        # Get the tokens from the line
        src, dst, src_type, dst_type, edge_type = l[:5]
        src_type_id = node_type_dict[src_type]
        dst_type_id = node_type_dict[dst_type]
        edge_type_id = edge_type_dict[edge_type]

        # Source node is not in the graph
        if src not in node_map:
            node_map[src] = node_cnt

            # Add the node
            g.add_node(node_cnt, type=src_type_id)
            node_list.append(src)
            node_type_map[src] = src_type
            node_cnt += 1
        
        # Destination node is not in the graph
        if dst not in node_map:
            node_map[dst] = node_cnt

            # Add the node
            g.add_node(node_cnt, type=dst_type_id)
            node_type_map[dst] = dst_type
            node_list.append(dst)
            node_cnt += 1
        
        # Edge is not in the graph
        if not g.has_edge(node_map[src], node_map[dst]):

            # Add the edge
            g.add_edge(node_map[src], node_map[dst], type=edge_type_id)
    
    # Return the node map as dictionary, and the graph
    return node_map, g


def preprocess_dataset(dataset):

    # Dictionaries to maintain node type and node name maps
    id_nodetype_map = {}
    id_nodename_map = {}

    # For every file in the data directory
    for file in os.listdir('../data/{}/'.format(dataset)):

        # Make sure this is a json file to be read, not txt, names, types or metadata
        if 'json' in file and not '.txt' in file and not 'names' in file and not 'types' in file and not 'metadata' in file:

            # Reading the file
            print('reading {} ...'.format(file))
            f = open('../data/{}/'.format(dataset) + file, 'r', encoding='utf-8')

            # For every line in the file
            for line in tqdm(f):

                # Ignore the following records
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line: continue
                if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line: continue
                if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line: continue

                # Parse uuid, subject_type
                if len(pattern_uuid.findall(line)) == 0: print(line)
                uuid = pattern_uuid.findall(line)[0]
                subject_type = pattern_type.findall(line)

                # Subject type
                if len(subject_type) < 1:
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        subject_type = 'MemoryObject'
                    if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        subject_type = 'NetFlowObject'
                    if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        subject_type = 'UnnamedPipeObject'
                else:
                    subject_type = subject_type[0]

                # Skip the trivial record
                if uuid == '00000000-0000-0000-0000-000000000000' or subject_type in ['SUBJECT_UNIT']:
                    continue

                # Node name mapping based on subject type
                id_nodetype_map[uuid] = subject_type

                # It is a file
                if 'FILE' in subject_type and len(pattern_file_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_file_name.findall(line)[0]
                
                # Subject process
                elif subject_type == 'SUBJECT_PROCESS' and len(pattern_process_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_process_name.findall(line)[0]
                
                # Netflow object
                elif subject_type == 'NetFlowObject' and len(pattern_netflow_object_name.findall(line)) > 0:
                    id_nodename_map[uuid] = pattern_netflow_object_name.findall(line)[0]
    
    # Loop over the dataset's metadata keys
    for key in metadata[dataset]:

        # for each json file
        for file in metadata[dataset][key]:

            # If there is a corresponding txt file, skip
            if os.path.exists('../data/{}/'.format(dataset) + file + '.txt'):
                continue

            # Open the json file to read, and a new corresponding txt file to write
            f = open('../data/{}/'.format(dataset) + file, 'r', encoding='utf-8')
            fw = open('../data/{}/'.format(dataset) + file + '.txt', 'w', encoding='utf-8')
            print('processing {} ...'.format(file))

            # For every line in the json file
            for line in tqdm(f):

                # We are interested in this particular event type only
                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:

                    # Parse edge type, timestamp, source id
                    edgeType = pattern_type.findall(line)[0]
                    timestamp = pattern_time.findall(line)[0]
                    srcId = pattern_src.findall(line)

                    # Skip source 0
                    if len(srcId) == 0: continue
                    srcId = srcId[0]

                    # Skip if source id is not in our node type map
                    if not srcId in id_nodetype_map:
                        continue

                    srcType = id_nodetype_map[srcId]

                    # Processing destination 1
                    dstId1 = pattern_dst1.findall(line)

                    # If it is available
                    if len(dstId1) > 0 and dstId1[0] != 'null':
                        dstId1 = dstId1[0]

                        if not dstId1 in id_nodetype_map:
                            continue

                        # Get the type
                        dstType1 = id_nodetype_map[dstId1]

                        # Make the edge record for src to dst1
                        this_edge1 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId1) + '\t' + str(
                            dstType1) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        
                        # Write the edge to file
                        fw.write(this_edge1)

                    # Processing destination 2
                    dstId2 = pattern_dst2.findall(line)

                    # If it is available
                    if len(dstId2) > 0 and dstId2[0] != 'null':
                        dstId2 = dstId2[0]

                        if not dstId2 in id_nodetype_map.keys():
                            continue
                        
                        # Get the type
                        dstType2 = id_nodetype_map[dstId2]

                        # Make the edge record for src to dst2
                        this_edge2 = str(srcId) + '\t' + str(srcType) + '\t' + str(dstId2) + '\t' + str(
                            dstType2) + '\t' + str(edgeType) + '\t' + str(timestamp) + '\n'
                        
                        # Write the edge to file
                        fw.write(this_edge2)
            fw.close()
            f.close()
    
    # Writing the node names to json
    if len(id_nodename_map) != 0:
        fw = open('../data/{}/'.format(dataset) + 'names.json', 'w', encoding='utf-8')
        json.dump(id_nodename_map, fw)
    
    # Writing the node types to json
    if len(id_nodetype_map) != 0:
        fw = open('../data/{}/'.format(dataset) + 'types.json', 'w', encoding='utf-8')
        json.dump(id_nodetype_map, fw)


def read_graphs(dataset):

    # Open the file for malicious entries
    malicious_entities = '../data/{}/{}.txt'.format(dataset, dataset)
    f = open(malicious_entities, 'r')

    # This set will remove the duplicates, if any
    malicious_entities = set()

    # Read the lines, strip and add to the set
    for l in f.readlines():
        malicious_entities.add(l.lstrip().rstrip())

    # Get the dataset preprocessed
    preprocess_dataset(dataset)

    # Training graphs list
    train_gs = []

    # Go over the training files and read the training graphs
    for file in metadata[dataset]['train']:
        _, train_g = read_single_graph(dataset, malicious_entities, file, False)
        train_gs.append(train_g)
    
    # Testing graphs list and node map
    test_gs = []
    test_node_map = {}
    count_node = 0

    # Go over the training files and read the training graphs and node map
    for file in metadata[dataset]['test']:

        node_map, test_g = read_single_graph(dataset, malicious_entities, file, True)

        # Ensure the sizes match or error out
        assert len(node_map) == test_g.number_of_nodes()
        test_gs.append(test_g)

        # For every key in node map
        for key in node_map:
            
            if key not in test_node_map:
                test_node_map[key] = node_map[key] + count_node
        
        # Aggregate the number of nodes
        count_node += test_g.number_of_nodes()

    if os.path.exists('../data/{}/names.json'.format(dataset)) and os.path.exists('../data/{}/types.json'.format(dataset)):
        
        # Open for read the json file for names
        with open('../data/{}/names.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodename_map = json.load(f)

        # Open for read the json file for types
        with open('../data/{}/types.json'.format(dataset), 'r', encoding='utf-8') as f:
            id_nodetype_map = json.load(f)

        # Open a new text file for writing malicious names
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')


        final_malicious_entities = []
        malicious_names = []

        # For every malicious entity
        for e in malicious_entities:
            
            # Ensure it is of appropriate type or ignore
            if e in test_node_map and e in id_nodetype_map and id_nodetype_map[e] != 'MemoryObject' and id_nodetype_map[e] != 'UnnamedPipeObject':
                final_malicious_entities.append(test_node_map[e])
                
                # If the name is in the dictionary, write it to output file
                if e in id_nodename_map:
                    malicious_names.append(id_nodename_map[e])
                    f.write('{}\t{}\n'.format(e, id_nodename_map[e]))
                else:
                    # Write malicious entity to output file
                    malicious_names.append(e)
                    f.write('{}\t{}\n'.format(e, e))
    else:
        # Open a new text file for writing malicious names
        f = open('../data/{}/malicious_names.txt'.format(dataset), 'w', encoding='utf-8')
        final_malicious_entities = []
        malicious_names = []

        # For each malicious entity
        for e in malicious_entities:

            # If it is in the test nodes, write malicious entity to output file
            if e in test_node_map:
                final_malicious_entities.append(test_node_map[e])
                malicious_names.append(e)
                f.write('{}\t{}\n'.format(e, e))

    # Save the pickles to disk for training, testing graphs, final malicious entities
    pkl.dump((final_malicious_entities, malicious_names), open('../data/{}/malicious.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(train_g) for train_g in train_gs], open('../data/{}/train.pkl'.format(dataset), 'wb'))
    pkl.dump([nx.node_link_data(test_g) for test_g in test_gs], open('../data/{}/test.pkl'.format(dataset), 'wb'))


if __name__ == '__main__':

    # Parse the arguments for this script
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="trace")
    args = parser.parse_args()

    # Dataset must be one of these three
    if args.dataset not in ['trace', 'theia', 'cadets']:
        raise NotImplementedError
    read_graphs(args.dataset)

