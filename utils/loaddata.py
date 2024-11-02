import pickle as pkl
import time
import torch.nn.functional as F
import dgl
import networkx as nx
import json
from tqdm import tqdm
import os


class StreamspotDataset(dgl.data.DGLDataset):
    def process(self):
        pass

    def __init__(self, name):
        super(StreamspotDataset, self).__init__(name=name)
        if name == 'streamspot':
            path = './data/streamspot'
            num_graphs = 600
            self.graphs = []
            self.labels = []
            print('Loading {} dataset...'.format(name))
            for i in tqdm(range(num_graphs)):
                idx = i
                g = dgl.from_networkx(
                    nx.node_link_graph(json.load(open('{}/{}.json'.format(path, str(idx + 1))))),
                    node_attrs=['type'],
                    edge_attrs=['type']
                )
                self.graphs.append(g)
                if 300 <= idx <= 399:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


class WgetDataset(dgl.data.DGLDataset):
    def process(self):
        pass

    def __init__(self, name):
        super(WgetDataset, self).__init__(name=name)
        if name == 'wget':
            path = './data/wget/final'
            num_graphs = 150
            self.graphs = []
            self.labels = []
            print('Loading {} dataset...'.format(name))
            for i in tqdm(range(num_graphs)):
                idx = i
                g = dgl.from_networkx(
                    nx.node_link_graph(json.load(open('{}/{}.json'.format(path, str(idx))))),
                    node_attrs=['type'],
                    edge_attrs=['type']
                )
                self.graphs.append(g)
                if 0 <= idx <= 24:
                    self.labels.append(1)
                else:
                    self.labels.append(0)
        else:
            raise NotImplementedError

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


# A function that determines the dataset to load "streamspot or wget" (are they workin on two datasets only ? Answer: for entity level yes)
def load_rawdata(name):
    if name == 'streamspot':
        path = './data/streamspot'
        if os.path.exists(path + '/graphs.pkl'):
            print('Loading processed {} dataset...'.format(name))
            raw_data = pkl.load(open(path + '/graphs.pkl', 'rb'))
        else:
            raw_data = StreamspotDataset(name)
            pkl.dump(raw_data, open(path + '/graphs.pkl', 'wb'))
    elif name == 'wget':
        path = './data/wget'
        if os.path.exists(path + '/graphs.pkl'):
            print('Loading processed {} dataset...'.format(name))
            raw_data = pkl.load(open(path + '/graphs.pkl', 'rb'))
        else:
            raw_data = WgetDataset(name)
            pkl.dump(raw_data, open(path + '/graphs.pkl', 'wb'))
    else:
        raise NotImplementedError
    return raw_data


def load_batch_level_dataset(dataset_name):

    # load raw dataset as a list of data
    dataset = load_rawdata(dataset_name)

    # select the fisrt graph 
    graph, _ = dataset[0]
    node_feature_dim = 0

    # loop on each grpah in the dataset and sets node_feature_dim variable to the maximum value found
    # This captures the number of unique node feature types based in "type" label in the dataset
    for g, _ in dataset:
        node_feature_dim = max(node_feature_dim, g.ndata["type"].max().item())

    edge_feature_dim = 0

    # loop on each grpah in the dataset and sets edge_feature_dim variable to the maximum value found
    # This captures the number of unique edge feature types based in "type" label in the dataset
    for g, _ in dataset:
        edge_feature_dim = max(edge_feature_dim, g.edata["type"].max().item())

    # adds extra dimension for the node and edge size
    node_feature_dim += 1
    edge_feature_dim += 1
    full_dataset = [i for i in range(len(dataset))]
    train_dataset = [i for i in range(len(dataset)) if dataset[i][1] == 0]
    print('[n_graph, n_node_feat, n_edge_feat]: [{}, {}, {}]'.format(len(dataset), node_feature_dim, edge_feature_dim))

    return {'dataset': dataset,
            'train_index': train_dataset,
            'full_index': full_dataset,
            'n_feat': node_feature_dim,
            'e_feat': edge_feature_dim}

# Transforms a graph by encoding node and edge types into one-hot feature vectors.
def transform_graph(g, node_feature_dim, edge_feature_dim):
    # Clone the original graph to create a new graph object
    new_g = g.clone()

    # g.ndata["type"].view(-1) reshapes the node type tensor to a 1D tensor
    # F.one_hot creates one-hot encoded vectors for each node type
    new_g.ndata["attr"] = F.one_hot(g.ndata["type"].view(-1), num_classes=node_feature_dim).float()

    # g.ndata["type"].view(-1) reshapes the edge type tensor to a 1D tensor
    # F.one_hot creates one-hot encoded vectors for each edge type
    new_g.edata["attr"] = F.one_hot(g.edata["type"].view(-1), num_classes=edge_feature_dim).float()
    return new_g


def preload_entity_level_dataset(path):
    """
    Preloads an entity-level dataset from the specified path. If the metadata file does not exist,
    it transforms the dataset from NetworkX graphs to DGL graphs, computes feature dimensions, 
    and saves the transformed graphs and metadata.
    """

    # Construct the full path to the dataset
    path = './data/' + path

    # Check if the metadata file already exists
    if os.path.exists(path + '/metadata.json'):
        pass
    else:
        print('transforming')

        # Load the training graphs from a pickle file and convert them from NetworkX to DGL format
        """
        raw: {
                "directed": false,
                "graph": {},
                "nodes": [
                    {"id": 0, "type": 1},
                    {"id": 1, "type": 2},
                    {"id": 2, "type": 1}
                ],
                "links": [
                    {"source": 0, "target": 1, "type": 0},
                    {"source": 1, "target": 2, "type": 1}
                ]
                }
        Networx Graph: 
                Nodes: [(0, {'type': 1}), (1, {'type': 2}), (2, {'type': 1})]
                Edges: [(0, 1, {'type': 0}), (1, 2, {'type': 1})]
        DGL Graph: 
                Node Features: {'type': tensor([1, 2, 1])} (assuming type was a node attribute)
                Edge Features: {'type': tensor([0, 1])} (assuming type was an edge attribute)
        """
        train_gs = [dgl.from_networkx(
            nx.node_link_graph(g),
            node_attrs=['type'],  
            edge_attrs=['type']
        ) for g in pkl.load(open(path + '/train.pkl', 'rb'))]
        print('transforming')
        test_gs = [dgl.from_networkx(
            nx.node_link_graph(g),
            node_attrs=['type'],
            edge_attrs=['type']
        ) for g in pkl.load(open(path + '/test.pkl', 'rb'))]

        # Load the malicious data from a pickle file
        malicious = pkl.load(open(path + '/malicious.pkl', 'rb'))

        node_feature_dim = 0

        # Iterate through training graphs to find the maximum node type
        for g in train_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)

        # Iterate through test graphs to find the maximum node type
        for g in test_gs:
            node_feature_dim = max(g.ndata["type"].max().item(), node_feature_dim)

        node_feature_dim += 1 # Increment to account for zero-indexing
        edge_feature_dim = 0

        # Iterate through training graphs to find the maximum edge type
        for g in train_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)

        # Iterate through test graphs to find the maximum edge type
        for g in test_gs:
            edge_feature_dim = max(g.edata["type"].max().item(), edge_feature_dim)

        edge_feature_dim += 1 # Increment to account for zero-indexing

        # Transform test and training graphs to one-hot encoded representation
        result_test_gs = []
        for g in test_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)
            result_test_gs.append(g)
        result_train_gs = []
        for g in train_gs:
            g = transform_graph(g, node_feature_dim, edge_feature_dim)
            result_train_gs.append(g)

        # Create metadata dictionary with relevant dataset information
        metadata = {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'malicious': malicious,
            'n_train': len(result_train_gs),
            'n_test': len(result_test_gs)
        }
        with open(path + '/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        for i, g in enumerate(result_train_gs):
            with open(path + '/train{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)
        for i, g in enumerate(result_test_gs):
            with open(path + '/test{}.pkl'.format(i), 'wb') as f:
                pkl.dump(g, f)


def load_metadata(path):
    preload_entity_level_dataset(path)
    with open('./data/' + path + '/metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    return metadata


def load_entity_level_dataset(path, t, n):
    preload_entity_level_dataset(path)
    with open('./data/' + path + '/{}{}.pkl'.format(t, n), 'rb') as f:
        data = pkl.load(f)
    return data
