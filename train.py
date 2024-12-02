import os
import random
import torch
import warnings
from tqdm import tqdm
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from torch.utils.data.sampler import SubsetRandomSampler
from dgl.dataloading import GraphDataLoader
from model.train import batch_level_train
from utils.utils import set_random_seed, create_optimizer
from utils.config import build_args
warnings.filterwarnings('ignore')

# Extracts a DataLoader for training data by shuffling the entries and creating a sampler.
def extract_dataloaders(entries, batch_size):
    # Shuffle the entries (entries are the train_endices extracted from raw dataset) randomly to ensure the model is trained on a diverse set of samples
    random.shuffle(entries)
    # Create a tensor of indices representing all entries in the dataset 
    train_idx = torch.arange(len(entries))
    # Create a random sampler that samples elements randomly from the dataset without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    # Create a DataLoader that loads batches of graphs using the defined sampler
    train_loader = GraphDataLoader(entries, batch_size=batch_size, sampler=train_sampler)
    # Return the DataLoader for training
    return train_loader


def main(main_args):
    # checking for gpu
    device = main_args.device if main_args.device >= 0 else "cpu"
    dataset_name = main_args.dataset

    # determining model parameters based on the type of dataset and appending them to the main argument vector
    if dataset_name == 'streamspot':
        main_args.num_hidden = 256
        main_args.max_epoch = 5
        main_args.num_layers = 4
    
    # wget
    elif dataset_name == 'wget':
        main_args.num_hidden = 256
        main_args.max_epoch = 2
        main_args.num_layers = 4
    
    # Entity level (DARPA)
    else:
        main_args.num_hidden = 64
        main_args.max_epoch = 50
        main_args.num_layers = 3

    # essential for experiments reproducibility. "generating the same random numbers everytime such as initial trainable parameters"
    set_random_seed(0)

    # for streamspot and wget, train model on batch level mode, otherwise train the model on entity level mode
    if dataset_name == 'streamspot' or dataset_name == 'wget':
        # selecting batch special batch size for streamspot 
        if dataset_name == 'streamspot':
            batch_size = 12
        else:
            batch_size = 1
        """
        dataset in form: 
        {
            'dataset': dataset, -> the raw dataset
            'train_index': train_dataset, ->  indices where the second element in the dataset tuple (the label) is 0, indicating training samples.
            'full_index': full_dataset, ->  list of all dataset indices
            'n_feat': node_feature_dim, -> the number of unique node feature
            'e_feat': edge_feature_dim -> the number of unique edge feature
        } 
        """
        dataset = load_batch_level_dataset(dataset_name)

        # selecting the number of node features
        n_node_feat = dataset['n_feat']
        # selecting the number of edge features
        n_edge_feat = dataset['e_feat']
        # selecting the graphs from raw dataset
        graphs = dataset['dataset']
        # selecting the training indices
        train_index = dataset['train_index']

        # passing the dimensions of nodes and edges in the main arguemnt vector
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat

        # initial the autoencoder model
        model = build_model(main_args)

        # moving it to the gpu
        model = model.to(device)

        # select the optimizer based on user choice passed in the argument vector
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)

        # it returns model trained on streamspot or wget dataset, on batch level not entity level training
        model = batch_level_train(model, graphs, (extract_dataloaders(train_index, batch_size)),
                                  optimizer, main_args.max_epoch, device, main_args.n_dim, main_args.e_dim)

        # save the model
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))
    
    # Darpa datasets
    else:
        """
        metadata = 
        {
            'node_feature_dim': node_feature_dim,
            'edge_feature_dim': edge_feature_dim,
            'malicious': malicious,
            'n_train': len(result_train_gs),
            'n_test': len(result_test_gs)
        }
        """

        # Load dataset's metadata
        metadata = load_metadata(dataset_name)

        # Assign node and edge feature dimensions
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']

        # Build the autoencoder model using parsed arguments
        model = build_model(main_args)
        
        # CPU or GPU assignment
        model = model.to(device)
        
        # Set the model in training mode
        model.train()

        # Initialize the model optimizer
        optimizer = create_optimizer(main_args.optimizer, model, main_args.lr, main_args.weight_decay)
        
        # Epoch iterator
        epoch_iter = tqdm(range(main_args.max_epoch))

        # Number of training samples
        n_train = metadata['n_train']

        # For every epoch
        for epoch in epoch_iter:

            # This will maintain the loss value
            epoch_loss = 0.0

            # For each training sample
            for i in range(n_train):
                
                # Load graph (training sample)
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)

                # Set the model in training mode
                model.train()

                # Make a forward pass - get the graph error
                loss = model(g)

                # Take the mean value
                loss /= n_train

                # Reset optimizer
                optimizer.zero_grad()

                # Accummulate epoch loss
                epoch_loss += loss.item()

                # Make a backward pass
                loss.backward()

                # Commit the gradients
                optimizer.step()
                del g

            # Report training epoch status
            epoch_iter.set_description(f"Epoch {epoch} | train_loss: {epoch_loss:.4f}")

        # Save trained model's parameters asa checkpoint
        torch.save(model.state_dict(), "./checkpoints/checkpoint-{}.pt".format(dataset_name))

        # Determine the file path for the dictionary of KNN distances
        save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset_name)

        # If the dictionary exists already, it shall be removed. Remove.
        if os.path.exists(save_dict_path):
            os.unlink(save_dict_path)
    return


if __name__ == '__main__':

    # Parse the arguments from config.py
    args = build_args()

    # Call main with args.
    main(args)
