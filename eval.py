import torch
import warnings
from utils.loaddata import load_batch_level_dataset, load_entity_level_dataset, load_metadata
from model.autoencoder import build_model
from utils.poolers import Pooling
from utils.utils import set_random_seed
import numpy as np
from model.eval import batch_level_evaluation, evaluate_entity_level_using_knn
from utils.config import build_args
warnings.filterwarnings('ignore')


def main(main_args):

    # Set the device as CPU (or GPU if available)
    device = main_args.device if main_args.device >= 0 else "cpu"
    device = torch.device(device)

    # Dataset to be used
    dataset_name = main_args.dataset

    # Set the model topology depending on entity or batch level dataset we will use
    if dataset_name in ['streamspot', 'wget']:
        main_args.num_hidden = 256
        main_args.num_layers = 4
    else:
        main_args.num_hidden = 64
        main_args.num_layers = 3

    # Seed for replicability
    set_random_seed(0)

    # Measures for the batch level datasets
    if dataset_name == 'streamspot' or dataset_name == 'wget':

        # Load the data set
        dataset = load_batch_level_dataset(dataset_name)

        # Parse and assign the node and edge dimensions
        n_node_feat = dataset['n_feat']
        n_edge_feat = dataset['e_feat']
        main_args.n_dim = n_node_feat
        main_args.e_dim = n_edge_feat

        # Build the graph masked autoencoder model using the arguments
        model = build_model(main_args)

        # Load the previously trained model parameters from the checkpoint
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))

        # Move the model to the device (CPU or GPU)
        model = model.to(device)

        # Establish the pooling mode from the arguments
        pooler = Pooling(main_args.pooling)

        # Execute batch level evaluation
        test_auc, test_std = batch_level_evaluation(model, pooler, device, ['knn'], args.dataset, main_args.n_dim,
                                                    main_args.e_dim)
    
    # Measures for the entity level datasets (i.e. DARPA)
    else:
        
        # Get the dataset metadata (node and edge feature dimensions)
        metadata = load_metadata(dataset_name)
        main_args.n_dim = metadata['node_feature_dim']
        main_args.e_dim = metadata['edge_feature_dim']

        # Build the graph masked autoencoder model using the arguments
        model = build_model(main_args)

        # Load the previously trained model parameters from the checkpoint
        model.load_state_dict(torch.load("./checkpoints/checkpoint-{}.pt".format(dataset_name), map_location=device))

        # Move the model to the device (CPU or GPU)
        model = model.to(device)

        # Set the model in evaluation mode
        model.eval()

        # Get metadata for training, test sample counts; malicious data
        malicious, _ = metadata['malicious']
        n_train = metadata['n_train']
        n_test = metadata['n_test']

        # Don't calculate gradients (eval mode)
        with torch.no_grad():

            # Training sample list
            x_train = []

            # For every training sample
            for i in range(n_train):

                # Get the graph from the dataset and append the embeddings to the list
                g = load_entity_level_dataset(dataset_name, 'train', i).to(device)
                x_train.append(model.embed(g).cpu().numpy())
                del g
            
            # Concatenate samples
            x_train = np.concatenate(x_train, axis=0)

            # Benign node counter for skipping
            skip_benign = 0

            # Test sample list
            x_test = []

            # For every test sample
            for i in range(n_test):

                # Get the graph from the dataset ...
                g = load_entity_level_dataset(dataset_name, 'test', i).to(device)

                # Exclude training samples from the test set
                if i != n_test - 1:
                    skip_benign += g.number_of_nodes()

                # ... and append the embeddings to the list
                x_test.append(model.embed(g).cpu().numpy())
                del g
            
            # Concatenate samples
            x_test = np.concatenate(x_test, axis=0)

            # Setting up the binary test labels for (benign, malicious). 
            n = x_test.shape[0]
            y_test = np.zeros(n)
            y_test[malicious] = 1.0

            # Malicious dictionary
            malicious_dict = {}

            # Populate it
            for i, m in enumerate(malicious):
                malicious_dict[m] = i

            # Indices for test samples (to exclude training samples from the test set)
            test_idx = []

            # For every test sample
            for i in range(x_test.shape[0]):

                # Add the malicious and non-skipped benign nodes to the test set
                if i >= skip_benign or y_test[i] == 1.0:
                    test_idx.append(i)
            
            # Final test set to be used
            result_x_test = x_test[test_idx]
            result_y_test = y_test[test_idx]
            del x_test, y_test

            # Execute evaluation
            test_auc, test_std, _, _ = evaluate_entity_level_using_knn(dataset_name, x_train, result_x_test,
                                                                       result_y_test)
    
    print(f"#Test_AUC: {test_auc:.4f}±{test_std:.4f}")
    return


if __name__ == '__main__':

    # Parse arguments from config.py and call the main function
    args = build_args()
    main(args)
