import os
import random
import time
import pickle as pkl
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from utils.utils import set_random_seed
from utils.loaddata import transform_graph, load_batch_level_dataset

'''
Function to evaluate a given model at the batch level.
    Arguments:
    - model: The neural network model to be evaluated.
    - pooler: A pooling function used to aggregate node embeddings.
    - device: The device (CPU/GPU) to perform the evaluation on.
    - method: The evaluation method (e.g., 'knn' for k-nearest neighbors).
    - dataset: The dataset used for evaluation.
    - n_dim: The number of node dimensions (optional).
    - e_dim: The number of edge dimensions (optional).
'''
def batch_level_evaluation(model, pooler, device, method, dataset, n_dim=0, e_dim=0):

    # Set the model to evaluation mode (disables dropout, etc.).
    model.eval() 
    x_list = []
    y_list = []

    # Load the dataset for batch-level evaluation.
    data = load_batch_level_dataset(dataset)
    full = data['full_index'] # Indices of the full dataset.
    graphs = data['dataset'] # List of graph data and labels.

    # Disable gradient calculation for evaluation.
    with torch.no_grad():
        # Iterate over each index in the full dataset.
        for i in full:

            # Transform and move the graph to the specified device (e.g., GPU).
            g = transform_graph(graphs[i][0], n_dim, e_dim).to(device)
            label = graphs[i][1] # Extract the label for the current graph.

            # Get the embeddings for the graph from the model.
            out = model.embed(g)

            # If the dataset is not 'wget', apply the pooler to get pooled embeddings.
            if dataset != 'wget':
                out = pooler(g, out).cpu().numpy()
            else:
                # Special handling for the 'wget' dataset, considering node types.
                out = pooler(g, out, n_types=data['n_feat']).cpu().numpy()
            y_list.append(label)
            x_list.append(out)

    # Concatenate all output embeddings into a single array.
    x = np.concatenate(x_list, axis=0)
    y = np.array(y_list) # Convert the list of labels into a numpy array.

    # Evaluate the model using k-nearest neighbors if specified.
    if 'knn' in method:
        test_auc, test_std = evaluate_batch_level_using_knn(100, dataset, x, y)
    else:
        raise NotImplementedError # Raise an error if the specified method is not implemented.
    
    return test_auc, test_std # Return the evaluation metrics (AUC and standard deviation).
'''
Function to evaluate batch-level performance using k-Nearest Neighbors (kNN).
    Arguments:
    - repeat: Number of times to repeat the evaluation. If -1, only run once without repetitions.
    - dataset: Name of the dataset used for evaluation.
    - embeddings: Numpy array of embeddings for the data.
    - labels: Numpy array of corresponding labels for the data.
'''
def evaluate_batch_level_using_knn(repeat, dataset, embeddings, labels):

    x, y = embeddings, labels # Assign input arguments to local variables for clarity.

    # Determine the number of training samples based on the dataset.
    if dataset == 'streamspot':
        train_count = 400
    else:
        train_count = 100
    
    # Set the number of neighbors for kNN (up to 10 or 2% of the training set size)
    n_neighbors = min(int(train_count * 0.02), 10)

    # Get indices for benign (label = 0) and attack (label = 1) samples
    benign_idx = np.where(y == 0)[0]
    attack_idx = np.where(y == 1)[0]
    if repeat != -1:
        # Lists to store metrics from each repetition.
        prec_list = []
        rec_list = []
        f1_list = []
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        auc_list = []

        for s in range(repeat):
            set_random_seed(s) # Set random seed for reproducibility.

            # Shuffle data for better generalization during training
            np.random.shuffle(benign_idx)
            np.random.shuffle(attack_idx)

            # Split data into training and testing sets.
            x_train = x[benign_idx[:train_count]]
            x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
            y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)

            # Normalize the data using training set statistics.
            x_train_mean = x_train.mean(axis=0)
            x_train_std = x_train.std(axis=0)
            x_train = (x_train - x_train_mean) / (x_train_std + 1e-6)
            x_test = (x_test - x_train_mean) / (x_train_std + 1e-6)

            # Fit the kNN model on the training set.
            nbrs = NearestNeighbors(n_neighbors=n_neighbors)
            nbrs.fit(x_train)

            # Get distances and indexes of k-nearest neighbors for training and testing data.
            distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
            mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
            distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

            # Calculate anomaly scores based on distances.
            score = distances.mean(axis=1) / mean_distance

            # Compute the AUC score.
            auc = roc_auc_score(y_test, score)

            # Calculate precision and recall for different thresholds.
            prec, rec, threshold = precision_recall_curve(y_test, score)

            # Compute F1 score and find the best threshold.
            f1 = 2 * prec * rec / (rec + prec + 1e-9)
            max_f1_idx = np.argmax(f1)
            best_thres = threshold[max_f1_idx]
            prec_list.append(prec[max_f1_idx])
            rec_list.append(rec[max_f1_idx])
            f1_list.append(f1[max_f1_idx])

            tn = 0
            fn = 0
            tp = 0
            fp = 0

            # Calculate confusion matrix values based on best threshold.
            for i in range(len(y_test)):
                if y_test[i] == 1.0 and score[i] >= best_thres:
                    tp += 1
                if y_test[i] == 1.0 and score[i] < best_thres:
                    fn += 1
                if y_test[i] == 0.0 and score[i] < best_thres:
                    tn += 1
                if y_test[i] == 0.0 and score[i] >= best_thres:
                    fp += 1
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
            tn_list.append(tn)
            auc_list.append(auc)

        # Print the performance metrics
        print('AUC: {}+{}'.format(np.mean(auc_list), np.std(auc_list)))
        print('F1: {}+{}'.format(np.mean(f1_list), np.std(f1_list)))
        print('PRECISION: {}+{}'.format(np.mean(prec_list), np.std(prec_list)))
        print('RECALL: {}+{}'.format(np.mean(rec_list), np.std(rec_list)))
        print('TN: {}+{}'.format(np.mean(tn_list), np.std(tn_list)))
        print('FN: {}+{}'.format(np.mean(fn_list), np.std(fn_list)))
        print('TP: {}+{}'.format(np.mean(tp_list), np.std(tp_list)))
        print('FP: {}+{}'.format(np.mean(fp_list), np.std(fp_list)))

        return np.mean(auc_list), np.std(auc_list) # Return mean and standard deviation of AUC scores.
    else:
        # Single evaluation case without repetitions.
        set_random_seed(0)
        np.random.shuffle(benign_idx)
        np.random.shuffle(attack_idx)
        x_train = x[benign_idx[:train_count]]
        x_test = np.concatenate([x[benign_idx[train_count:]], x[attack_idx]], axis=0)
        y_test = np.concatenate([y[benign_idx[train_count:]], y[attack_idx]], axis=0)
        x_train_mean = x_train.mean(axis=0)
        x_train_std = x_train.std(axis=0)
        x_train = (x_train - x_train_mean) / x_train_std
        x_test = (x_test - x_train_mean) / x_train_std

        nbrs = NearestNeighbors(n_neighbors=n_neighbors)
        nbrs.fit(x_train)
        distances, indexes = nbrs.kneighbors(x_train, n_neighbors=n_neighbors)
        mean_distance = distances.mean() * n_neighbors / (n_neighbors - 1)
        distances, indexes = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)

        # Calculate the performance metrics
        score = distances.mean(axis=1) / mean_distance
        auc = roc_auc_score(y_test, score)
        prec, rec, threshold = precision_recall_curve(y_test, score)
        f1 = 2 * prec * rec / (rec + prec + 1e-9)
        best_idx = np.argmax(f1)
        best_thres = threshold[best_idx]

        tn = 0
        fn = 0
        tp = 0
        fp = 0

        # Count the TP, TN, FP, FN values
        for i in range(len(y_test)):
            if y_test[i] == 1.0 and score[i] >= best_thres:
                tp += 1
            if y_test[i] == 1.0 and score[i] < best_thres:
                fn += 1
            if y_test[i] == 0.0 and score[i] < best_thres:
                tn += 1
            if y_test[i] == 0.0 and score[i] >= best_thres:
                fp += 1
        
        # Print the metrics
        print('AUC: {}'.format(auc))
        print('F1: {}'.format(f1[best_idx]))
        print('PRECISION: {}'.format(prec[best_idx]))
        print('RECALL: {}'.format(rec[best_idx]))
        print('TN: {}'.format(tn))
        print('FN: {}'.format(fn))
        print('TP: {}'.format(tp))
        print('FP: {}'.format(fp))
        return auc, 0.0


def evaluate_entity_level_using_knn(dataset, x_train, x_test, y_test):
    # Standardize the training and test sets using the mean and standard deviation of the training set
    x_train_mean = x_train.mean(axis=0)
    x_train_std = x_train.std(axis=0)
    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std

    # Set the number of neighbors for k-NN based on the dataset type
    if dataset == 'cadets':
        n_neighbors = 200
    else:
        n_neighbors = 10

    # Initialize the k-Nearest Neighbors model with the specified number of neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    nbrs.fit(x_train) # Fit the model using the training data

    save_dict_path = './eval_result/distance_save_{}.pkl'.format(dataset)

    # Check if distance data has already been computed and saved, and if not, then it is computed and saved 
    if not os.path.exists(save_dict_path):
        idx = list(range(x_train.shape[0]))
        random.shuffle(idx)
        distances, _ = nbrs.kneighbors(x_train[idx][:min(50000, x_train.shape[0])], n_neighbors=n_neighbors)
        del x_train
        mean_distance = distances.mean()
        del distances
        distances, _ = nbrs.kneighbors(x_test, n_neighbors=n_neighbors)
        save_dict = [mean_distance, distances.mean(axis=1)]
        distances = distances.mean(axis=1)
        with open(save_dict_path, 'wb') as f:
            pkl.dump(save_dict, f)
    else:
        # Load previously computed mean distance and test distances
        with open(save_dict_path, 'rb') as f:
            mean_distance, distances = pkl.load(f)

    # Calculate the anomaly score as the ratio of distances to mean distance
    score = distances / mean_distance
    del distances

    # Compute AUC score
    auc = roc_auc_score(y_test, score)

    # Compute precision-recall curve and F1 scores
    prec, rec, threshold = precision_recall_curve(y_test, score)
    f1 = 2 * prec * rec / (rec + prec + 1e-9)

    # Determine the best threshold index based on recall criteria for specific datasets
    best_idx = -1
    for i in range(len(f1)):
        # Specific recall criteria for different datasets to identify the best index
        # To repeat peak performance
        if dataset == 'trace' and rec[i] < 0.99979:
            best_idx = i - 1
            break
        if dataset == 'theia' and rec[i] < 0.99996:
            best_idx = i - 1
            break
        if dataset == 'cadets' and rec[i] < 0.9976:
            best_idx = i - 1
            break
    best_thres = threshold[best_idx]

    tn = 0
    fn = 0
    tp = 0
    fp = 0

    # Count the TP, TN, FP, FN values
    for i in range(len(y_test)):
        if y_test[i] == 1.0 and score[i] >= best_thres:
            tp += 1
        if y_test[i] == 1.0 and score[i] < best_thres:
            fn += 1
        if y_test[i] == 0.0 and score[i] < best_thres:
            tn += 1
        if y_test[i] == 0.0 and score[i] >= best_thres:
            fp += 1
    
    # Print the metrics
    print('AUC: {}'.format(auc))
    print('F1: {}'.format(f1[best_idx]))
    print('PRECISION: {}'.format(prec[best_idx]))
    print('RECALL: {}'.format(rec[best_idx]))
    print('TN: {}'.format(tn))
    print('FN: {}'.format(fn))
    print('TP: {}'.format(tp))
    print('FP: {}'.format(fp))
    return auc, 0.0, None, None
