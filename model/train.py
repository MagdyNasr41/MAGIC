import dgl
import numpy as np
from tqdm import tqdm
from utils.loaddata import transform_graph

# Trains the model using batched graph data over a specified number of epochs.
"""
BIG QUESTION: WHAT DOES THE DATASSET LOOK LIKE
"""
def batch_level_train(model, graphs, train_loader, optimizer, max_epoch, device, n_dim=0, e_dim=0):

    # Get an epoch iterator for the maximum number of epochs
    epoch_iter = tqdm(range(max_epoch))

    # Loop over each epoch
    for epoch in epoch_iter:

        # Set the model to training mode
        model.train()
        loss_list = []

        # train_loader is the return value of extract_dataloaders function in train.py file 
        # which Extracts a DataLoader for training data by shuffling the entries "training dataset" and creating a sampler "batches of data samples". 
        for _, batch in enumerate(train_loader):
            # for each batch "set of data samples" transform_graph Transforms a graph by encoding node and edge types into one-hot feature vectors.
            batch_g = [transform_graph(graphs[idx][0], n_dim, e_dim).to(device) for idx in batch]

            # Combine the list of individual graphs into a single batched graph
            batch_g = dgl.batch(batch_g)

            # Sets the model to training mode
            model.train()

            # Perform a forward pass through the model with the batched graph
            loss = model(batch_g)

            # Zero out the gradients from the previous step
            optimizer.zero_grad()

            # Compute the gradients of the loss with respect to model parameters
            loss.backward()

            # Update the model parameters using the optimizer
            optimizer.step()

            # Append the current batch's loss to the loss list
            loss_list.append(loss.item())

            # Delete the batch_g variable to free up memory
            del batch_g

        # Update the progress bar with the current epoch number and average training loss
        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.4f}")
    return model
