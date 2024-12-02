from .gat import GAT
from utils.utils import create_norm
from functools import partial
from itertools import chain
from .loss_func import sce_loss
import torch
import torch.nn as nn
import dgl
import random


def build_model(args):
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    negative_slope = args.negative_slope
    mask_rate = args.mask_rate
    alpha_l = args.alpha_l
    n_dim = args.n_dim
    e_dim = args.e_dim

    # Instantiate the GMAEModel with the specified parameters
    model = GMAEModel(
        n_dim=n_dim,              # Set the input node feature dimension
        e_dim=e_dim,              # Set the input edge feature dimension
        hidden_dim=num_hidden,    # Set the number of hidden units for each layer
        n_layers=num_layers,      # Set the number of layers in the model
        n_heads=4,                # Set the number of attention heads (fixed at 4)
        activation="prelu",       # Use PReLU (Parametric ReLU) as the activation function
        feat_drop=0.1,            # Set the dropout rate for features
        negative_slope=negative_slope,  # Set the negative slope for the LeakyReLU activation
        residual=True,            # Enable residual connections in the model
        mask_rate=mask_rate,      # Set the masking rate for training
        norm='BatchNorm',         # Use Batch Normalization for normalization
        loss_fn='sce',            # Set the loss function type (e.g., Soft Cross Entropy)
        alpha_l=alpha_l           # Set the alpha parameter for the loss function
    )
    return model

# a new class of Graph Masked Auto Encoder that inherits from PyTorch's nn.Module
class GMAEModel(nn.Module):
    def __init__(self, n_dim, e_dim, hidden_dim, n_layers, n_heads, activation,
                 feat_drop, negative_slope, residual, norm, mask_rate=0.5, loss_fn="sce", alpha_l=2):
        
        super(GMAEModel, self).__init__() # Initializes the parent class (nn.Module)
        self._mask_rate = mask_rate # Sets the rate of masking nodes during training
        self._output_hidden_size = hidden_dim # Sets the size of hidden layers output
        self.recon_loss = nn.BCELoss(reduction='mean') # Binary cross-entropy loss for edge reconstruction

        # Helper function for initializing weights in the model
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Xavier uniform initialization is a technique used to initialize the weights of neural networks in a way that helps improve 
                # convergence during training. The idea is to keep the scale of the gradients roughly the same across all layers of the network.
                # it helps to prevent the vanishing or exploding gradient problem when gradients are too small or too largeand. Also, it helps to stabilize learning.
                nn.init.xavier_uniform(m.weight) # Applies Xavier uniform initialization to weights
                nn.init.constant_(m.bias, 0)

        # Defines the fully connected layers for edge reconstruction
        self.edge_recon_fc = nn.Sequential(
            nn.Linear(hidden_dim * n_layers * 2, hidden_dim),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.edge_recon_fc.apply(init_weights)

        # Ensures that hidden_dim is divisible by n_heads "it is fixed and = 4 btw" (necessary for attention mechanisms) 
        assert hidden_dim % n_heads == 0
        enc_num_hidden = hidden_dim // n_heads
        enc_nhead = n_heads

        # Sets up the input and hidden dimensions for the decoder
        dec_in_dim = hidden_dim 
        dec_num_hidden = hidden_dim

        # build encoder
        # Constructs the encoder (Graph Attention Network model) 
        self.encoder = GAT(
            n_dim=n_dim,  # Input node feature dimension
            e_dim=e_dim,  # Input edge feature dimension
            hidden_dim=enc_num_hidden,  # Hidden dimension per head
            out_dim=enc_num_hidden,  # Output dimension per head
            n_layers=n_layers,  # Number of GAT layers
            n_heads=enc_nhead,  # Number of attention heads per layer
            n_heads_out=enc_nhead,  # Number of output attention heads
            concat_out=True,  # Concatenate output from all heads
            activation=activation,  # Activation function
            feat_drop=feat_drop,  # Feature dropout rate
            attn_drop=0.0,  # Attention dropout rate
            negative_slope=negative_slope,  # Negative slope for LeakyReLU
            residual=residual,  # Whether to include residual connections
            norm=create_norm(norm),  # Normalization type
            encoding=True,  # Indicates that this GAT model acts as an encoder
        )

        # build decoder for attribute prediction "reconstruction"
        self.decoder = GAT(
            n_dim=dec_in_dim,  # Input feature dimension for decoder
            e_dim=e_dim,  # Input edge feature dimension for decoder
            hidden_dim=dec_num_hidden,  # Hidden dimension for decoder
            out_dim=n_dim,  # Output dimension (same as input node feature)
            n_layers=1,  # Single GAT layer for decoder
            n_heads=n_heads,  # Number of attention heads
            n_heads_out=1,  # Single output head
            concat_out=True,  # Concatenate output from all heads
            activation=activation,  # Activation function
            feat_drop=feat_drop,  # Feature dropout rate
            attn_drop=0.0,  # Attention dropout rate
            negative_slope=negative_slope,  # Negative slope for LeakyReLU
            residual=residual,  # Whether to include residual connections
            norm=create_norm(norm),  # Normalization type
            encoding=False,  # Indicates that this GAT model acts as a decoder
        )

        # Initializes a parameter for masked node attributes
        self.enc_mask_token = nn.Parameter(torch.zeros(1, n_dim))

        # Linear transformation for mapping encoder output to decoder input
        self.encoder_to_decoder = nn.Linear(dec_in_dim * n_layers, dec_in_dim, bias=False)

        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    # Configures the loss function
    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError # Raises an error if an unsupported loss function is specified
        return criterion

    def encoding_mask_noise(self, g, mask_rate=0.3):
        # Applies random masking to node attributes in the input graph
        new_g = g.clone() # Clones the input graph
        num_nodes = g.num_nodes() # Gets the number of nodes in the graph
        perm = torch.randperm(num_nodes, device=g.device) # Generates a random permutation of node indices

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes) 
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        # Sets the attributes of masked nodes to the mask token
        new_g.ndata["attr"][mask_nodes] = self.enc_mask_token

        # Returns the masked graph and node indices
        return new_g, (mask_nodes, keep_nodes)

    def forward(self, g):
        loss = self.compute_loss(g)
        return loss

    def compute_loss(self, g):
        # Feature Reconstruction

        # Applies noise masking to the input graph and gets the mask and keep indices
        pre_use_g, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, self._mask_rate)
        pre_use_x = pre_use_g.ndata['attr'].to(pre_use_g.device)
        use_g = pre_use_g

        # Runs the encoder and gets representations
        enc_rep, all_hidden = self.encoder(use_g, pre_use_x, return_hidden=True)
        enc_rep = torch.cat(all_hidden, dim=1)
        rep = self.encoder_to_decoder(enc_rep)

        # Runs the decoder for reconstruction
        recon = self.decoder(pre_use_g, rep)
        x_init = g.ndata['attr'][mask_nodes]
        x_rec = recon[mask_nodes]
        loss = self.criterion(x_rec, x_init)

        # Structural Reconstruction
        threshold = min(10000, g.num_nodes()) # Sets a threshold for sampling

        # Generates negative samples for edge prediction
        negative_edge_pairs = dgl.sampling.global_uniform_negative_sampling(g, threshold)
        positive_edge_pairs = random.sample(range(g.number_of_edges()), threshold)
        positive_edge_pairs = (g.edges()[0][positive_edge_pairs], g.edges()[1][positive_edge_pairs])

        # Gets the source and destination nodes for edge reconstruction
        sample_src = enc_rep[torch.cat([positive_edge_pairs[0], negative_edge_pairs[0]])].to(g.device)
        sample_dst = enc_rep[torch.cat([positive_edge_pairs[1], negative_edge_pairs[1]])].to(g.device)

        # Predicts edges using the fully connected layer
        y_pred = self.edge_recon_fc(torch.cat([sample_src, sample_dst], dim=-1)).squeeze(-1)

        # Creates the true labels (1 for positive, 0 for negative)
        y = torch.cat([torch.ones(len(positive_edge_pairs[0])), torch.zeros(len(negative_edge_pairs[0]))]).to(
            g.device)
        
        # Adds the structural reconstruction loss
        loss += self.recon_loss(y_pred, y)
        return loss

    def embed(self, g):
        # Generates node embeddings using the encoder
        x = g.ndata['attr'].to(g.device)
        rep = self.encoder(g, x)
        return rep

    @property
    def enc_params(self):
        # Get encoder parameters
        return self.encoder.parameters()

    @property
    def dec_params(self):
        # Get decoder parameters
        return chain(*[self.encoder_to_decoder.parameters(), self.decoder.parameters()])
