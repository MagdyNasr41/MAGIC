import torch
import torch.nn as nn
from functools import partial
import numpy as np
import random
import torch.optim as optim

"""
Importance of an Optimizer:
- The optimizer is essential for training neural networks, as it updates the model's parameters 
    (weights and biases) to minimize the loss function.
- It determines how the model learns from data by adjusting the learning rate, applying momentum, 
    or adapting step sizes for each parameter.
- Different optimizers can lead to different convergence speeds, stability, and generalization 
    in model performance, so choosing the right optimizer is crucial for effective training.
"""
def create_optimizer(opt, model, lr, weight_decay):
    """
    This function selects an optimizer for training a model based on the specified type.
    - Adam: Combines momentum and adaptive learning rate per parameter, suitable for sparse data.
        Update rule: θ_{t+1} = θ_t - (α / (√v̂_t + ε)) * m̂_t
            where m̂_t, v̂_t are bias-corrected first and second moment estimates.
    - AdamW: Variant of Adam with decoupled weight decay for better generalization.
        Update rule is the same as Adam, but with weight decay applied separately.
    - Adadelta: Adapts learning rates based on model updates without requiring an initial learning rate.
        Update rule: θ_{t+1} = θ_t - (RMS(Δθ_{t-1}) / RMS(g_t)) * g_t
            where RMS(g_t) is a running average of recent squared gradients.
    - RAdam: Rectified Adam, which adds variance rectification to Adam for improved training stability.
        Update rule: similar to Adam with an additional rectification term to correct variance.
    - SGD with momentum: Standard optimizer for large datasets, SGD with momentum provides stable updates by adding inertia to parameter updates.
        Update rule: v_{t+1} = μ * v_t + g_t, θ_{t+1} = θ_t - α * v_{t+1}
            where μ is the momentum factor, and g_t is the gradient.
    """

    # Convert optimizer name to lowercase for case-insensitivity
    opt_lower = opt.lower()

    # Get model parameters and set basic optimizer arguments
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)
    optimizer = None
    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    # Choose the optimizer based on the specified type
    # Adam: Adaptive Moment Estimation; combines momentum and adaptive learning rate per parameter
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    # AdamW: Decouples weight decay from the adaptive learning rate in Adam for improved regularization
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    # Adadelta: Adaptive learning rate method that adjusts learning rates based on model updates
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    # RAdam: Rectified Adam; adds a variance rectification term to Adam to improve stability
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    # SGD with momentum: Stochastic Gradient Descent with momentum; moves along the gradient with added inertia
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    # Raise an error if an invalid optimizer type is specified
    else:
        assert False and "Invalid optimizer"
    return optimizer


def random_shuffle(x, y):
    # Shuffle x and y data 

    idx = list(range(len(x)))
    random.shuffle(idx)
    return x[idx], y[idx]


def set_random_seed(seed):
    # Apply random seed to all applicable modules

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def create_activation(name):
    # Create activation functions, name specified by the inputs

    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    # Create normalization layers, name specified by the input

    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return None


class NormLayer(nn.Module):
    # Normalization layer class, extended from torch nn.Module 

    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        
        # Initialize as appropriate
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)

        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)

        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        # Forward pass for the normalizer

        tensor = x

        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)

        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)

        return self.weight * sub / std + self.bias
