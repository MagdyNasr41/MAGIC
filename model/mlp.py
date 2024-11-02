import torch.nn as nn
import torch.nn.functional as F

# Very simple Multi-Layer Perceptron for Sample-Based Structure Reconstruction "edges between nodes"
class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
