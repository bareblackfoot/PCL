import torch
import torch.nn as nn
import numpy as np
from random import sample


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, category_dim=31, r=16384, m=0.999, T=0.1, mlp=False):
        """
        dim: feature dimension (default: 128)
        r: queue size; number of negative samples/prototypes (default: 16384)
        m: momentum for updating key encoder (default: 0.999)
        T: softmax temperature
        mlp: whether to use mlp projection
        """
        super(MoCo, self).__init__()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder.fc.weight.shape[1]
            self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc)
        self.class_layer = nn.Sequential(nn.ReLU(), nn.Linear(dim, category_dim))

    def forward(self, image):
        # compute query features
        q = self.encoder(image)  # queries: NxC
        q = self.class_layer(q)
        return q
