import torch
import torch.nn as nn


def make_layer(block, num_layers, **kwargs):
    layers = []
    for _ in range(num_layers):
        layers.append(block(**kwargs))
    return nn.Sequential(*layers)