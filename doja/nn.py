import numpy as np

from .tensor import *

class Parameter:
    def __init__(self, tensor):
        self.tensor = tensor

    def __repr__(self):
        return f"Parameter({self.tensor})"

class Module:
    def __init__(self):
        self.parameters = []

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.parameters.append(value)
        if isinstance(value, Module):
            self.parameters.extend(value.parameters)
        super().__setattr__(name, value)

    def __call__(self, x):
        return self.forward(x)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(Tensor(np.random.randn(out_features, in_features)))
        self.bias = Parameter(Tensor(np.random.randn(out_features)))
    
    def forward(self, x):
        return x @ self.weight.tensor.T + self.bias.tensor