import numpy as np

from .tensor import *

class Parameter:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Parameter({self.data})"

class Module:
    def __init__(self):
        self.parameters = []

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.parameters.append(value)
        super().__setattr__(name, value)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(Tensor(np.random.randn(out_features, in_features)))
        self.bias = Parameter(Tensor(np.random.randn(out_features)))
    
    def __call__(self, x):
        return self.weight.data @ x + self.bias.data