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
        assert isinstance(x, Tensor), "The input must be a doja.Tensor"
        return self.forward(x)

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier initialization https://paperswithcode.com/method/xavier-initialization
        limit = np.sqrt(6/(in_features + out_features))
        self.weight = Parameter(Tensor(np.random.uniform(-limit, limit, size=(out_features, in_features))))
        self.bias = Parameter(Tensor(np.zeros(out_features)))
    
    def forward(self, x):
        return x @ self.weight.tensor.T + self.bias.tensor