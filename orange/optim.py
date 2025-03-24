class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.tensor.data -= self.lr * param.tensor.grad