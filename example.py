import doja
import numpy as np

class Model(doja.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = doja.Linear(28 * 28, 64)
        self.linear2 = doja.Linear(64, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

model = Model()

inputs = doja.Tensor(np.random.randn(16, 28 * 28))
labels = doja.Tensor(np.random.randn(16, 10))
logits = model(inputs)
loss = logits.cross_entropy(labels)
loss.backward()