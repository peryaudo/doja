import doja
import numpy as np
from datasets import load_dataset
from itertools import islice

class Model(doja.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = doja.Linear(28 * 28, 64)
        self.linear2 = doja.Linear(64, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        return x

model = Model()

optimizer = doja.SGD(model.parameters, lr=3e-4)

dataset = load_dataset("mnist").with_format('numpy')
train_dataset = dataset['train']

MEAN = 0.1307
STD = 0.3081
train_images = train_dataset['image'].astype(np.float32)
train_images = (train_images / 255.0 - MEAN) / STD
train_images = train_images.reshape(-1, 28 * 28)
train_labels = train_dataset['label']
train_labels =  np.identity(10, dtype=np.float32)[train_labels]

for i in range(0, len(train_dataset), 16):
    images = doja.Tensor(train_images[i:i+16])
    labels = doja.Tensor(train_labels[i:i+16])
    logits = model(images)
    loss = logits.cross_entropy(labels)
    loss.zero_grad()
    loss.backward()
    optimizer.step()