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

def format_dataset(dataset):
    MEAN = 0.1307
    STD = 0.3081
    images = dataset['image'].astype(np.float32)
    images = (images / 255.0 - MEAN) / STD
    images = images.reshape(-1, 28 * 28)
    labels = dataset['label']
    labels =  np.identity(10, dtype=np.float32)[labels]
    return images, labels

dataset = load_dataset("mnist").with_format('numpy')
train_images, train_labels = format_dataset(dataset['train'])
val_images, val_labels = format_dataset(dataset['test'])

for epoch_idx in range(100):
    for i in range(0, train_images.shape[0], 16):
        images = doja.Tensor(train_images[i:i+16])
        labels = doja.Tensor(train_labels[i:i+16])
        logits = model(images)
        loss = logits.cross_entropy(labels)
        loss.zero_grad()
        loss.backward()
        optimizer.step()
    
    val_loss = 0
    num_batches = 0
    num_correct = 0

    for i in range(0, val_images.shape[0], 16):
        images = doja.Tensor(val_images[i:i+16])
        labels = doja.Tensor(val_labels[i:i+16])
        logits = model(images)
        loss = logits.cross_entropy(labels)
        num_correct += (
            (np.argmax(labels.data, axis=-1) == np.argmax(logits.data, axis=-1))
            .astype(np.float32).sum())
        val_loss += float(loss.data)
        num_batches += 1
    
    val_loss /= num_batches
    accuracy = num_correct / (num_batches * 16) * 100.0
    print("epoch {}: val loss: {:.4f} accuracy: {:.2f}%".format(epoch_idx, val_loss, accuracy))