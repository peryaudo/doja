# 97.8% accuracy on MNIST's validation set
import doja
import numpy as np
from datasets import load_dataset
import argparse
import pickle

class Model(doja.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = doja.Linear(28 * 28, 64)
        self.linear2 = doja.Linear(64, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = doja.relu(x)
        x = self.linear2(x)
        return x

def format_dataset(dataset):
    MEAN = 0.1307
    STD = 0.3081
    images = dataset['image'].astype(np.float32)
    images = (images / 255.0 - MEAN) / STD
    images = images.reshape(-1, 28 * 28)
    labels = dataset['label']
    labels =  np.identity(10, dtype=np.float32)[labels]
    return images, labels

if __name__ == "__main__":
    dataset = load_dataset("mnist").with_format('numpy')
    train_images, train_labels = format_dataset(dataset['train'])
    val_images, val_labels = format_dataset(dataset['test'])

    BATCH_SIZE = 16
    NUM_EPOCH = 200

    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', action='store_true', help='Enable logging with wandb')
    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb.init(project="doja_mnist")

    model = Model()
    optimizer = doja.SGD(model.parameters, lr=1e-3)

    step = 0
    for epoch_idx in range(NUM_EPOCH):
        train_loss = 0
        num_batches = 0
        for i in range(0, train_images.shape[0], BATCH_SIZE):
            images = doja.tensor(train_images[i:i+BATCH_SIZE])
            labels = doja.tensor(train_labels[i:i+BATCH_SIZE])
            logits = model(images)
            loss = doja.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.data)
            num_batches += 1
            step += 1
        
        train_loss /= num_batches
        
        val_loss = 0
        num_batches = 0
        num_correct = 0

        for i in range(0, val_images.shape[0], BATCH_SIZE):
            images = doja.tensor(val_images[i:i+BATCH_SIZE])
            labels = doja.tensor(val_labels[i:i+BATCH_SIZE])
            logits = model(images)
            loss = doja.cross_entropy(logits, labels)
            num_correct += (
                (np.argmax(labels.data, axis=-1) == np.argmax(logits.data, axis=-1))
                .astype(np.float32).sum())
            val_loss += float(loss.data)
            num_batches += 1
        
        val_loss /= num_batches
        accuracy = num_correct / (num_batches * BATCH_SIZE)

        if args.wandb:
            wandb.log({"epoch": epoch_idx, "train_loss": train_loss, "val_loss": val_loss, "accuracy": accuracy}, step=step)
        print("epoch {}: val loss: {:.4f} accuracy: {:.2f}%".format(epoch_idx, val_loss, accuracy * 100.0))

    # Save the final checkpoint as pickle.
    with open("ckpt.pkl", "wb") as f:
        pickle.dump(model, f)

    if args.wandb:
        wandb.finish()