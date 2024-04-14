# doja: Re-implementation of PyTorch-like Autograd with numpy!

Inspired by [micrograd](https://github.com/karpathy/micrograd) :)

Doja is a re-implementation of PyTorch-like Autograd. If you've ever worked with PyTorch, you will notice the similarity:

```py
import doja

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

model = Model()
optimizer = doja.SGD(model.parameters, lr=1e-3)

# Loads the input images and labels from the dataset.
# images = ...
# labels = ...

logits = model(images)
loss = doja.cross_entropy(logits, labels)
loss.backward()
optimizer.step()
```

Doja implements `doja.Tensor` which is the equivalent of `torch.Tensor`. `doja.Tensor` is backed by numpy arrays. Compared to micrograd which only supports scalar variables, it's capable of doing something a bit more serious such as MNIST classification.

The Doja library itself only depends on numpy and has no other dependencies. (It does not depend on PyTorch!) example.py also has dependency on HuggingFace Datasets for using the MNIST dataset.

## Features

* Basic Tensor arithmetic operations with `doja.Tensor`
* Backpropagation with `Tensor.backward()`
* Stochastic Gradient Descent with `doja.SGD` optimizer
* Various functions: `relu()`, `exp()`, `log()`, `sum()`, `max()`, `softmax()`, `cross_entropy()`

## Example

[example.py](example.py) is the full training code of three layer MLP that can classify MNIST with the accuracy of 97.8% on the validation set.

[example_inference.ipynb](example_inference.ipynb) loads the checkpoint and visualizes actual results from MNIST.

## TODO

Other things I'd like to try implementing:

- [ ] BatchNorm
- [ ] Weight decay
- [ ] Adam
- [ ] Conv2d (useful resources: [1](https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo), [2](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c))
- [ ] GPU support (can be probably easily added with [cupy](https://cupy.dev/))

## License

MIT