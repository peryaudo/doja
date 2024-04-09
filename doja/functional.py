from .tensor import *

def relu(self):
    out = Tensor(np.where(self.data > 0, self.data, 0))
    out.children = [self]
    def _grad_fn():
        self.grad += np.where(out.data > 0, out.grad, 0)
    out.grad_fn = _grad_fn
    return out

def exp(self):
    out = Tensor(np.exp(self.data))
    out.children = [self]
    def _grad_fn():
        self.grad += out.grad * np.exp(self.data)
    out.grad_fn = _grad_fn
    return out

def log(self):
    out = Tensor(np.log(self.data))
    out.children = [self]
    def _grad_fn():
        self.grad += out.grad / self.data
    out.grad_fn = _grad_fn
    return out

def sum(self, axis=None, keepdims=False):
    out_data = np.sum(self.data, axis=axis, keepdims=keepdims)
    out_shape = out_data.shape
    if not keepdims:
        out_data = np.squeeze(out_data, axis=axis)
    out = Tensor(out_data)
    out.children = [self]
    def _grad_fn():
        self.grad += out.grad.reshape(out_shape)
    out.grad_fn = _grad_fn
    return out

def max(self, axis=None, keepdims=False):
    assert axis == -1 and keepdims == True
    out = Tensor(np.max(self.data, axis=-1, keepdims=True))
    out.children = [self]
    max_indices = np.argmax(self.data, axis=-1, keepdims=False)
    max_onehot = np.identity(self.data.shape[-1])[max_indices]
    def _grad_fn():
        self.grad += out.grad * max_onehot
    out.grad_fn = _grad_fn
    return out

def softmax(self):
    logits = self
    logits_max = max(logits, axis=-1, keepdims=True)
    e_logits = exp(logits - logits_max)
    return e_logits / sum(e_logits, axis=-1, keepdims=True)

def cross_entropy(self, labels):
    # TODO: This should be calculated from log softmax instead of softmax
    # https://youtu.be/vGdB4eI4KBs?si=GU-3emSF0S7601Ox&t=5679
    # https://github.com/fastai/course22p2/blob/master/nbs/04_minibatch_training.ipynb
    # Add 1e-9 to avoid log(0)
    probs = softmax(self) + 1e-9
    return -sum(labels * log(probs)) / self.shape[0]