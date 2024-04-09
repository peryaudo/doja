from .tensor import *

def relu(input):
    out = Tensor(np.where(input.data > 0, input.data, 0))
    out.children = [input]
    def _grad_fn():
        input.grad += np.where(out.data > 0, out.grad, 0)
    out.grad_fn = _grad_fn
    return out

def exp(input):
    out = Tensor(np.exp(input.data))
    out.children = [input]
    def _grad_fn():
        input.grad += out.grad * np.exp(input.data)
    out.grad_fn = _grad_fn
    return out

def log(input):
    out = Tensor(np.log(input.data))
    out.children = [input]
    def _grad_fn():
        input.grad += out.grad / input.data
    out.grad_fn = _grad_fn
    return out

def sum(input, axis=None, keepdims=False):
    out_data = np.sum(input.data, axis=axis, keepdims=keepdims)
    out_shape = out_data.shape
    if not keepdims:
        out_data = np.squeeze(out_data, axis=axis)
    out = Tensor(out_data)
    out.children = [input]
    def _grad_fn():
        input.grad += out.grad.reshape(out_shape)
    out.grad_fn = _grad_fn
    return out

def max(input, axis=None, keepdims=False):
    assert axis == -1 and keepdims == True
    out = Tensor(np.max(input.data, axis=-1, keepdims=True))
    out.children = [input]
    max_indices = np.argmax(input.data, axis=-1, keepdims=False)
    max_onehot = np.identity(input.data.shape[-1])[max_indices]
    def _grad_fn():
        input.grad += out.grad * max_onehot
    out.grad_fn = _grad_fn
    return out

def softmax(logits):
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