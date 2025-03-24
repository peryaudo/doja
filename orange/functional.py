import numpy as np

from .tensor import *
from . import cuda_ops

def relu(input):
    if input.device == 'cuda':
        out_data = cuda_ops.relu(input.data)
    else:
        out_data = np.where(input.data > 0, input.data, 0)
        
    out = Tensor(out_data, device=input.device)
    out.children = [input]
    def _grad_fn():
        if input.device == 'cuda':
            # For ReLU, gradient is 1 where input > 0, 0 otherwise
            input.grad = cuda_ops.add(input.grad, cuda_ops.mul(out.grad, cuda_ops.relu(input.data)))
        else:
            input.grad += np.where(out.data > 0, out.grad, 0)
    out.grad_fn = _grad_fn
    return out

def exp(input):
    if input.device == 'cuda':
        out_data = cuda_ops.exp(input.data)
    else:
        out_data = np.exp(input.data)
        
    out = Tensor(out_data, device=input.device)
    out.children = [input]
    def _grad_fn():
        if input.device == 'cuda':
            # For exp, gradient is exp(x) * grad
            input.grad = cuda_ops.add(input.grad, cuda_ops.mul(out.grad, out_data))
        else:
            input.grad += out.grad * np.exp(input.data)
    out.grad_fn = _grad_fn
    return out

def log(input):
    if input.device == 'cuda':
        out_data = cuda_ops.log(input.data)
    else:
        out_data = np.log(input.data)
        
    out = Tensor(out_data, device=input.device)
    out.children = [input]
    def _grad_fn():
        if input.device == 'cuda':
            # For log, gradient is grad / x
            input.grad = cuda_ops.add(input.grad, cuda_ops.mul(out.grad, cuda_ops.div(1.0, input.data)))
        else:
            input.grad += out.grad / input.data
    out.grad_fn = _grad_fn
    return out

def sum(input, axis=None, keepdims=False):
    if input.device == 'cuda':
        out_data = cuda_ops.sum(input.data, axis, keepdims)
    else:
        out_data = np.sum(input.data, axis=axis, keepdims=keepdims)
        
    out_shape = out_data.shape
    if not keepdims:
        out_data = np.squeeze(out_data, axis=axis)
    out = Tensor(out_data, device=input.device)
    out.children = [input]
    def _grad_fn():
        if input.device == 'cuda':
            # For sum, gradient is broadcasted back to input shape
            input.grad = cuda_ops.add(input.grad, cuda_ops.broadcast(out.grad, input.shape))
        else:
            input.grad += out.grad.reshape(out_shape)
    out.grad_fn = _grad_fn
    return out

def max(input, axis=None, keepdims=False):
    assert axis == -1 and keepdims == True
    if input.device == 'cuda':
        out_data = cuda_ops.max(input.data, axis=-1, keepdims=True)
        max_indices = cuda_ops.argmax(input.data, axis=-1)
        max_onehot = cuda_ops.onehot(max_indices, input.data.shape[-1])
    else:
        out_data = np.max(input.data, axis=-1, keepdims=True)
        max_indices = np.argmax(input.data, axis=-1, keepdims=False)
        max_onehot = np.identity(input.data.shape[-1])[max_indices]
        
    out = Tensor(out_data, device=input.device)
    out.children = [input]
    def _grad_fn():
        if input.device == 'cuda':
            input.grad = cuda_ops.add(input.grad, cuda_ops.mul(out.grad, max_onehot))
        else:
            input.grad += out.grad * max_onehot
    out.grad_fn = _grad_fn
    return out

def softmax(logits):
    if logits.device == 'cuda':
        out_data = cuda_ops.softmax(logits.data)
    else:
        logits_max = max(logits, axis=-1, keepdims=True)
        e_logits = exp(logits - logits_max)
        out_data = e_logits.data / sum(e_logits, axis=-1, keepdims=True).data
        
    out = Tensor(out_data, device=logits.device)
    out.children = [logits]
    def _grad_fn():
        if logits.device == 'cuda':
            # Softmax gradient: out * (grad - sum(grad * out, axis=-1, keepdims=True))
            sum_term = cuda_ops.sum(cuda_ops.mul(out.grad, out_data), axis=-1, keepdims=True)
            logits.grad = cuda_ops.add(logits.grad, cuda_ops.mul(out_data, cuda_ops.sub(out.grad, sum_term)))
        else:
            logits.grad += out.data * (out.grad - sum(out.grad * out.data, axis=-1, keepdims=True).data)
    out.grad_fn = _grad_fn
    return out

def cross_entropy(logits, labels):
    # TODO: This should be calculated from log softmax instead of softmax
    # https://youtu.be/vGdB4eI4KBs?si=GU-3emSF0S7601Ox&t=5679
    # https://github.com/fastai/course22p2/blob/master/nbs/04_minibatch_training.ipynb
    # Add 1e-9 to avoid log(0)
    probs = softmax(logits) + 1e-9
    return -sum(labels * log(probs)) / logits.shape[0]