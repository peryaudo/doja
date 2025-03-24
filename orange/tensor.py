import numpy as np
from .cuda_ops import add as cuda_add, mul as cuda_mul, matmul as cuda_matmul, relu as cuda_relu, create_cuda_tensor, CUDATensor, pow as cuda_pow

def _reduce_grad(grad, target_shape):
    # TODO: Add more comments here
    if grad.shape == target_shape:
        return grad
    
    assert len(grad.shape) >= len(target_shape)
    stretched_shape = (1,) * (len(grad.shape) - len(target_shape)) + target_shape

    reduced_axes = []
    for i in range(len(grad.shape)):
        if grad.shape[i] != stretched_shape[i]:
            assert stretched_shape[i] == 1
            reduced_axes.append(i)
    reduced_grad = np.sum(grad, axis=tuple(reduced_axes), keepdims=True)
    return reduced_grad.reshape(target_shape)

class Tensor:
    def __init__(self, data, device='cpu'):
        self.device = device
        if device == 'cuda':
            # Convert numpy array to CUDA tensor
            self.data = create_cuda_tensor(np.array(data).astype(np.float32))
            self.grad = create_cuda_tensor(np.zeros_like(data))
        else:
            self.data = np.array(data).astype(np.float32)
            self.grad = np.zeros_like(self.data)
        self.grad_fn = None
        self.children = []

    def __repr__(self):
        if self.device == 'cuda':
            # Convert back to CPU for display
            return f"Tensor({self.data.to_cpu()}, device='{self.device}')"
        return f"Tensor({self.data}, device='{self.device}')"

    @property
    def shape(self):
        if self.device == 'cuda':
            return self.data.shape()
        return self.data.shape
    
    def cpu(self):
        """Transfer the tensor to CPU memory."""
        if self.device == 'cuda':
            self.data = self.data.to_cpu()
            self.grad = self.grad.to_cpu()
            self.device = 'cpu'
        return self
    
    def cuda(self):
        """Transfer the tensor to CUDA memory."""
        if self.device == 'cpu':
            self.data = create_cuda_tensor(self.data)
            self.grad = create_cuda_tensor(self.grad)
            self.device = 'cuda'
        return self
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
        
        if self.device == 'cuda' and other.device == 'cuda':
            out_data = cuda_add(self.data, other.data)
        else:
            out_data = self.data + other.data
            
        out = Tensor(out_data, device=self.device)
        out.children = [self, other]
        def _grad_fn():
            if self.device == 'cuda':
                self.grad = cuda_add(self.grad, _reduce_grad(out.grad, self.shape))
                other.grad = cuda_add(other.grad, _reduce_grad(out.grad, other.shape))
            else:
                self.grad += _reduce_grad(out.grad, self.shape)
                other.grad += _reduce_grad(out.grad, other.shape)
        out.grad_fn = _grad_fn
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
            
        if self.device == 'cuda' and other.device == 'cuda':
            out_data = cuda_mul(self.data, other.data)
        else:
            out_data = self.data * other.data
            
        out = Tensor(out_data, device=self.device)
        out.children = [self, other]
        def _grad_fn():
            if self.device == 'cuda':
                self.grad = cuda_add(self.grad, cuda_mul(other.data, _reduce_grad(out.grad, self.shape)))
                other.grad = cuda_add(other.grad, cuda_mul(self.data, _reduce_grad(out.grad, other.shape)))
            else:
                self.grad += _reduce_grad(other.data * out.grad, self.shape)
                other.grad += _reduce_grad(self.data * out.grad, other.shape)
        out.grad_fn = _grad_fn
        return out

    def __matmul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, device=self.device)
            
        if self.device == 'cuda' and other.device == 'cuda':
            out_data = cuda_matmul(self.data, other.data)
        else:
            out_data = self.data @ other.data
            
        out = Tensor(out_data, device=self.device)
        out.children = [self, other]
        def _grad_fn():
            if self.device == 'cuda':
                self.grad = cuda_add(self.grad, cuda_matmul(out.grad, other.data.T))
                other.grad = cuda_add(other.grad, cuda_matmul(self.data.T, out.grad))
            else:
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
        out.grad_fn = _grad_fn
        return out

    def __pow__(self, other):
        if self.device == 'cuda':
            out_data = cuda_pow(self.data, other)
        else:
            out_data = self.data**other
            
        out = Tensor(out_data, device=self.device)
        out.children = [self]
        def _grad_fn():
            if self.device == 'cuda':
                self.grad = cuda_add(self.grad, cuda_mul(cuda_mul(other, cuda_pow(self.data, other - 1)), out.grad))
            else:
                self.grad += (other * self.data**(other - 1)) * out.grad
        out.grad_fn = _grad_fn
        return out

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    @property
    def T(self):
        out = Tensor(self.data.T, device=self.device)
        out.children = [self]
        def _grad_fn():
            if self.device == 'cuda':
                self.grad = cuda_add(self.grad, out.grad.T)
            else:
                self.grad += out.grad.T
        out.grad_fn = _grad_fn
        return out

    def _topological_sort(self):
        result = []
        visited = set()
        def visit(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    visit(child)
                result.append(node)
        visit(self)
        return reversed(result)

    def backward(self):
        assert len(self.shape) == 0, "self should be scalar"

        # Gradient accumulation is not supported.
        for node in self._topological_sort():
            if len(node.shape) == 0:
                if node.device == 'cuda':
                    node.grad = create_cuda_tensor(np.array(0.0, dtype=np.float32))
                else:
                    node.grad = np.array(0.0, dtype=np.float32)
            else:
                if node.device == 'cuda':
                    node.grad = create_cuda_tensor(np.zeros_like(node.data.to_cpu()))
                else:
                    node.grad[:] = 0.0

        if self.device == 'cuda':
            self.grad = create_cuda_tensor(np.ones_like(self.data.to_cpu()))
        else:
            self.grad = np.ones_like(self.data)
            
        for node in self._topological_sort():
            if node.grad_fn:
                node.grad_fn()

def tensor(data, device='cpu'):
    return Tensor(data, device=device)