import numpy as np

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
    def __init__(self, data):
        self.data = np.array(data).astype(np.float32)
        self.grad = np.zeros_like(self.data)
        self.grad_fn = None
        self.children = []

    def __repr__(self):
        return f"Tensor({self.data})"

    @property
    def shape(self):
        return self.data.shape
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data)
        out.children = [self, other]
        def _grad_fn():
            self.grad += _reduce_grad(out.grad, self.shape)
            other.grad += _reduce_grad(out.grad, other.shape)
        out.grad_fn = _grad_fn
        return out
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data * other.data)
        out.children = [self, other]
        def _grad_fn():
            self.grad += _reduce_grad(other.data * out.grad, self.shape)
            other.grad += _reduce_grad(self.data * out.grad, other.shape)
        out.grad_fn = _grad_fn
        return out

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data)
        out.children = [self, other]
        def _grad_fn():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out.grad_fn = _grad_fn
        return out

    def __pow__(self, other):
        out = Tensor(self.data**other)
        out.children = [self]
        def _grad_fn():
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
        # TODO: Avoid grad reallocation
        out = Tensor(self.data.T)
        out.children = [self]
        def _grad_fn():
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
                node.grad = np.array(0.0, dtype=np.float32)
            else:
                node.grad[:] = 0.0

        self.grad = np.ones_like(self.data)
        for node in self._topological_sort():
            if node.grad_fn:
                node.grad_fn()