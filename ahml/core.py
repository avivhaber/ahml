import math

class Scalar:
    @classmethod
    def wrap(cls, obj):
        return obj if isinstance(obj, Scalar) else Scalar(obj)

    def __init__(self, val, op=None, children=(), childgrads=()):
        self.val = val
        self.op = op
        self.children = children
        self._childgrads = childgrads
        self.grad = 0

    def zero_grad(self):
        self.grad = 0

    def _backward_from(self, acc):
        for child, cgrad in zip(self.children, self._childgrads):
            child.grad += acc * cgrad
            child._backward_from(acc * cgrad)

    # Compute derivative of this scalar wrt all dependent variables
    def backward(self):
        self.grad = 1
        self._backward_from(1)

    def __add__(self, other):
        other = Scalar.wrap(other)
        return Scalar(self.val + other.val, '+', (self,other), (1,1))
    
    def __mul__(self, other):
        other = Scalar.wrap(other)
        dleft = other.val
        dright = self.val
        return Scalar(self.val * other.val, '*', (self,other), (dleft,dright))
    
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        d = other * self.val**(other - 1)
        return Scalar(self.val ** other, '^', (self,), (d,))
    
    def relu(self):
        d = 0 if self.val <= 0 else 1
        return Scalar(max(0, self.val), 'r', (self,), (d,))
    
    def __repr__(self):
        return str(self.val)
    
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