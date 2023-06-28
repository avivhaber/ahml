import random
import math

from ahml.core import Scalar

class Module:
    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def params(self):
        return []

class Neuron(Module):
    def __init__(self, ins):
        self.W = [Scalar(random.uniform(-1, 1)) for _ in range(ins)]
        self.b = Scalar(0)

    def params(self):
        return self.W + [self.b]
    
    def __call__(self, X, activate=True):
        dot = sum(w*x for w,x in zip(self.W, X)) + self.b
        if activate:
            return dot.relu()
        return dot
    
class Layer(Module):
    def __init__(self, ins, size, activate=True):
        self.neurons = []
        self.activate = activate
        for _ in range(size):
            self.neurons.append(Neuron(ins))
    
    def params(self):
        return [p for n in self.neurons for p in n.params()]
    
    def __call__(self, X):
        out = []
        for n in self.neurons:
            out.append(n(X, self.activate))
        return out
    
class MLP(Module):
    def __init__(self, ins, outs, hidden=[], lr=0.01):
        self.ins = ins
        self.layers = []
        self.lr = lr
        sizes = [ins] + hidden + [outs]
        for i in range(len(hidden) + 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1]))
        self.layers[-1].activate = False
    
    def params(self):
        return [p for l in self.layers for p in l.params()]

    def __call__(self, X):
        for l in self.layers:
            X = l(X)
        return X
    
    def update(self):
        for p in self.params():
            p.val -= self.lr * p.grad

def mse_loss(out, target):
    loss = Scalar(0)
    for x,y in zip(out, target):
        loss += (x - y)**2
    return loss

def total_loss(outs, targets):
    loss = Scalar(0)
    for out,target in zip(outs, targets):
        loss += mse_loss(out, target)
    return loss / len(outs)