import math
from dataclasses import dataclass
import numpy as np

@dataclass(eq=False)
class Value:
    """Scalar value supporting automatic differentiation with complete second-order derivatives"""
    data: float
    grad: float = 0.0
    hess: float = 0.0  # Second-order derivative
    cross_hess: dict = None  # Store cross derivatives with respect to other variables
    _prev: set = None
    _op: str = ''
    
    def __post_init__(self):
        self._prev = set()
        self._backward = lambda: None
        self._backward2 = lambda: None  # Second-order derivative
        self.cross_hess = {}  # Initialize cross derivatives dictionary
        self._id = id(self)  # Unique identifier for cross derivatives
    
    def __hash__(self):
        return self._id
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data)
        out._prev = {self, other}
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        def _backward2():
            self.hess += out.hess
            other.hess += out.hess
            # Cross derivatives propagate additively
            for var_id, cross_deriv in out.cross_hess.items():
                self.cross_hess[var_id] = self.cross_hess.get(var_id, 0) + cross_deriv
                other.cross_hess[var_id] = other.cross_hess.get(var_id, 0) + cross_deriv
        
        out._backward = _backward
        out._backward2 = _backward2
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data)
        out._prev = {self, other}
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        def _backward2():
            # Second derivatives using product rule
            self.hess += (other.data ** 2) * out.hess + other.data * out.grad
            other.hess += (self.data ** 2) * out.hess + self.data * out.grad
            
            # Cross derivatives
            cross_term = out.grad + self.data * other.data * out.hess
            self.cross_hess[other._id] = self.cross_hess.get(other._id, 0) + cross_term
            other.cross_hess[self._id] = other.cross_hess.get(self._id, 0) + cross_term
            
            # Propagate other cross derivatives
            for var_id, cross_deriv in out.cross_hess.items():
                if var_id not in {self._id, other._id}:
                    self.cross_hess[var_id] = self.cross_hess.get(var_id, 0) + other.data * cross_deriv
                    other.cross_hess[var_id] = other.cross_hess.get(var_id, 0) + self.data * cross_deriv
        
        out._backward = _backward
        out._backward2 = _backward2
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supports numeric powers"
        out = Value(self.data ** other)
        out._prev = {self}
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        
        def _backward2():
            # Second derivative of power function
            self.hess += (other * (other - 1) * (self.data ** (other - 2)) * out.grad +
                         (other * self.data ** (other - 1)) ** 2 * out.hess)
            
            # Cross derivatives propagate through chain rule
            for var_id, cross_deriv in out.cross_hess.items():
                power_term = other * (self.data ** (other - 1))
                self.cross_hess[var_id] = self.cross_hess.get(var_id, 0) + power_term * cross_deriv
        
        out._backward = _backward
        out._backward2 = _backward2
        return out
    
    def exp(self):
        """Exponential function and its second-order derivative"""
        x = self.data
        out = Value(math.exp(x))
        out._prev = {self}
        
        def _backward():
            self.grad += out.data * out.grad  # exp'(x) = exp(x)
            
        def _backward2():
            # exp''(x) = exp(x)
            self.hess += out.data * out.grad + out.data * out.hess
        
        out._backward = _backward
        out._backward2 = _backward2
        return out
    
    def log(self):
        """Logarithmic function and its second-order derivative"""
        assert self.data > 0, "Logarithmic function domain is positive numbers"
        out = Value(math.log(self.data))
        out._prev = {self}
        
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
            
        def _backward2():
            # log''(x) = -1/x^2
            self.hess += (-1.0 / (self.data ** 2)) * out.grad + (1.0 / self.data) ** 2 * out.hess
        
        out._backward = _backward
        out._backward2 = _backward2
        return out
    
    def sin(self):
        """Sine function and its second-order derivative"""
        out = Value(math.sin(self.data))
        out._prev = {self}
        
        def _backward():
            self.grad += math.cos(self.data) * out.grad
            
        def _backward2():
            # sin''(x) = -sin(x)
            self.hess += -math.sin(self.data) * out.grad + math.cos(self.data) ** 2 * out.hess
        
        out._backward = _backward
        out._backward2 = _backward2
        return out
    
    def cos(self):
        """Cosine function and its second-order derivative"""
        out = Value(math.cos(self.data))
        out._prev = {self}
        
        def _backward():
            self.grad += -math.sin(self.data) * out.grad
            
        def _backward2():
            # cos''(x) = -cos(x)
            self.hess += -math.cos(self.data) * out.grad + math.sin(self.data) ** 2 * out.hess
        
        out._backward = _backward
        out._backward2 = _backward2
        return out
    
    
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return Value(other) + (-self)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * (other**-1)
    def __rtruediv__(self, other): return Value(other) / self

    def backward(self, compute_hessian=False):
        """Execute backpropagation with complete second-order derivatives"""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1.0
        
        if compute_hessian:
            self.hess = 0.0
            # Initialize cross derivatives for the output
            self.cross_hess = {}
            
            # First backward pass for gradients
            for node in reversed(topo):
                node._backward()
            
            # Second backward pass for Hessian and cross derivatives
            for node in reversed(topo):
                node._backward2()
        else:
            # Standard backward pass for gradients only
            for node in reversed(topo):
                node._backward()
    
    def get_hessian(self, vars_list):
        """Get complete Hessian matrix with respect to given variables"""
        n = len(vars_list)
        H = np.zeros((n, n))
        
        # Fill diagonal elements
        for i, var in enumerate(vars_list):
            H[i, i] = var.hess
        
        # Fill cross derivatives
        for i, var1 in enumerate(vars_list):
            for j, var2 in enumerate(vars_list):
                if i != j:
                    H[i, j] = var1.cross_hess.get(var2._id, 0)
        
        return H