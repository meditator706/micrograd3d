from abc import ABC, abstractmethod
import numpy as np
from .engine import Value


class Optimizer(ABC):
    """Base class for all optimizers"""
    def __init__(self, learning_rate=0.1, clip_value=1.0):
        self.learning_rate = learning_rate
        self.clip_value = clip_value
    
    def clip_gradients(self, grads):
        """Gradient clipping"""
        return [max(min(g, self.clip_value), -self.clip_value) for g in grads]
    
    @abstractmethod
    def step(self, params, grads):
        grads = self.clip_gradients(grads)
        pass

class GradientDescent(Optimizer):
    """Standard Gradient Descent optimizer"""
    def step(self, params, grads):
        return [p - self.learning_rate * g for p, g in zip(params, grads)]

class Momentum(Optimizer):
    """Momentum optimizer"""
    def __init__(self, learning_rate=0.1, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] + self.learning_rate * g
            new_params.append(p - self.velocity[i])
        return new_params

class AdaGrad(Optimizer):
    """AdaGrad optimizer"""
    def __init__(self, learning_rate=0.1, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.square_grads = None
    
    def step(self, params, grads):
        if self.square_grads is None:
            self.square_grads = [np.zeros_like(p) for p in params]
        
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.square_grads[i] += g**2
            new_params.append(p - self.learning_rate * g / (np.sqrt(self.square_grads[i]) + self.epsilon))
        return new_params

class RMSprop(Optimizer):
    """RMSprop optimizer"""
    def __init__(self, learning_rate=0.01, decay_rate=0.99, epsilon=1e-8):
        super().__init__(learning_rate)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.square_grads = None
    
    def step(self, params, grads):
        if self.square_grads is None:
            self.square_grads = [np.zeros_like(p) for p in params]
        
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.square_grads[i] = self.decay_rate * self.square_grads[i] + (1 - self.decay_rate) * g**2
            new_params.append(p - self.learning_rate * g / (np.sqrt(self.square_grads[i]) + self.epsilon))
        return new_params

class Adam(Optimizer):
    """Adam optimizer"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    
    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            new_params.append(p - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
        return new_params
    


class NewtonMethod(Optimizer):
    """Newton's Method optimizer using complete Hessian matrix"""
    def __init__(self, learning_rate=1.0, epsilon=1e-8, min_step=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self._func = None
        self.min_step = min_step  
    
    def compute_full_hessian(self, x_val, y_val):
        """Compute complete Hessian matrix using second-order backpropagation"""
        if self._func is None:
            raise ValueError("Objective function not set. This should be handled automatically by Visualizer3D.")
        
        # Compute all derivatives
        loss = self._func(x_val, y_val)
        loss.backward(compute_hessian=True)
        
        # Get complete Hessian matrix
        return loss.get_hessian([x_val, y_val])
    
    def step(self, params, grads):
        x_val = Value(params[0])
        y_val = Value(params[1])
        
       
        grad_norm = np.linalg.norm(grads)
        if grad_norm < self.min_step:
            return params 
        
        # Compute Hessian
        H = self.compute_full_hessian(x_val, y_val)
        
        try:
            U, s, Vh = np.linalg.svd(H)
            s[s < self.epsilon] = self.epsilon 
            H_inv = (U * (1/s)) @ Vh
            

            delta = H_inv @ np.array(grads)
            

            step_size = np.linalg.norm(delta)
            if step_size > 1.0:
                delta *= 1.0 / step_size
                

            new_params = [p - self.learning_rate * d for p, d in zip(params, delta)]
            return new_params
            
        except np.linalg.LinAlgError:
            return [p - self.learning_rate * g for p, g in zip(params, grads)]

class LBFGS(Optimizer):
    """Limited-memory BFGS optimizer with second-order derivative support"""
    def __init__(self, learning_rate=1.0, history_size=10, epsilon=1e-8):
        super().__init__(learning_rate)
        self.history_size = history_size
        self.epsilon = epsilon
        self.s_history = []  # Parameter difference history
        self.y_history = []  # Gradient difference history
        self.rho_history = []  # ρ value history
        self.prev_params = None
        self.prev_grads = None
        self._func = None  # Store objective function for Hessian computation
    
    def compute_hessian_vector_product(self, x_val, y_val, vector):
        """Compute Hessian-vector product directly using second-order backpropagation"""
        if self._func is None:
            raise ValueError("Objective function not set. This should be handled automatically by Visualizer3D.")
        
        # Create directional derivative function
        def directional_derivative(t):
            x_t = Value(x_val.data + t * vector[0])
            y_t = Value(y_val.data + t * vector[1])
            return self._func(x_t, y_t)
        
        # Compute first directional derivative
        t = Value(0.0)
        d1 = directional_derivative(t)
        d1.backward(compute_hessian=True)
        
        # The gradient of the directional derivative gives us Hv directly
        Hv_x = x_val.grad * vector[0] + y_val.grad * vector[1]
        Hv_y = x_val.hess * vector[0] + y_val.hess * vector[1]
        
        return np.array([Hv_x, Hv_y])
    
    def step(self, params, grads):
        x_val = Value(params[0])
        y_val = Value(params[1])
        params = np.array(params)
        grads = np.array(grads)
        
        if self.prev_params is None:
            self.prev_params = params.copy()
            self.prev_grads = grads.copy()
            return params - self.learning_rate * grads
        
        # Compute differences
        s = params - self.prev_params
        y = grads - self.prev_grads
        
        # Use Hessian information to improve y estimate
        Bs = self.compute_hessian_vector_product(x_val, y_val, s)
        y = 0.8 * y + 0.2 * Bs  # Blend gradient difference with Hessian estimate
        
        # Compute ρ with safeguard
        sy = np.dot(s, y)
        if abs(sy) < self.epsilon:
            sy = self.epsilon * (1.0 if sy >= 0 else -1.0)
        rho = 1.0 / sy
        
        # Update history
        self.s_history.append(s)
        self.y_history.append(y)
        self.rho_history.append(rho)
        
        # Keep history size within limit
        if len(self.s_history) > self.history_size:
            self.s_history.pop(0)
            self.y_history.pop(0)
            self.rho_history.pop(0)
        
        # Two-loop recursion method to compute search direction
        q = grads.copy()
        alpha_history = []
        
        # First loop
        for i in range(len(self.s_history)-1, -1, -1):
            alpha = self.rho_history[i] * np.dot(self.s_history[i], q)
            alpha_history.append(alpha)
            q = q - alpha * self.y_history[i]
        
        # Compute initial Hessian approximation using second-order info
        if len(self.s_history) > 0:
            # Use actual Hessian diagonal for scaling
            loss = self._func(x_val, y_val)
            loss.backward(compute_hessian=True)
            h_diag = np.array([x_val.hess, y_val.hess])
            H0 = np.mean(h_diag + self.epsilon)
        else:
            H0 = 1.0
        
        # Compute search direction
        z = H0 * q
        
        # Second loop
        for i, alpha in enumerate(reversed(alpha_history)):
            beta = self.rho_history[i] * np.dot(self.y_history[i], z)
            z = z + self.s_history[i] * (alpha - beta)
        
        # Line search with second-order information
        step_size = 1.0
        if len(self.s_history) > 0:
            # Use Hessian to estimate good step size
            Hz = self.compute_hessian_vector_product(x_val, y_val, z)
            zHz = np.dot(z, Hz)
            if abs(zHz) > self.epsilon:
                step_size = min(1.0, 2.0 * abs(np.dot(z, grads) / zHz))
        
        # Update history parameters and gradients
        self.prev_params = params.copy()
        self.prev_grads = grads.copy()
        
        # Return new parameters
        return list(params - step_size * self.learning_rate * z)

class NaturalGradient(Optimizer):
    """Natural Gradient Descent optimizer

    Uses Fisher Information Matrix (FIM) as Riemannian metric tensor.
    For general optimization problems, we approximate FIM using absolute values of Hessian matrix.
    """
    def __init__(self, learning_rate=0.1, epsilon=1e-8, damping=1e-4):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.damping = damping  # Damping factor to ensure FIM's positive definiteness
        self._func = None
    
    def compute_fim(self, x_val, y_val):
        """Compute Fisher Information Matrix
        
        For general optimization problems, we approximate FIM using absolute values of Hessian matrix.
        """
        if self._func is None:
            raise ValueError("Objective function not set. This should be handled automatically by Visualizer3D.")
        
        # Compute first derivatives
        loss = self._func(x_val, y_val)
        loss.backward()
        grad_x = x_val.grad
        grad_y = y_val.grad
        
        # Compute second derivatives
        loss.backward(compute_hessian=True)
        h_xx = abs(x_val.hess)  # Take absolute value to ensure positive definiteness
        h_yy = abs(y_val.hess)
        
        # Compute cross derivative
        h = 1e-5
        x_plus = Value(x_val.data + h)
        y_plus = Value(y_val.data + h)
        
        f_pp = self._func(x_plus, y_plus).data
        f_pm = self._func(x_plus, Value(y_val.data - h)).data
        f_mp = self._func(Value(x_val.data - h), y_plus).data
        f_mm = self._func(Value(x_val.data - h), Value(y_val.data - h)).data
        
        h_xy = abs((f_pp - f_pm - f_mp + f_mm) / (4 * h * h))
        
        # Build Fisher Information Matrix
        fim = np.array([[h_xx, h_xy],
                       [h_xy, h_yy]])
        
        # Add damping term to ensure positive definiteness
        fim += self.damping * np.eye(2)
        
        return fim
    
    def step(self, params, grads):
        x_val = Value(params[0])
        y_val = Value(params[1])
        
        # Compute Fisher Information Matrix
        fim = self.compute_fim(x_val, y_val)
        
        try:
            # Compute natural gradient direction: F^(-1)g
            natural_grad = np.linalg.solve(fim, np.array(grads))
            
            # Update parameters
            new_params = [p - self.learning_rate * ng 
                         for p, ng in zip(params, natural_grad)]
            return new_params
            
        except np.linalg.LinAlgError:
            # If FIM is not invertible, fall back to standard gradient descent
            return [p - self.learning_rate * g for p, g in zip(params, grads)]
    


