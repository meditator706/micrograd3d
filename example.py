from micrograd3d.optimizer import *
from micrograd3d.viz import Visualizer3D
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter



def square(x, y):
    """Square function - Simple quadratic test function"""
    return x**2 + y**2

def cosine(x, y):
    """Cosine function - Simple trigonometric test function"""
    return np.cos(x) + np.cos(y)

# Create optimizers with different configurations
optimizers = {
    'GD': GradientDescent(learning_rate=0.1),
    'Momentum': Momentum(learning_rate=0.0001, momentum=0.9),
    'AdaGrad': AdaGrad(learning_rate=0.01),
    'RMSprop': RMSprop(learning_rate=0.001),
    'Adam': Adam(learning_rate=0.1),
    'Newton': NewtonMethod(learning_rate=0.1),
    'LBFGS': LBFGS(learning_rate=0.1),
    'Natural': NaturalGradient(learning_rate=0.5, damping=1e-4),
}

# Create visualizer instance
viz = Visualizer3D(
    func=square,
    optimizer=optimizers['Adam'],
    n_steps=200,
    window_size=5
)

# Run optimization and create animation
anim = viz.optimize_and_plot(start_point=(-10.0, 10.0))

# Save animation as GIF
writer = PillowWriter(fps=30)
anim.save('optimization_square.gif', writer=writer)

plt.show()