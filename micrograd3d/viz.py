from .engine import Value
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from .optimizer import NewtonMethod, NaturalGradient, LBFGS

class Visualizer3D:
    """Enhanced 3D visualization for optimization processes"""
    
    def __init__(self, func, optimizer, n_steps=100, window_size=10):
        """
        Args:
            func (function): The objective function to optimize.
            optimizer (Optimizer): The optimizer to use for optimization.
            n_steps (int): The number of steps to run the optimization.
            window_size (int): The size of the visualization window.
        """
        self.func = func
        self.optimizer = optimizer
        self.n_steps = n_steps
        self.window_size = window_size  # Size of visualization window
        
        # Use built-in style
        plt.style.use('default')
        
        # Set color scheme
        self.colors = {
            'surface': plt.cm.viridis,
            'path': '#FF4B4B',     # Vibrant red
            'start': '#00FF00',    # Vibrant green
            'end': '#FF0000',      # Vibrant red
            'background': 'white'   # White background
        }
        
        # Set global style
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#333333',
            'axes.labelcolor': 'black',
            'axes.grid': True,
            'grid.color': '#CCCCCC',
            'grid.alpha': 0.3,
            'font.size': 10,
            'axes.labelsize': 12,
            'axes.titlesize': 14
        })
        
        # Automatically set objective function for optimizers that need it
        if isinstance(self.optimizer, (NewtonMethod, NaturalGradient, LBFGS)):
            self.optimizer._func = self.func
    
    def optimize_and_plot(self, start_point=(0, 0)):
        """Run optimization and display animation"""
        path = self._optimize(start_point)
        
        # Create figure and 3D axes
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate grid data
        x = np.linspace(min(start_point[0], path[-1][0]) - self.window_size, max(start_point[0], path[-1][0]) + self.window_size, 100)
        y = np.linspace(min(start_point[1], path[-1][1]) - self.window_size, max(start_point[1], path[-1][1]) + self.window_size, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self.func(Value(xi), Value(yi)).data for xi in x] for yi in y])
        
        # Get optimization path
        path = self._optimize(start_point)
        path_z = [self.func(Value(x), Value(y)).data for x, y in path]
        
        # Set surface plot style
        surf = ax.plot_surface(X, Y, Z, 
                             cmap=self.colors['surface'],
                             alpha=0.8,
                             linewidth=0,
                             antialiased=True)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Initialize path line and point
        line, = ax.plot([], [], [], 
                       color=self.colors['path'],
                       linewidth=2,
                       label='Optimization Path')
        point, = ax.plot([], [], [], 
                        'o',
                        color=self.colors['path'],
                        markersize=8)
        
        # Mark start and end points
        ax.scatter(*start_point, path_z[0],
                  color=self.colors['start'],
                  s=100,
                  label='Start')
        ax.scatter(*path[-1, :2], path_z[-1],
                  color=self.colors['end'],
                  s=100,
                  marker='*',
                  label='End')
        
        # Set plot style
        ax.set_xlabel('X', labelpad=10)
        ax.set_ylabel('Y', labelpad=10)
        ax.set_zlabel('Z', labelpad=10)
        ax.set_title('Optimization Process', pad=20)
        
        # Set grid
        ax.grid(True, alpha=0.3)
        
        # Set view angle
        ax.view_init(elev=30, azim=45)
        
        # Add legend
        ax.legend(loc='upper right')
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        
        def update(frame):
            # Calculate current frame path length
            idx = int((frame + 1) * len(path) / self.n_steps)
            
            # Update path
            line.set_data(path[:idx, 0], path[:idx, 1])
            line.set_3d_properties(path_z[:idx])
            
            # Update current point
            point.set_data([path[idx-1, 0]], [path[idx-1, 1]])
            point.set_3d_properties([path_z[idx-1]])
        
        # Create animation
        anim = FuncAnimation(
            fig, 
            update, 
            frames=self.n_steps,
            init_func=init,
            interval=100,  # Interval between frames (ms)
            blit=False,
            repeat=False  # Don't repeat animation
        )
        
        plt.show()
        return anim

    def _optimize(self, start_point):
        """Execute optimization"""
        path = [start_point]
        params = list(start_point)
        
        for _ in range(self.n_steps):
            # Calculate gradients
            x_val = Value(params[0])
            y_val = Value(params[1])
            loss = self.func(x_val, y_val)
            loss.backward()
            
            # Update parameters
            grads = [x_val.grad, y_val.grad]
            params = self.optimizer.step(params, grads)
            path.append(tuple(params))
        
        return np.array(path)
