from jax import jit, grad
from network import root_mean_square_loss

@jit
def gradient_descent_update(parameters, x, y, learning_rate):
    gradients = grad(root_mean_square_loss)(parameters, x, y)
    return [(w - learning_rate * dw, b - learning_rate * db)
            for (w, b), (dw, db) in zip(parameters, gradients)]
