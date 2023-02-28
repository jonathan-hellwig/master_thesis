from jax import grad, jit
import jax.numpy as jnp
from helpers.network import root_mean_square_loss


@jit
def svag_update(parameters, x_batch_first, y_batch_first, x_batch_second,
                y_batch_second, l, learning_rate):
    svag_loss = lambda parameters: (
        1 + jnp.sqrt(2.0 * l - 1)) / 2.0 * root_mean_square_loss(
            parameters, x_batch_first, y_batch_first) + (1 - jnp.sqrt(
                2.0 * l - 1)) / 2.0 * root_mean_square_loss(
                    parameters, x_batch_second, y_batch_second)
    gradients = grad(svag_loss)(parameters)
    return [(w - learning_rate * dw, b - learning_rate * db)
            for (w, b), (dw, db) in zip(parameters, gradients)]
