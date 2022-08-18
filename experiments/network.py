from jax import grad, jit, random
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.scipy.linalg as linalg


def random_layer_parameters(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key,
                                 (n, m)), scale * random.normal(b_key, (n, 1))


def initialize_network_parameters(sizes, key):
    keys = random.split(key, len(sizes))
    return [
        random_layer_parameters(m, n, k)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def predict(parameters, x):
    activation = x
    for w, b in parameters[:-1]:
        outputs = jnp.dot(w, activation) + b
        activation = jnp.tanh(outputs)
    final_w, final_b = parameters[-1]
    logits = jnp.dot(final_w, activation) + final_b
    return logits


@jit
def batched_predict(parameters, x):
    activation = x.T
    for w, b in parameters[:-1]:
        outputs = jnp.dot(w, activation) + b
        activation = jnp.tanh(outputs)
    final_w, final_b = parameters[-1]
    output = jnp.dot(final_w, activation) + final_b
    return output.T


@jit
def root_mean_square_loss(parameters, x, y):
    error = batched_predict(parameters, x) - y
    return jnp.mean(jnp.square(error))
