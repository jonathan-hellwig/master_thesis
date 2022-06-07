from random import random
from jax import grad, jit, random
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax.scipy.linalg as linalg

def random_layer_parameters(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key,
                                 (n, m)), scale * random.normal(b_key, (n, 1))


def init_network_parameters(sizes, key):
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
def loss(parameters, x, y):
    error = batched_predict(parameters, x) - y
    return jnp.mean(jnp.square(error))


@jit
def sgd_update(parameters, x, y, step_size=0.1):
    gradients = grad(loss)(parameters, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(parameters, gradients)]


@jit
def svag_update(parameters,
                x_batch_first,
                y_batch_first,
                x_batch_second,
                y_batch_second,
                l,
                step_size=0.1):
    svag_loss = lambda parameters: (1 + jnp.sqrt(2.0 * l - 1)) / 2.0 * loss(
        parameters, x_batch_first,
        y_batch_first) + (1 - jnp.sqrt(2.0 * l - 1)) / 2.0 * loss(
            parameters, x_batch_second, y_batch_second)
    gradients = grad(svag_loss)(parameters)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(parameters, gradients)]


def sqrt_matrix(matrix):
    eigen_values, eigen_vectors = linalg.eigh(matrix)
    return eigen_vectors @ jnp.diag(jnp.sqrt(eigen_values)) @ eigen_vectors.T


def get_batches(x, y, batch_size, key):
    number_of_splits = jnp.ceil(len(x) / batch_size)
    return zip(jnp.array_split(random.shuffle(key, x), number_of_splits),
               jnp.array_split(random.shuffle(key, y), number_of_splits))


def main_svag():
    jax.config.update('jax_platform_name', 'cpu')
    number_of_points = 1024
    x = jnp.linspace(0.0, 2.0 * jnp.pi, number_of_points)
    x = x.reshape((number_of_points, 1))
    y = jnp.sin(x)
    batch_size = 32
    key = random.PRNGKey(0)
    sizes = [1, 2048, 1]
    parameters = init_network_parameters(sizes, key)
    y_hat = batched_predict(parameters, jnp.array([10.0]).reshape((1, 1)))
    step_size = 0.1
    l = 1
    loss_values = []
    for epoch in range(1000):
        batches = list(get_batches(x, y, batch_size, key))
        for (x_batch_first,
             y_batch_first), (x_batch_second,
                              y_batch_second) in zip(batches[::2],
                                                     batches[1::2]):
            parameters = svag_update(parameters, x_batch_first, y_batch_first,
                                     x_batch_second, y_batch_second, l)
            loss_value = loss(parameters, x_batch_first, y_batch_first)
            print(
                f'Epoch: {epoch}, loss: {loss_value}, step_size: {step_size}')
            loss_values.append(loss(parameters, x, y))

    y_hat = batched_predict(parameters, x)
    print(y_hat.shape)
    print(jnp.mean(jnp.square(y - y_hat)))
    plt.figure()
    plt.plot(loss_values)
    plt.figure()
    plt.plot(x, y_hat)
    plt.show()


def main():
    jax.config.update('jax_platform_name', 'cpu')
    number_of_points = 128
    x = jnp.linspace(0.0, 2.0 * jnp.pi, number_of_points)
    x = x.reshape((number_of_points, 1))
    y = jnp.sin(x)

    key = random.PRNGKey(1)
    sizes = [1, 64, 1]
    parameters = init_network_parameters(sizes, key)
    step_size = 0.1
    batch_size = 8
    loss_values = []
    for epoch in range(4000):
        for x_batch, y_batch in get_batches(x, y, batch_size, key):
            parameters = sgd_update(parameters, x_batch, y_batch)
            loss_value = loss(parameters, x_batch, y_batch)
            print(
                f'Epoch: {epoch}, loss: {loss_value}, step_size: {step_size}')
        loss_values.append(loss(parameters, x, y))

    y_hat = batched_predict(parameters, x)
    print(jnp.mean(jnp.square(y - y_hat)))
    plt.figure()
    plt.plot(loss_values)
    plt.figure()
    plt.plot(x, y_hat, x, y)

    plt.show()


if __name__ == "__main__":
    main()
