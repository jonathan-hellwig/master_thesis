from jax import grad, jit, random
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from network import root_mean_square_loss, initialize_network_parameters, batched_predict


def sgd_update(parameters, x, y, step_size=0.1):
    gradients = grad(root_mean_square_loss)(parameters, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(parameters, gradients)]


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


def get_batches(key, x, y, batch_size):
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
    parameters = initialize_network_parameters(sizes, key)
    y_hat = batched_predict(parameters, jnp.array([10.0]).reshape((1, 1)))
    step_size = 0.1
    l = 1
    loss_values = []
    for epoch in range(1000):
        batches = list(get_batches(key, x, y, batch_size))
        for (x_batch_first,
             y_batch_first), (x_batch_second,
                              y_batch_second) in zip(batches[::2],
                                                     batches[1::2]):
            parameters = svag_update(parameters, x_batch_first, y_batch_first,
                                     x_batch_second, y_batch_second, l)
            loss_value = root_mean_square_loss(parameters, x_batch_first,
                                               y_batch_first)
            print(
                f'Epoch: {epoch}, root_mean_square_loss: {loss_value}, step_size: {step_size}'
            )
            loss_values.append(root_mean_square_loss(parameters, x, y))

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
    parameters = initialize_network_parameters(sizes, key)
    step_size = 0.1
    batch_size = 8
    loss_values = []
    for epoch in range(4000):
        for x_batch, y_batch in get_batches(x, y, batch_size, key):
            parameters = sgd_update(parameters, x_batch, y_batch)
            loss_value = root_mean_square_loss(parameters, x_batch, y_batch)
            print(
                f'Epoch: {epoch}, root_mean_square_loss: {loss_value}, step_size: {step_size}'
            )
        loss_values.append(root_mean_square_loss(parameters, x, y))

    y_hat = batched_predict(parameters, x)
    print(jnp.mean(jnp.square(y - y_hat)))
    plt.figure()
    plt.plot(loss_values)
    plt.figure()
    plt.plot(x, y_hat, x, y)

    plt.show()


if __name__ == "__main__":
    main()
