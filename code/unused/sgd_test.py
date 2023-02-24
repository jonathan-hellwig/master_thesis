from functools import partial
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    NUMBER_OF_DATA_POINTS = 100
    NOISE_STANDARD_DEVIATION = 1
    ITERATIONS = 100
    STEP_SIZE = 0.5
    W = 3
    key = jax.random.PRNGKey(42)
    key, X_key, Y_key = jax.random.split(key, 3)
    X = jnp.linspace(-1, 1, NUMBER_OF_DATA_POINTS)
    print(X)
    Y = W * X + NOISE_STANDARD_DEVIATION * \
        jax.random.normal(Y_key, (NUMBER_OF_DATA_POINTS,))

    @partial(jax.vmap, in_axes=(0, None, None))
    def batched_loss(W, X, Y):
        return jnp.mean((W * X - Y)**2)

    def loss(W, X, Y):
        return jnp.mean((W * X - Y)**2)

    W_GD = jnp.float32(10)
    weights_gd = []
    for _ in range(ITERATIONS):
        gradient = jax.grad(loss)(W_GD, X, Y)
        W_GD = W_GD - STEP_SIZE * gradient
        weights_gd.append(W_GD)

    def split_batches(X, Y, batch_size):
        return jnp.split(X, NUMBER_OF_DATA_POINTS // batch_size), jnp.split(Y, NUMBER_OF_DATA_POINTS // batch_size)
    min_loss = loss(jnp.sum(X * Y)/jnp.sum(X**2), X, Y)
    print(min_loss)
    BATCH_SIZE = 1
    REPETITIONS = 10
    sgd_loss = [[] for _ in range(REPETITIONS)]
    for i in range(REPETITIONS):
        key, X_key, Y_key = jax.random.split(key, 3)
        X = jax.random.shuffle(X_key, X)
        Y = jax.random.shuffle(Y_key, Y)
        W_SGD = jnp.float32(10)
        weights_sgd = []
        X_batches, Y_batches = split_batches(X, Y, BATCH_SIZE)
        for x, y in zip(X_batches, Y_batches):
            value, gradient = jax.value_and_grad(loss)(W_SGD, x, y)
            sgd_loss[i].append(value)
            W_SGD = W_SGD - STEP_SIZE * gradient
            weights_sgd.append(W_SGD)
    average_sgd = np.zeros((NUMBER_OF_DATA_POINTS // BATCH_SIZE,))
    for sgd_run in sgd_loss:
        average_sgd += (np.array(sgd_run) - min_loss)
    average_sgd /= REPETITIONS
    plt.figure()
    plt.plot(average_sgd)
    weights_gd = jnp.array(weights_gd)
    weights_sgd = jnp.array(weights_sgd)
    print(weights_gd)
    weights = jnp.linspace(-2, 8, 100)
    plt.figure()
    plt.plot(weights, batched_loss(weights, X, Y), 'r')
    plt.plot(weights_gd, batched_loss(weights_gd, X, Y), '*--')
    plt.plot(weights_sgd, batched_loss(weights_sgd, X, Y), '*--')
    plt.show()
