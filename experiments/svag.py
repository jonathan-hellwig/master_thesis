from cmath import pi
from random import random
from jax import block_until_ready, grad, jit, random, vmap
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.special import logsumexp

# Goals
# 1. define a neural network model using jax
# 2. make a prediction
# 3. train the neural network model using GD
# 4. train the neural network model using SGD
# 5. implement SVAG

def tanh(x):
    y = jnp.zeros(x.shape)
    for i in range(100):
        y += jnp.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)

def random_layer_parameters(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_network_parameters(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_parameters(m,n,k) for m,n,k in zip(sizes[:-1], sizes[1:], keys)]

def predict(parameters, x):
    activation = x
    for w, b in parameters[:-1]:
        outputs = jnp.dot(w, activation) + b
        activation = jnp.tanh(outputs)
    final_w, final_b = parameters[-1]
    logits = jnp.dot(final_w, activation) + final_b
    return logits

@jit
def loss(parameters, x, y):
    batched_predict = vmap(predict, in_axes=(None, 0))
    error = batched_predict(parameters, x) - y
    return jnp.mean(jnp.square(batched_predict(parameters, x) - y))

@jit
def sgd_update(parameters, x, y, step_size = 0.1):
    gradients = grad(loss)(parameters, x, y)
    return [(w - step_size * dw, b - step_size * db) for (w,b), (dw, db) in zip(parameters, gradients)]


@jit
def svag_update(parameters, x, y, step_size=0.1):
    gradients = grad(loss)(parameters, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(parameters, gradients)]

def get_batches(x, y, batch_size, key):
    number_of_splits = jnp.ceil(len(x)/batch_size)
    return zip(jnp.array_split(random.shuffle(key, x), number_of_splits),
               jnp.array_split(random.shuffle(key, y), number_of_splits))

def main():
    jax.config.update('jax_platform_name', 'cpu')
    # grad_tanh = grad(tanh)
    # print(grad_tanh(1.0))
    number_of_points = 1024
    x = jnp.linspace(0.0, 2.0 * jnp.pi, number_of_points)
    # x = jnp.split(x,2)
    x = x.reshape((number_of_points, 1))
    # print(x)
    y = jnp.sin(x)

    # plt.plot(x,y)
    # plt.show()
    batch_size = 64
    key = random.PRNGKey(0)
    sizes = [1, 2048, 1]
    parameters = init_network_parameters(sizes, key)
    step_size = 0.1
    # decay = 0.0001
    for epoch in range(1000):
        # step_size *= 1.0 / (1.0 + decay * epoch)
        for x_batch, y_batch in get_batches(x,y,batch_size, key):
            parameters = sgd_update(parameters, x_batch, y_batch)
            loss_value = loss(parameters, x_batch, y_batch)
            print(f'Epoch: {epoch}, loss: {loss_value}, step_size: {step_size}')

    batched_predict = vmap(predict, in_axes=(None, 0))
    y_hat = batched_predict(parameters, x)
    print(y_hat.shape)
    print(jnp.mean(jnp.square(y - y_hat)))
    plt.plot(x, jnp.square(y - y_hat))
    plt.plot(x, y_hat, x, y)
    plt.show()




if __name__ == "__main__":
    main()
