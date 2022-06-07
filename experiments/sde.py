from svag import *

jax.config.update('jax_platform_name', 'cpu')

# Goals
# There are two points to investigate:
# 1. Do the covariances get calculated correctly?
# 2. Does the random noise get sampled correctly?

# Goal
# 1. How do I make sure I have implemented the algorithm correctly?
#   - Extract funtionality into methods
#   - Test output of methods
#   - Look at very basic test cases
#   - Look at averages and standard deviations
#   - The average should result in gradient descent
# 2. How can I make the algorithm run more quickly?
#   - Benchmark code
#   - Find the parts of the code that run the longest
#   - Look for speed optimizations using jax


# What does W * W.T mean when W is a matrix?
@jit
def sde_update(parameters, x, y, step_size, learning_rate, key):
    full_gradients = grad(loss)(parameters, x, y)

    drifts = [(-dw, -db) for (dw, db) in full_gradients]
    diffusions = diffusion(parameters, x, y, learning_rate)
    keys = random.split(key, 2 * len(parameters))
    keys = [(key_w, key_b) for key_w, key_b in zip(keys[::2], keys[1::2])]
    return [
        (euler_step(w.reshape(w.size, 1), drift_w.reshape(w.size, 1),
                    diffusion_w, step_size, key_w).reshape(w.shape),
         euler_step(b, drift_b, diffusion_b, step_size, key_b))
        for (w,
             b), (drift_w, drift_b), (diffusion_w, diffusion_b), (
                 key_w, key_b) in zip(parameters, drifts, diffusions, keys)
    ]


@jit
def gd_update(parameters, x, y, learning_rate):
    gradients = grad(loss)(parameters, x, y)
    return [(w - learning_rate * dw, b - learning_rate * db)
            for (w, b), (dw, db) in zip(parameters, gradients)]


# Use the usual euler update and convert everything to the correct size after the update
@jit
def euler_step(x, drift, diffusion, step_size, key):
    random_normal_increment = random.normal(key, x.shape)
    # print(
    #     f'x: {x}\n drift: {drift}\n diffusion: {diffusion}\n random_normal_increment: {random_normal_increment}'
    # )
    return x + drift * step_size + step_size * jnp.dot(
        diffusion, random_normal_increment)


@jit
def diffusion(parameters, x, y, learning_rate):
    covariances = one_sample_covariance(parameters, x, y)
    sqrt_covariances = [(jnp.real(linalg.sqrtm(learning_rate * covariance_w)),
                         jnp.real(linalg.sqrtm(learning_rate * covariance_b)))
                        for (covariance_w, covariance_b) in covariances]
    return sqrt_covariances


@jit
def one_sample_covariance(parameters, inputs, outputs):
    covariances_dw = []
    covariances_db = []
    full_gradients = grad(loss)(parameters, inputs, outputs)
    for (w, b) in parameters:
        covariances_dw.append(jnp.zeros((w.size, w.size)))
        covariances_db.append(jnp.zeros(b.shape))

    for input, output in zip(inputs, outputs):
        partial_gradients = grad(loss)(parameters, input.reshape((1, 1)),
                                       output.reshape((1, 1)))
        for j, ((full_dw, full_db), (partial_dw, partial_db)) in enumerate(
                zip(full_gradients, partial_gradients)):
            covariances_dw[j] += covariate(
                full_dw.reshape((full_dw.size, 1)),
                partial_dw.reshape((partial_dw.size, 1)))
            covariances_db[j] += covariate(full_db, partial_db)

    return [
        (covariance_dw, covariance_db)
        for covariance_dw, covariance_db in zip(covariances_dw, covariances_db)
    ]


@jit
def covariate(full, partial):
    return jnp.dot((partial - full), (partial - full).T)


def sde_run():
    number_of_points = 128
    x = jnp.linspace(0.0, 2.0 * jnp.pi, number_of_points)
    x = x.reshape((number_of_points, 1))
    y = jnp.sin(x)
    print(y.shape)

    key = random.PRNGKey(1)
    sizes = [1, 8, 1]
    parameters = init_network_parameters(sizes, key)
    step_size = 0.01
    learning_rate = 0.1
    loss_values = []
    for i in range(1):
        loss_values.append([])
        for epoch in range(50000):
            key, subkey = random.split(key)
            parameters = sde_update(parameters, x, y, step_size, learning_rate,
                                    subkey)
            loss_values[i].append(loss(parameters, x, y))
            print(f'epoch: {epoch}, loss: {loss_values[i][epoch]}')
    loss_values = jnp.array(loss_values)
    y_hat = manual_batched_predict(parameters, x)
    print(y_hat.shape)
    print(jnp.mean(jnp.square(y - y_hat)))
    plt.figure()
    plt.plot(jnp.mean(loss_values, axis=0))

    plt.figure()
    plt.plot(x, y_hat, x, y)
    plt.show()


def batches(x, y, batch_size, key):
    assert (len(x) == len(y))
    number_of_splits = jnp.ceil(len(x) / batch_size)
    permuated_indices = random.permutation(key, len(x))
    return zip(jnp.array_split(x[permuated_indices], number_of_splits),
               jnp.array_split(y[permuated_indices], number_of_splits))

# Goals
# 1. Write test cases
# 2. Adjust the iterations of each run such that they can be compared
def linear_run():
    x = jnp.linspace(0.0, 2.0, 100).reshape((100, 1))
    y = x

    key = random.PRNGKey(1)
    sizes = [1, 1]
    sde_parameters = init_network_parameters(sizes, key)
    gd_parameters = sde_parameters.copy()
    sgd_parameters = sde_parameters.copy()
    print(f'parameters: {sde_parameters}')
    step_size = 0.001
    learning_rate = 0.1
    sde_iterations = 500
    sde_losses = []
    for iteration in range(sde_iterations):
        key, subkey = random.split(key)
        sde_parameters = sde_update(sde_parameters, x, y, step_size,
                                    learning_rate, subkey)
        sde_losses.append(loss(sde_parameters, x, y))
        print(f'iteration: {iteration}, loss: {sde_losses[-1]}')
    gd_epochs = 5
    gd_losses = []
    for epoch in range(gd_epochs):
        gd_parameters = gd_update(gd_parameters, x, y, learning_rate)
        gd_losses.append(loss(gd_parameters, x, y))
        print(f'iteration: {epoch}, loss: {gd_losses[-1]}')

    sgd_epochs = 5
    batch_size = 16
    sgd_losses = []
    for epoch in range(sgd_epochs):
        key, sub_key = random.split(key)
        for x_batch, y_batch in batches(x, y, batch_size, sub_key):
            sgd_parameters = gd_update(sgd_parameters, x_batch, y_batch,
                                       learning_rate)
            sgd_losses.append(loss(sgd_parameters, x, y))
        print(f'iteration: {epoch}, loss: {sgd_losses[-1]}')

    y_hat = manual_batched_predict(sde_parameters, x)
    print(jnp.square(y - y_hat))
    plt.figure()
    plt.plot(x, y_hat, x, y)

    plt.figure()
    plt.plot(
        jnp.arange(len(sde_losses)) / len(sde_losses), sde_losses,
        jnp.arange(len(gd_losses)) / len(gd_losses), gd_losses,
        jnp.arange(len(sgd_losses)) / len(sgd_losses), sgd_losses)
    plt.show()


if __name__ == "__main__":
    linear_run()
