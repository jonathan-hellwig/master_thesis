from svag import *
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
def sde_update(parameters, x, y, key, step_size, learning_rate=0.1):
    gradients = grad(loss)
    full_gradients = gradients(parameters, x, y)
    brownian_motion = []
    # Does the normal noise get generated correclty?
    for (w, b) in parameters:
        key, key_w, key_b = random.split(key, 3)
        brownian_motion.append((step_size * random.normal(key_w, (w.size, 1)),
                                step_size * random.normal(key_b, b.shape)))
    covariances = one_sample_covariance(parameters, x, y, gradients)
    sqrt_covariances = [
        (jnp.sqrt(learning_rate) * jnp.real(linalg.sqrtm(sigma_w)),
         jnp.sqrt(learning_rate) * jnp.real(linalg.sqrtm(sigma_b)))
        for (sigma_w, sigma_b) in covariances
    ]
    additive_noise = [
        (jnp.dot(sqrt_sigma_w,
                 noise_w).reshape(w.shape), jnp.dot(sqrt_sigma_b, noise_b))
        for (sqrt_sigma_w, sqrt_sigma_b), (noise_w, noise_b), (
            w, _) in zip(sqrt_covariances, brownian_motion, parameters)
    ]
    return [(w - step_size * dw, b - step_size * db) for (w, b), (
        dw, db), (noise_w,
                  noise_b) in zip(parameters, full_gradients, additive_noise)]

def euler_step(x, drift, diffusion, step_size, key):
    random_normal_increment = random.normal(key, x.shape)
    return x + drift * step_size + jnp.sqrt(step_size) * jnp.dot(diffusion, random_normal_increment)

def diffusion(parameters, x, y, gradients, learning_rate):
    covariances = one_sample_covariance(parameters, x, y, gradients)
    sqrt_covariances = [
        (jnp.sqrt(learning_rate) * jnp.real(linalg.sqrtm(sigma_w)),
         jnp.sqrt(learning_rate) * jnp.real(linalg.sqrtm(sigma_b)))
        for (sigma_w, sigma_b) in covariances
    ]
def one_sample_covariance(parameters, x, y, gradients):
    covariances_dw = []
    covariances_db = []
    full_gradients = gradients(parameters, x, y)
    for parameter in parameters:
        covariances_dw.append(jnp.zeros(
            (parameter[0].size, parameter[0].size)))
        covariances_db.append(jnp.zeros((parameter[1].size, 1)))
    number_of_points, _ = x.shape
    for i in range(number_of_points):
        partial_gradients = gradients(parameters, x[i].reshape((1, 1)),
                                      y[i].reshape((1, 1)))
        for j, (full_dw, full_db) in enumerate(full_gradients):
            partial_dw, partial_db = partial_gradients[j]
            covariances_dw[j] += covariate(full_dw, partial_dw)
            covariances_db[j] += covariate(full_db, partial_db)
    return [
        (covariance_dw, covariance_db)
        for covariance_dw, covariance_db in zip(covariances_dw, covariances_db)
    ]


def covariate(full, partial):
    return jnp.dot((jnp.ravel(partial) - jnp.ravel(full))[:, jnp.newaxis],
                   (jnp.ravel(partial) - jnp.ravel(full))[:, jnp.newaxis].T)


def sde_run():
    jax.config.update('jax_platform_name', 'cpu')
    number_of_points = 128
    x = jnp.linspace(0.0, 2.0 * jnp.pi, number_of_points)
    x = x.reshape((number_of_points, 1))
    y = jnp.sin(x)
    print(y.shape)

    key = random.PRNGKey(1)
    sizes = [1, 64, 1]
    parameters = init_network_parameters(sizes, key)
    step_size = 0.01
    loss_values = []
    for i in range(1):
        loss_values.append([])
        for epoch in range(50000):
            key, subkey = random.split(key)
            parameters = sde_update(parameters,
                                    x,
                                    y,
                                    subkey,
                                    step_size,
                                    learning_rate=0.1)
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


if __name__ == "__main__":
    sde_run()
