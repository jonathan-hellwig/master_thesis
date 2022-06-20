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
                    diffusion_w, step_size,
                    jnp.sqrt(step_size) *
                    random.normal(key_w, (w.size, 1))).reshape(w.shape),
         euler_step(b, drift_b, diffusion_b, step_size,
                    jnp.sqrt(step_size) * random.normal(key_b, (b.size, 1))))
        for (w,
             b), (drift_w, drift_b), (diffusion_w, diffusion_b), (
                 key_w, key_b) in zip(parameters, drifts, diffusions, keys)
    ]


@jit
def gd_update(parameters, x, y, learning_rate):
    gradients = grad(loss)(parameters, x, y)
    return [(w - learning_rate * dw, b - learning_rate * db)
            for (w, b), (dw, db) in zip(parameters, gradients)]


@jit
def euler_step(x, drift, diffusion, step_size, brownian_increment):
    return x + drift * step_size + jnp.dot(diffusion, brownian_increment)


@jit
def diffusion(parameters, x, y, learning_rate):
    covariances = diagonal_one_sample_covariance(parameters, x, y)
    sqrt_covariances = [(jnp.real(linalg.sqrtm(learning_rate * covariance_w)),
                         jnp.real(linalg.sqrtm(learning_rate * covariance_b)))
                        for (covariance_w, covariance_b) in covariances]
    return sqrt_covariances


@jit
def diagonal_one_sample_covariance(parameters, inputs, outputs):
    assert (len(inputs) == len(outputs))
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
        (covariance_dw / len(inputs), covariance_db / len(inputs))
        for covariance_dw, covariance_db in zip(covariances_dw, covariances_db)
    ]


def full_one_sample_covariance(parameters, inputs, outputs):
    pass


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
    y_hat = batched_predict(parameters, x)
    print(y_hat.shape)
    print(jnp.mean(jnp.square(y - y_hat)))
    plt.figure()
    plt.plot(jnp.mean(loss_values, axis=0))

    plt.figure()
    plt.plot(x, y_hat, x, y)
    plt.show()


def flatten(xss):
    return [x for xs in xss for x in xs]


def batches(x, y, batch_size, key):
    assert (len(x) == len(y))
    number_of_splits = jnp.ceil(len(x) / batch_size)
    permuated_indices = random.permutation(key, len(x))
    return zip(jnp.array_split(x[permuated_indices], number_of_splits),
               jnp.array_split(y[permuated_indices], number_of_splits))


def sample_batch(key, x, y, batch_size):
    key, key_x, key_y = random.split(key, 3)
    return random.choice(key_x, x,
                         shape=(batch_size,
                                1)), random.choice(key_y,
                                                   y,
                                                   shape=(batch_size, 1))


@jit
def get_sqrt_matrix(normalized_matrix):
    rows, columns = normalized_matrix.shape
    # TODO: Handle the other case
    # assert (rows < columns)
    u, s, _ = jnp.linalg.svd(normalized_matrix)
    return u @ jnp.diag(s)


def get_split_indices(sizes):
    indices = []
    current_index = 0
    for in_size, out_size in zip(sizes, sizes[1:]):
        indices.append(current_index + in_size * out_size)
        current_index += in_size * out_size
        indices.append(current_index + out_size)
        current_index += out_size
    return indices


def stack_parameters(parameters):
    flattend_parameters = flatten(parameters)
    stacked_parameters = jnp.vstack(
        tuple(parameter.reshape((-1, 1)) for parameter in flattend_parameters))
    return stacked_parameters


def unstack_parameters(sizes, stacked_parameters):
    split_indices = get_split_indices(sizes)
    split_parameters = jnp.split(stacked_parameters, split_indices)
    reshaped_parameters = [(w.reshape(
        (out_size, in_size)), b) for w, b, (
            in_size,
            out_size) in zip(split_parameters[::2], split_parameters[1::2],
                             zip(sizes, sizes[1:]))]
    return reshaped_parameters


@jit
def get_full_covariance(parameters, x, y):
    # assert (len(x) == len(y))
    gradient = grad(loss)
    full_gradient = stack_parameters(gradient(parameters, x, y))
    gradient_samples = jnp.hstack(
        tuple(
            stack_parameters(gradient(parameters, a, b))
            for a, b in zip(x, y)))
    normalized_gradient_samples = (gradient_samples - full_gradient) / len(x)
    sqrt_covariance = get_sqrt_matrix(normalized_gradient_samples)
    return sqrt_covariance


@jit
def get_drift(parameters, x, y):
    return -stack_parameters(grad(loss)(parameters, x, y))


def new_sde_update(key, parameters, sizes, x, y, time_step, learning_rate):
    stacked_parameters = stack_parameters(parameters)
    mu = -stack_parameters(grad(loss)(parameters, x, y))
    sigma = jnp.sqrt(learning_rate) * get_full_covariance(parameters, x, y)
    brownian_increment = jnp.sqrt(time_step) * random.normal(key, mu.shape)
    updated_parameters = euler_step(stacked_parameters, mu, sigma, time_step,
                                    brownian_increment)
    unstacked_parameters = unstack_parameters(sizes, updated_parameters)
    return unstacked_parameters


def linear_test_case():
    x = jnp.linspace(0.0, 2.0, 100).reshape((100, 1))
    y = x

    key = random.PRNGKey(1)
    sizes = [1, 1]
    sde_parameters = init_network_parameters(sizes, key)
    gd_parameters = sde_parameters.copy()
    sgd_parameters = sde_parameters.copy()
    svag_parameters = sde_parameters.copy()
    step_size = 0.001
    final_time = 5.0
    t = jnp.arange(0.0, final_time, step_size)
    learning_rate = 0.1
    batch_size = 32
    svag_l = 4
    solver_iterations = 10
    gd_losses = []
    sgd_losses = []
    sde_losses = []
    svag_losses = []
    for i in range(len(t[::solver_iterations])):
        for _ in range(solver_iterations):
            key, subkey = random.split(key)
            sde_parameters = sde_update(sde_parameters, x, y, step_size,
                                        learning_rate, subkey)
            sde_losses.append(loss(sde_parameters, x, y))

        # sampling with replacement
        key, subkey = random.split(key)
        sgd_x_batch, sgd_y_batch = sample_batch(subkey, x, y, batch_size)
        sgd_parameters = gd_update(sgd_parameters, sgd_x_batch, sgd_y_batch,
                                   learning_rate)
        sgd_losses.append(loss(sgd_parameters, x, y))

        key, subkey = random.split(key)
        svag_x_batch_first, svag_y_batch_first = sample_batch(
            subkey, x, y, batch_size)
        key, subkey = random.split(key)
        svag_x_batch_second, svag_y_batch_second = sample_batch(
            subkey, x, y, batch_size)
        svag_parameters = svag_update(svag_parameters, svag_x_batch_first,
                                      svag_y_batch_first, svag_x_batch_second,
                                      svag_y_batch_second, svag_l)
        svag_losses.append(loss(svag_parameters, x, y))

        gd_parameters = gd_update(gd_parameters, x, y, learning_rate)
        gd_losses.append(loss(gd_parameters, x, y))
        print(f'sgd t: {t[i]}, loss: {sgd_losses[-1]}')
        print(f'sde: {t[i]}, loss: {sde_losses[-1]}')
        print(f'gd t: {t[i]}, loss: {gd_losses[-1]}')

    y_hat_sde = batched_predict(sde_parameters, x)
    y_hat_gd = batched_predict(gd_parameters, x)
    y_hat_sgd = batched_predict(sgd_parameters, x)
    y_hat_svag = batched_predict(svag_parameters, x)
    plt.figure()
    plt.plot(x, y_hat_sde, label="sde")
    plt.plot(x, y_hat_gd, label="gd")
    plt.plot(x, y_hat_sgd, label="sgd")
    plt.plot(x, y_hat_svag, label="svag")
    plt.plot(x, y, label="ground truth")
    plt.legend()

    plt.figure()
    plt.plot(t, sde_losses, label="sde")
    plt.plot(t[::solver_iterations], gd_losses, label="gd")
    plt.plot(t[::solver_iterations], sgd_losses, label="sgd")
    plt.plot(t[::solver_iterations], svag_losses, label="svag")
    plt.legend()
    plt.show()


def linear_svag_test_case_with_replacement():
    x = jnp.linspace(0.0, 2.0, 100).reshape((100, 1))
    y = x

    key = random.PRNGKey(1)
    sizes = [1, 1]
    svag_l = [1, 2, 4, 8, 32, 64]
    svag_parameters = []
    for _ in svag_l:
        svag_parameters.append(init_network_parameters(sizes, key))
    step_size = 0.001
    final_time = 0.5
    learning_rate = 0.01
    t = jnp.arange(0.0, final_time, step_size)
    batch_size = 32
    svag_losses = []
    for _ in svag_l:
        svag_losses.append([])
    for i in range(len(t)):
        key, subkey = random.split(key)
        svag_x_batch_first, svag_y_batch_first = sample_batch(
            subkey, x, y, batch_size)
        key, subkey = random.split(key)
        svag_x_batch_second, svag_y_batch_second = sample_batch(
            subkey, x, y, batch_size)
        for i, l in enumerate(svag_l):
            svag_parameters[i] = svag_update(
                svag_parameters[i], svag_x_batch_first, svag_y_batch_first,
                svag_x_batch_second, svag_y_batch_second, l, learning_rate)
            svag_losses[i].append(loss(svag_parameters[i], x, y))
            print(f'sgd t: {t[i]}, loss: {svag_losses[i][-1]}')

    plt.figure()
    for i, l in enumerate(svag_l):
        y_hat_svag = batched_predict(svag_parameters[i], x)
        plt.plot(x, y_hat_svag, label=f"svag l={l}")

    plt.plot(x, y, label="ground truth")
    plt.legend()

    plt.figure()
    for i, l in enumerate(svag_l):
        plt.plot(t, svag_losses[i], label=f"svag l={l}")
    plt.legend()
    plt.show()


def linear_svag_test_case_without_replacement():
    x = jnp.linspace(0.0, 2.0, 100).reshape((100, 1))
    y = x

    key = random.PRNGKey(1)
    sizes = [1, 1]
    svag_l = [1, 2, 4, 8, 16, 32, 64]
    svag_parameters = []
    for _ in svag_l:
        svag_parameters.append(init_network_parameters(sizes, key))
    step_size = 0.001
    final_time = 0.5
    learning_rate = 0.1
    t = jnp.arange(0.0, final_time, step_size)
    batch_size = 32
    svag_losses = []
    current_time = 0.0
    for _ in svag_l:
        svag_losses.append([])
    while current_time < final_time:
        key, subkey = random.split(key)
        batches = list(get_batches(subkey, x, y, batch_size))
        for (svag_x_batch_first,
             svag_y_batch_first), (svag_x_batch_second,
                                   svag_y_batch_second) in zip(
                                       batches[::2], batches[1::2]):
            for i, l in enumerate(svag_l):
                svag_parameters[i] = svag_update(
                    svag_parameters[i], svag_x_batch_first, svag_y_batch_first,
                    svag_x_batch_second, svag_y_batch_second, l, learning_rate)
                svag_losses[i].append(loss(svag_parameters[i], x, y))
                print(f'sgd t: {t[i]}, loss: {svag_losses[i][-1]}')
            current_time += step_size

    plt.figure()
    for i, l in enumerate(svag_l):
        y_hat_svag = batched_predict(svag_parameters[i], x)
        plt.plot(x, y_hat_svag, label=f"svag l={l}")

    plt.plot(x, y, label="ground truth")
    plt.legend()

    plt.figure()
    for i, l in enumerate(svag_l):
        plt.plot(t, svag_losses[i], label=f"svag l={l}")
    plt.legend()
    plt.show()


def linear_svag_sde_test():
    x = jnp.linspace(0.0, 2.0, 100).reshape((100, 1))
    y = x

    key = random.PRNGKey(2)
    sizes = [1, 1]
    svag_l = [1, 2, 4, 8, 16, 32, 64]
    svag_parameters = []
    sde_parameters = init_network_parameters(sizes, key)
    old_sde_parameters = init_network_parameters(sizes, key)
    for _ in svag_l:
        svag_parameters.append(sde_parameters.copy())
    step_size = 0.001
    final_time = 5.5
    learning_rate = 0.05
    t = jnp.arange(0.0, final_time, step_size)
    batch_size = 32
    sde_solver_iterations = 10
    w_svag = []
    w_sde = []
    old_w_sde = []
    current_time = 0.0
    for _ in svag_l:
        w_svag.append([])
    # print(svag_parameters[0][0][0][0])
    # print(svag_parameters[0][0][1][0])

    # print(sde_parameters[0][0][0])
    # print(sde_parameters[0][1][0])
    while current_time < final_time - 0.0001:
        key, subkey = random.split(key)
        batches = list(get_batches(subkey, x, y, batch_size))
        for (svag_x_batch_first,
             svag_y_batch_first), (svag_x_batch_second,
                                   svag_y_batch_second) in zip(
                                       batches[::2], batches[1::2]):
            for _ in range(sde_solver_iterations):
                key, subkey = random.split(key)
                sde_parameters = new_sde_update(subkey, sde_parameters, sizes,
                                                x, y, step_size, learning_rate)
                w_sde.append(sde_parameters[0][0][0])

                key, subkey = random.split(key)
                old_sde_parameters = sde_update(old_sde_parameters, x, y,
                                                step_size, learning_rate, key)
                old_w_sde.append(old_sde_parameters[0][0][0])

            for i, l in enumerate(svag_l):
                svag_parameters[i] = svag_update(
                    svag_parameters[i], svag_x_batch_first, svag_y_batch_first,
                    svag_x_batch_second, svag_y_batch_second, l, learning_rate)
                w_svag[i].append(svag_parameters[i][0][0][0])

            current_time += sde_solver_iterations * step_size
            if current_time >= final_time - 0.0001:
                break
            print(current_time)

    plt.figure()
    for i, l in enumerate(svag_l):
        plt.plot(t[::sde_solver_iterations],
                 jnp.array(w_svag[i]).flatten(),
                 label=f"svag l={l}")
    plt.plot(t, jnp.array(w_sde).flatten(), label=f"sde w")
    plt.plot(t, jnp.array(old_w_sde).flatten(), label=f"old sde w")
    # plt.plot(t, jnp.array(b_sde).flatten(), label=f"sde b")
    plt.legend()
    plt.show()


def sin_test_case():
    x = jnp.linspace(0.0, 2.0, 100).reshape((100, 1))
    y = jnp.sin(x)

    key = random.PRNGKey(1)
    sizes = [1, 128, 1]
    sde_parameters = init_network_parameters(sizes, key)
    gd_parameters = sde_parameters.copy()
    sgd_parameters = sde_parameters.copy()
    step_size = 0.001
    final_time = 5.0
    t = jnp.arange(0.0, final_time, step_size)
    learning_rate = 0.1
    batch_size = 32
    solver_iterations = 10
    gd_losses = []
    sgd_losses = []
    sde_losses = []
    for i in range(len(t[::solver_iterations])):
        for _ in range(solver_iterations):
            key, subkey = random.split(key)
            sde_parameters = sde_update(sde_parameters, x, y, step_size,
                                        learning_rate, subkey)
            sde_losses.append(loss(sde_parameters, x, y))

        key, key_x, key_y = random.split(key, 3)
        sgd_parameters = gd_update(
            sgd_parameters, random.choice(key_x, x, shape=(batch_size, 1)),
            random.choice(key_y, y, shape=(batch_size, 1)), learning_rate)
        sgd_losses.append(loss(sgd_parameters, x, y))

        gd_parameters = gd_update(gd_parameters, x, y, learning_rate)
        gd_losses.append(loss(gd_parameters, x, y))
        print(f'sgd t: {t[i]}, loss: {sgd_losses[-1]}')
        print(f'sde: {t[i]}, loss: {sde_losses[-1]}')
        print(f'gd t: {t[i]}, loss: {gd_losses[-1]}')

    y_hat_sde = batched_predict(sde_parameters, x)
    y_hat_gd = batched_predict(sde_parameters, x)
    y_hat_sgd = batched_predict(sde_parameters, x)
    plt.figure()
    plt.plot(x, y_hat_sde, label="sde")
    plt.plot(x, y_hat_gd, label="gd")
    plt.plot(x, y_hat_sgd, label="sgd")
    plt.plot(x, y, label="ground truth")
    plt.legend()

    plt.figure()
    plt.plot(t, sde_losses, label="sde")
    plt.plot(t[::solver_iterations], gd_losses, label="gd")
    plt.plot(t[::solver_iterations], sgd_losses, label="sgd")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    linear_svag_sde_test()
