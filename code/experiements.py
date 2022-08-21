import jax.numpy as jnp
from jax import random
from experiments.noisy_gradient_descent import *


def sde_run():
    number_of_points = 128
    x = jnp.linspace(0.0, 2.0 * jnp.pi, number_of_points)
    x = x.reshape((number_of_points, 1))
    y = jnp.sin(x)
    print(y.shape)

    key = random.PRNGKey(1)
    sizes = [1, 8, 1]
    parameters = initialize_network_parameters(sizes, key)
    step_size = 0.01
    learning_rate = 0.1
    loss_values = []
    for i in range(1):
        loss_values.append([])
        for epoch in range(50000):
            key, subkey = random.split(key)
            parameters = partial_covariance_update(parameters, x, y,
                                                       step_size,
                                                       learning_rate, subkey)
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


def linear_test_case():
    x = jnp.linspace(0.0, 2.0, 100).reshape((100, 1))
    y = x

    key = random.PRNGKey(1)
    sizes = [1, 1]
    sde_parameters = initialize_network_parameters(sizes, key)
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
            sde_parameters = partial_covariance_update(sde_parameters, x, y, step_size,
                                        learning_rate, subkey)
            sde_losses.append(loss(sde_parameters, x, y))

        # sampling with replacement
        key, subkey = random.split(key)
        sgd_x_batch, sgd_y_batch = sample_batch(subkey, x, y, batch_size)
        sgd_parameters = gradient_descent_update(sgd_parameters, sgd_x_batch, sgd_y_batch,
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

        gd_parameters = gradient_descent_update(gd_parameters, x, y, learning_rate)
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
        svag_parameters.append(initialize_network_parameters(sizes, key))
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
        svag_parameters.append(initialize_network_parameters(sizes, key))
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
    sde_parameters = initialize_network_parameters(sizes, key)
    old_sde_parameters = initialize_network_parameters(sizes, key)
    for _ in svag_l:
        svag_parameters.append(sde_parameters.copy())
    step_size = 0.001
    final_time = 5.5
    learning_rate = 0.1
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
                sde_parameters = full_covariance_update(subkey, sde_parameters, sizes,
                                                x, y, step_size, learning_rate)
                w_sde.append(sde_parameters[0][0][0])

                key, subkey = random.split(key)
                old_sde_parameters = partial_covariance_update(old_sde_parameters, x, y,
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
                 "b-",
                 alpha=0.5,
                 label=f"svag l={l}")
    plt.plot(t, jnp.array(w_sde).flatten(), "r-.", label=f"sde w")
    plt.plot(t, jnp.array(old_w_sde).flatten(), label=f"old sde w")
    # plt.plot(t, jnp.array(b_sde).flatten(), label=f"sde b")
    plt.legend()
    plt.show()


def linear_svag_sde_test_2():
    x = jnp.linspace(0.0, 2.0, 100).reshape((100, 1))
    y = x

    key = random.PRNGKey(2)
    sizes = [1, 1]
    sde_parameters = initialize_network_parameters(sizes, key)
    old_sde_parameters = initialize_network_parameters(sizes, key)
    step_size = 0.001
    final_time = 5.5
    learning_rate = 0.1
    t = jnp.arange(0.0, final_time, step_size)
    batch_size = 32
    sde_solver_iterations = 10
    w_sde = []
    old_w_sde = []
    current_time = 0.0

    # print(sde_parameters[0][0][0])
    # print(sde_parameters[0][1][0])
    while current_time < final_time - 0.0001:
        key, subkey = random.split(key)
        for _ in range(sde_solver_iterations):
            key, subkey = random.split(key)
            sde_parameters = full_covariance_update(subkey, sde_parameters, sizes, x,
                                            y, step_size, learning_rate)
            w_sde.append(sde_parameters[0][0][0])

            key, subkey = random.split(key)
            old_sde_parameters = partial_covariance_update(old_sde_parameters, x, y,
                                            step_size, learning_rate, key)
            old_w_sde.append(old_sde_parameters[0][0][0])

        current_time += sde_solver_iterations * step_size
        if current_time >= final_time - 0.0001:
            break
        print(current_time)

    plt.figure()
    plt.plot(t, jnp.array(w_sde).flatten(), "r-.", label=f"sde w")
    plt.plot(t, jnp.array(old_w_sde).flatten(), label=f"old sde w")
    # plt.plot(t, jnp.array(b_sde).flatten(), label=f"sde b")
    plt.legend()
    plt.show()


def sin_test_case():
    x = jnp.linspace(0.0, 2.0, 100).reshape((100, 1))
    y = jnp.sin(x)

    key = random.PRNGKey(1)
    sizes = [1, 128, 1]
    sde_parameters = initialize_network_parameters(sizes, key)
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
            sde_parameters = partial_covariance_update(sde_parameters, x, y, step_size,
                                        learning_rate, subkey)
            sde_losses.append(loss(sde_parameters, x, y))

        key, key_x, key_y = random.split(key, 3)
        sgd_parameters = gradient_descent_update(
            sgd_parameters, random.choice(key_x, x, shape=(batch_size, 1)),
            random.choice(key_y, y, shape=(batch_size, 1)), learning_rate)
        sgd_losses.append(loss(sgd_parameters, x, y))

        gd_parameters = gradient_descent_update(gd_parameters, x, y, learning_rate)
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
