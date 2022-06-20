from locale import normalize
import math
from scipy.integrate import odeint
import jax.numpy as jnp
from svag import *
from sde import *
from numpy.testing import assert_allclose


# Take the square root in this example!!
def test_one_sample_covariance():
    points = 100
    x = jnp.linspace(0, 10, points).reshape((points, 1))
    y = 2.0 * x + 1.0
    key = random.PRNGKey(1)
    for _ in range(100):
        key, key_w, key_b = random.split(key, 3)
        w = 3.0 * random.normal(key_w, (1, 1))
        b = 3.0 * random.normal(key_b, (1, 1))
        mean_gradient_w = jnp.sum(2.0 * (w * x + b - y) * x) / points
        mean_gradient_b = jnp.sum(2.0 * (w * x + b - y)) / points

        expected_covariance_w = jnp.sum(
            jnp.square(mean_gradient_w - 2.0 * (w * x + b - y) * x)) / points
        expected_covariance_b = jnp.sum(
            jnp.square(mean_gradient_b - 2.0 * (w * x + b - y))) / points

        parameters = [(w * jnp.ones((1, 1)), b * jnp.ones((1, 1)))]
        covariance_w, covariance_b = diagonal_one_sample_covariance(
            parameters, x, y)[0]
        assert (jnp.allclose(expected_covariance_w, covariance_w))
        assert (jnp.allclose(expected_covariance_b, covariance_b))


def test_expected_value():
    points = 100
    x = jnp.linspace(0, 1.0, points).reshape((points, 1))
    y = 2.0 * x + 1.0
    key = random.PRNGKey(1)
    final_time = 10.0
    step_size = 0.01
    learning_rate = 0.1
    t = jnp.arange(0.0, final_time, 0.1)
    w = 3.0
    b = 1.0
    initial_parameters = [w, b]
    coefficients = (-2.0 * jnp.sum(jnp.square(x)) / len(x),
                    -2.0 * jnp.sum(x) / len(x), 2.0 * jnp.sum(x * y) / len(x),
                    -2.0 * jnp.sum(x) / len(x), -2.0,
                    2.0 * jnp.sum(y) / len(x))
    solution = odeint(right_hand_side,
                      initial_parameters,
                      t,
                      args=coefficients)
    # sde_parameters = sde_update(sde_parameters, x, y, step_size, learning_rate,
    #                             subkey)
    plt.plot(t, solution)
    plt.show()


def test_euler_murayama():
    key = random.PRNGKey(1)
    number_of_points = 1000
    sigma = 1.0
    mu = 1.0
    t = jnp.linspace(0.0, 10.0, number_of_points).reshape(
        (number_of_points, 1))
    dt = t[1] - t[0]
    initial_value = 10.0
    normal_increments = jnp.sqrt(dt) * random.normal(key,
                                                     (number_of_points, 1))
    brownian_motion = jnp.cumsum(normal_increments, axis=0)
    t = jnp.linspace(0.0, 10.0, number_of_points).reshape(
        (number_of_points, 1))
    solution = initial_value * jnp.exp((mu - sigma**2 / 2.0) * t +
                                       sigma * brownian_motion)
    approximation_steps = number_of_points // 10
    euler_approximation = jnp.zeros((approximation_steps, 1))
    euler_approximation = euler_approximation.at[0].set(initial_value)
    for i in range(1, approximation_steps):
        brownian_increment = brownian_motion[10 * i] - brownian_motion[10 *
                                                                       (i - 1)]
        euler_approximation = euler_approximation.at[i].set(
            euler_step(euler_approximation[i - 1],
                       mu * euler_approximation[i - 1],
                       sigma * euler_approximation[i - 1], dt * 10,
                       brownian_increment))
    plt.plot(t, solution, t[::10], euler_approximation)
    plt.show()


def right_hand_side(parameters, t, coefficients_0, coefficients_1,
                    coefficients_2, coefficients_3, coefficients_4,
                    coefficients_5):
    w, b = parameters
    dparametersdt = [
        coefficients_0 * w + coefficients_1 * b + coefficients_2,
        coefficients_3 * w + coefficients_4 * b + coefficients_5
    ]
    return dparametersdt


def test_square_root_matrix():
    key = random.PRNGKey(1)
    n = 100
    x = random.normal(key, (2, n))
    mean = jnp.mean(x, axis=1)
    x_normalized = x - mean.reshape((2, 1))
    sqrt_matrix = get_sqrt_matrix(x_normalized)
    assert_allclose(sqrt_matrix @ sqrt_matrix.T, x @ x.T, rtol=1e-3)


def test_map_parameters_to_vector():
    key = random.PRNGKey(1)
    sizes = [1, 10, 15, 20, 1]
    parameters = init_network_parameters(sizes, key)
    flattend_parameters = [
        weight_bias for parameter in parameters for weight_bias in parameter
    ]
    stacked_parameters = jnp.vstack(
        tuple(parameter.reshape((-1, 1)) for parameter in flattend_parameters))
    unstacked_parameters = unstack_parameters(sizes, stacked_parameters)
    for (w, b), (expected_w, expected_b) in zip(unstacked_parameters,
                                                parameters):
        assert_allclose(w, expected_w)
        assert_allclose(b, expected_b)



def test_euler_update():
    key = random.PRNGKey(1)
    sizes = [1, 5, 1]
    parameters = init_network_parameters(sizes, key)

    points = 100
    x = jnp.linspace(0, 10, points).reshape((points, 1))
    y = 2.0 * x + 1.0
    learning_rate = 0.1
    time_step = 0.001
    key, subkey = random.split(key)
    parameters = new_sde_update(key, parameters, sizes, x, y, time_step,
                                learning_rate)
    print(parameters)


def test_full_covariance():
    key = random.PRNGKey(1)
    sizes = [1, 5, 1]
    parameters = init_network_parameters(sizes, key)

    points = 100
    x = jnp.linspace(0, 10, points).reshape((points, 1))
    y = 2.0 * x + 1.0

    sqrt_covariance = get_full_covariance(parameters, x, y)

    # expected_gradient_w = jnp.vstack((2 * x[:, 0].reshape(
    #     (points, 1, 1)) * ((W @ x.reshape((points, 3, 1))) + b - y.reshape(
    #         (points, 3, 1)))).sum(axis=0), (2 * x[:, 1].reshape(
    #             (points, 1, 1)) * ((W @ x.reshape(
    #                 (points, 3, 1))) + b - y.reshape(
    #                     (points, 3, 1)))).sum(axis=0), (2 * x[:, 2].reshape(
    #                         (points, 1, 1)) * ((W @ x.reshape(
    #                             (points, 3, 1))) + b - y.reshape(
    #                                 (points, 3, 1)))).sum(axis=0)) / points
    # expected_gradient_b = (2 * ((W @ x.reshape(
    #     (points, 3, 1))) + b - y.reshape((100, 3, 1)))).sum(axis=0)
    # expected_gradient = jnp.vstack((expected_gradient_w, expected_gradient_b))
    # assert_allclose(expected_gradient, full_gradient)
    # # print(sqrt_covariance)
    # print(sqrt_covariance.shape)


# def test_one_sample_covariance():
#     points = 100
#     x = jnp.linspace(0, 10, points).reshape((points, 1))
#     y = 2.0 * x + 1.0
#     w = 1.0
#     b = 1.0
#     mean_gradient = jnp.sum(2.0 * (w * x + b - y) * x) / points
#     expected_covariance_w = jnp.sum(
#         jnp.square(mean_gradient - 2.0 * (w * x + b - y) * x)) / points
#     parameters = [(w * jnp.ones((1, 1)), b * jnp.ones((1, 1)))]
#     covariance_w, covariance_b = one_sample_covariance(parameters, x, y)[0]
#     print(expected_covariance_w, covariance_w)
#     assert (jnp.allclose(expected_covariance_w, covariance_w))

if __name__ == "__main__":
    test_euler_update()
