import jax.numpy as jnp
import jax.scipy.linalg as linalg
from jax import random
from helpers.noisy_gradient_descent import *
from helpers.gradient_descent import gradient_descent_update
from helpers.network import *
from helpers.stochastic_variance_amplified_gradient import *
from tqdm.auto import tqdm
import ml_collections

def calculate_error(learning_rates, sampled_values, expected_loss, x_0, H, final_time):
    error = []
    for learning_rate, sampled_value in zip(learning_rates, sampled_values):
        time = jnp.arange(0.0, final_time, learning_rate) + learning_rate
        expected_loss_value = expected_loss(
            time, x_0, H, learning_rate)
        error.append(jnp.max(jnp.abs(sampled_value - expected_loss_value)))
    return error

def fit_convergance_line(error, learning_rates):
    A = jnp.column_stack(
        (jnp.log(jnp.array(learning_rates)), jnp.ones((len(learning_rates), ))))
    log_error = jnp.log(jnp.array(error))
    coefficients, _, _, _ = jnp.linalg.lstsq(A, log_error)
    return coefficients

def sampled_loss(x, standard_normal_noise, H):
    return 0.5 * (x - standard_normal_noise).T @ H @ (x - standard_normal_noise) - 0.5 * jnp.trace(H)

@jit
@partial(jax.vmap, in_axes=(0, None, None, None))
def expected_first_order_loss(time, initial_value, H, learning_rate):
    decay_term = 0.5 * initial_value.T @ H @ linalg.expm(
        -2.0 * time * H) @ initial_value
    # Eigenvalue in ascending order
    eigenvalues, _ = linalg.eigh(H)
    noise_term = 0.25 * learning_rate * jnp.sum(
        jnp.square(eigenvalues) * (1 - jnp.exp(-2.0 * time * eigenvalues)))
    return decay_term + noise_term


@jit
@partial(jax.vmap, in_axes=(0, None, None, None))
def expected_second_order_loss(time, initial_value, H, learning_rate):
    decay_term = 0.5 * initial_value.T @ H @ linalg.expm(
        -(2 * H + learning_rate * H @ H) * time) @ initial_value
    eigenvalues, _ = linalg.eigh(H)
    noise_term = 0.5 * learning_rate * jnp.sum(
        jnp.square(eigenvalues) / (2 + learning_rate * eigenvalues) *
        (1 - jnp.exp(-eigenvalues *
                     (learning_rate * eigenvalues + 2.0) * time)))
    return decay_term + noise_term


@jit
@partial(jax.vmap, in_axes=(0, None, 0, None))
def stochastic_gradient_update(x, H, gamma, learning_rate):
    gradient = jax.grad(sampled_loss)(x, gamma, H)
    updated_x = x - learning_rate* gradient
    value = 0.5 * updated_x.T @ H @ updated_x
    return value, updated_x

if __name__ == "__main__":
    jax.config.update('jax_platform_name', 'cpu')

    config = ml_collections.ConfigDict()
    config.seed = 4
    config.dimension = 3
    config.initial_value_noise_scaling = 5.0
    config.final_time = 2.0
    config.number_of_samples = 100000
    config.learning_rates = 2.0**jnp.arange(-1, -5, -1)

    key = random.PRNGKey(config.seed)
    H = jnp.identity(config.dimension)
    key, subkey = random.split(key)
    x_0 = config.initial_value_noise_scaling * random.normal(subkey,
                                                    (config.dimension,))
    sampled_values = [[] for i in range(len(config.learning_rates))]
    for index, learning_rate in enumerate(config.learning_rates):
        max_iterations = int(config.final_time / learning_rate)
        x = jnp.tile(x_0, (config.number_of_samples,1))
        for _ in tqdm(range(max_iterations)):
            key, subkey = random.split(key)
            normal_noise = random.normal(subkey, (config.number_of_samples, config.dimension))
            value, x = stochastic_gradient_update(x, H, normal_noise,
                                                learning_rate)
            sampled_values[index].append(jnp.average(value))
    sampled_values = [jnp.array(sampled_value) for sampled_value in sampled_values]
    first_order_error = calculate_error(config.learning_rates, sampled_values, expected_first_order_loss, x_0, H, config.final_time)
    second_order_error = calculate_error(config.learning_rates, sampled_values, expected_second_order_loss, x_0, H, config.final_time)

    plt.figure(0)
    for i, learning_rate in enumerate(config.learning_rates):
        if i == 3:
            values = jnp.insert(sampled_values[i], 0, 0.5 * x_0.T @ H @ x_0)
            plt.plot(values, label = "Average SGD loss value")
            time = jnp.arange(0.0, config.final_time + learning_rate,
                            learning_rate)
            expected_first_order_loss_value = expected_first_order_loss(
                time, x_0, H, learning_rate)
            plt.plot(expected_first_order_loss_value, label = 'Expected loss value')
    plt.xlabel('time') 
    plt.ylabel('loss value')
    plt.legend()

    first_order_coefficients = fit_convergance_line(first_order_error, config.learning_rates)
    print(first_order_coefficients)
    second_order_coefficients = fit_convergance_line(second_order_error, config.learning_rates)
    print(second_order_coefficients)
    plt.figure(1)
    for learning_rate, error in zip(config.learning_rates, first_order_error):
        plt.plot(learning_rate,
                error,
                'b*')
    for learning_rate, error in zip(config.learning_rates, second_order_error):
        plt.plot(learning_rate,
                error,
                'r*')
    t = jnp.linspace(config.learning_rates[-1], config.learning_rates[0], 1000)
    y = jnp.exp(first_order_coefficients[1]) * t**first_order_coefficients[0]
    plt.plot(t, y, '-', label='first order')
    y = jnp.exp(second_order_coefficients[1]) * t**second_order_coefficients[0]
    plt.plot(t, y, '-', label='second order')
    plt.gca().invert_xaxis()
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('learning rate')
    plt.ylabel('max error')
    plt.legend()
    plt.show()
    # plt.savefig('/home/jonathan/forest/images/weak_order_2_SGD_fixed.png')

