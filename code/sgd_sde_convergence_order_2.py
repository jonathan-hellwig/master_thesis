
from functools import partial
from sgd_sde_convergence_order import calculate_error, fit_convergance_line
import matplotlib.pyplot as plt
from tqdm import tqdm
import ml_collections
import jax
import jax.numpy as jnp
from jax.scipy import linalg

def objective(x, gamma, Q, D):
    return 0.5 * (Q.T @ x).T @ (D + jnp.diag(gamma)) @ (Q.T @ x)

@partial(jax.vmap, in_axes = (0, None, None, None))
def expected_objective(time, initial_value, H, learning_rate):
    return 0.5 * jnp.exp(learning_rate * time) * initial_value.T @ H @ linalg.expm(-2 * H * time) @ initial_value

@jax.jit
@partial(jax.vmap, in_axes=(0, 0, None, None, None))
def stochastic_gradient_update(x, gamma, Q, D, learning_rate):
    gradient = jax.grad(objective)(x, gamma, Q, D)
    updated_x = x - learning_rate * gradient
    value = 0.5 * updated_x.T @ Q @ D @ Q.T @ updated_x
    return value, updated_x


if __name__ == "__main__":

    config = ml_collections.ConfigDict()
    config.seed = 4
    config.dimension = 3
    config.initial_value_noise_scaling = 5.0
    config.final_time = 2.0
    config.number_of_samples = 1000000
    config.learning_rates = 2.0**jnp.arange(-1, -8, -1)

    key = jax.random.PRNGKey(config.seed)
    H = jnp.identity(config.dimension)
    key, subkey = jax.random.split(key)
    x_0 = config.initial_value_noise_scaling * jax.random.normal(subkey,
                                                    (config.dimension,))
    sampled_values = [[] for i in range(len(config.learning_rates))]
    for index, learning_rate in enumerate(config.learning_rates):
        max_iterations = int(config.final_time / learning_rate)
        x = jnp.tile(x_0, (config.number_of_samples,1))
        for _ in tqdm(range(max_iterations)):
            key, subkey = jax.random.split(key)
            normal_noise = jax.random.normal(subkey, (config.number_of_samples, config.dimension))
            value, x = stochastic_gradient_update(x, normal_noise, H, H,
                                                learning_rate)
            sampled_values[index].append(jnp.average(value))
    sampled_values = [jnp.array(sampled_value) for sampled_value in sampled_values]
    first_order_error = calculate_error(config.learning_rates, sampled_values, expected_objective, x_0, H, config.final_time)

    plt.figure(0)
    for i, learning_rate in enumerate(config.learning_rates):
        if i == 3:
            values = jnp.insert(sampled_values[i], 0, 0.5 * x_0.T @ H @ x_0)
            plt.plot(values, label = "Average SGD loss value")
            time = jnp.arange(0.0, config.final_time + learning_rate,
                            learning_rate)
            expected_first_order_loss_value = expected_objective(
                time, x_0, H, learning_rate)
            plt.plot(expected_first_order_loss_value, label = 'Expected loss value')
    plt.xlabel('time') 
    plt.ylabel('loss value')
    plt.legend()
    first_order_coefficients = fit_convergance_line(first_order_error, config.learning_rates)
    print(first_order_coefficients)
    plt.figure(1)
    for learning_rate, error in zip(config.learning_rates, first_order_error):
        plt.plot(learning_rate, error, 'b*')
    t = jnp.linspace(config.learning_rates[-1], config.learning_rates[0], 1000)
    y = jnp.exp(first_order_coefficients[1]) * t**first_order_coefficients[0]
    plt.plot(t, y, '-', label='first order')
    plt.gca().invert_xaxis()
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.show()
