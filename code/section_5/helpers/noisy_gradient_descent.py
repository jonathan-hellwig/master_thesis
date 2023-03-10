from functools import partial
import jax.scipy.linalg as linalg
import jax.numpy as jnp
from helpers.network import root_mean_square_loss
import jax



@jax.jit
def partial_covariance_update(key, parameters, x, y, time_step, learning_rate):
    full_gradients = jax.grad(root_mean_square_loss)(parameters, x, y)

    drifts = [(-dw, -db) for (dw, db) in full_gradients]
    diffusions = diffusion(parameters, x, y, learning_rate)
    keys = jax.random.split(key, 2 * len(parameters))
    keys = [(key_w, key_b) for key_w, key_b in zip(keys[::2], keys[1::2])]
    updated_parameters = []
    for (weight, bias), (drift_w,
                         drift_b), (diffusion_w,
                                    diffusion_b), (key_w, key_b) in zip(
                                        parameters, drifts, diffusions, keys):
        weight_brownian_increment = jnp.sqrt(time_step) * jax.random.normal(
            key_w, (weight.size, 1))
        bias_brownian_increment = jnp.sqrt(time_step) * jax.random.normal(
            key_b, (bias.size, 1))
        sigma_norm = jnp.linalg.norm(diffusion_w)
        updated_parameters.append(
            (euler_step(weight.reshape(weight.size, 1),
                        drift_w.reshape(weight.size,
                                        1), diffusion_w, time_step,
                        weight_brownian_increment).reshape(weight.shape),
             euler_step(bias, drift_b, diffusion_b, time_step,
                        bias_brownian_increment)))
    return updated_parameters, sigma_norm


@jax.jit
def euler_step(x, drift, diffusion, step_size, brownian_increment):
    return x + drift * step_size + jnp.dot(diffusion, brownian_increment)


@jax.jit
def diffusion(parameters, x, y, learning_rate):
    covariances = diagonal_one_sample_covariance(parameters, x, y)
    sqrt_covariances = [(jnp.real(linalg.sqrtm(learning_rate * covariance_w)),
                         jnp.real(linalg.sqrtm(learning_rate * covariance_b)))
                        for (covariance_w, covariance_b) in covariances]
    return sqrt_covariances


@jax.jit
def diagonal_one_sample_covariance(parameters, inputs, outputs):
    assert (len(inputs) == len(outputs))
    covariances_dw = []
    covariances_db = []
    full_gradients = jax.grad(root_mean_square_loss)(parameters, inputs, outputs)
    for (w, b) in parameters:
        covariances_dw.append(jnp.zeros((w.size, w.size)))
        covariances_db.append(jnp.zeros(b.shape))

    for input, output in zip(inputs, outputs):
        partial_gradients = jax.grad(root_mean_square_loss)(parameters,
                                                        input.reshape((1, 1)),
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


@jax.jit
def covariate(full, partial):
    return jnp.dot((partial - full), (partial - full).T)


@jax.jit
def sqrt_matrix(normalized_matrix):
    # TODO: Handle the other case
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


def flatten(xss):
    return [x for xs in xss for x in xs]


@jax.jit
def stack_parameters(parameters):
    flattend_parameters = flatten(parameters)
    stacked_parameters = jnp.vstack(
        tuple(parameter.reshape((-1, 1)) for parameter in flattend_parameters))
    return stacked_parameters


@partial(jax.jax.jit, static_argnames=['sizes'])
def unstack_parameters(sizes, stacked_parameters):
    split_indices = get_split_indices(sizes)
    split_parameters = jnp.split(stacked_parameters, split_indices)
    reshaped_parameters = [(w.reshape(
        (out_size, in_size)), b) for w, b, (
            in_size,
            out_size) in zip(split_parameters[::2], split_parameters[1::2],
                             zip(sizes, sizes[1:]))]
    return reshaped_parameters


# TODO: Correct for batch size
@jax.jit
def get_full_covariance(parameters, x, y):
    gradient = jax.grad(root_mean_square_loss)
    full_gradient = stack_parameters(gradient(parameters, x, y))
    gradient_samples = jnp.hstack(
        tuple(
            stack_parameters(gradient(parameters, a, b))
            for a, b in zip(x, y)))
    normalized_gradient_samples = gradient_samples - full_gradient
    sqrt_covariance = sqrt_matrix(normalized_gradient_samples) / jnp.sqrt(
        len(x))
    return sqrt_covariance


@jax.jit
def get_drift(parameters, x, y):
    return -stack_parameters(jax.grad(root_mean_square_loss)(parameters, x, y))


@partial(jax.jax.jit, static_argnames=['sizes'])
def full_covariance_update(key, parameters, sizes, x, y, time_step,
                           learning_rate):
    stacked_parameters = stack_parameters(parameters)
    mu = -stack_parameters(jax.grad(root_mean_square_loss)(parameters, x, y))
    sigma = jnp.sqrt(learning_rate) * get_full_covariance(parameters, x, y)
    brownian_increment = jnp.sqrt(time_step) * jax.random.normal(key, mu.shape)
    updated_parameters = euler_step(stacked_parameters, mu, sigma, time_step,
                                    brownian_increment)
    unstacked_parameters = unstack_parameters(sizes, updated_parameters)
    return unstacked_parameters, jnp.linalg.norm(sigma)
