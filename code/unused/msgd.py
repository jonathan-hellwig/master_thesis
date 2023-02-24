# %%
import jax.numpy as jnp
import flax.linen as nn
import jax
key = jax.random.PRNGKey(3)
input = jax.random.normal(key, (3,))
model = nn.Dense(features=5)
key, init_key = jax.random.split(key)
parameters = model.init(init_key, input)
# %%
print(model.apply(parameters, input))

# %%


def normal_sampling(key, n, b):
    sampling_covariance = 1 / \
        jnp.sqrt(n * b) * (jnp.eye(n) - 1/n * jnp.ones((n, n)))
    normal_increment = jax.random.normal(key, (n, ))
    return jnp.dot(sampling_covariance, normal_increment)


def loss(parameters, x, y, sampling):
    prediction = model.apply(parameters, x)
    return jnp.inner(jnp.square(prediction - y), sampling)


key, subkey, sampling_key = jax.random.split(key, 3)
output = jax.random.normal(subkey, (5,))
sampling = jnp.ones((5,)) / 5 + normal_sampling(sampling_key, 5, 1)
gradient = jax.grad(loss)(parameters, input, output, sampling)
print(loss(parameters, input, output, sampling))
print(gradient)
