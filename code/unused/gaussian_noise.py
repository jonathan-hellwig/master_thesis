from random import random
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax

key = jax.random.PRNGKey(3)

SAMPLES = 100000
DIMENSION = 500

key, normal_noise_key, transform_key = jax.random.split(key, 3)
transformation = jax.random.normal(transform_key, (DIMENSION, DIMENSION)) * 8
# transformation = jnp.array([[1,0], [0,1]])
# transformation = jnp.identity(DIMENSION)
transformation = transformation.at[0,1].add(50)
normal_noise = jax.random.normal(normal_noise_key, (SAMPLES, DIMENSION))[...,jnp.newaxis]
# transformed_noise = jnp.einsum('ik,jk', transformation, normal_noise)
transformed_noise = transformation @ normal_noise
norm = jnp.linalg.norm(transformed_noise.reshape((SAMPLES,-1)), axis=1)**2
plt.hist(norm, bins=400)
plt.show()
