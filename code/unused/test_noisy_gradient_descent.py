import jax.numpy as jnp
from stochastic_variance_amplified_gradient import *
from noisy_gradient_descent import *
from absl.testing import absltest
import chex


# Take the square root in this example!!
class NoisyGradientDescentTest(absltest.TestCase):

    def test_one_sample_covariance(self):
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
                jnp.square(mean_gradient_w - 2.0 *
                           (w * x + b - y) * x)) / points
            expected_covariance_b = jnp.sum(
                jnp.square(mean_gradient_b - 2.0 * (w * x + b - y))) / points

            parameters = [(w * jnp.ones((1, 1)), b * jnp.ones((1, 1)))]
            covariance_w, covariance_b = diagonal_one_sample_covariance(
                parameters, x, y)[0]
            self.assert_trees_all_close(expected_covariance_w, covariance_w)
            self.assert_trees_all_close(expected_covariance_b, covariance_b)

    def right_hand_side(parameters, t, coefficients_0, coefficients_1,
                        coefficients_2, coefficients_3, coefficients_4,
                        coefficients_5):
        w, b = parameters
        dparametersdt = [
            coefficients_0 * w + coefficients_1 * b + coefficients_2,
            coefficients_3 * w + coefficients_4 * b + coefficients_5
        ]
        return dparametersdt

    def test_square_root_matrix(self):
        key = random.PRNGKey(1)
        n = 100
        x = random.normal(key, (2, n))
        mean = jnp.mean(x, axis=1)
        x_normalized = x - mean.reshape((2, 1))
        sqrt_matrix = sqrt_matrix(x_normalized)
        self.assert_trees_all_close(sqrt_matrix @ sqrt_matrix.T,
                                    x @ x.T,
                                    rtol=1e-3)

    def test_map_parameters_to_vector(self):
        key = random.PRNGKey(1)
        sizes = [1, 10, 15, 20, 1]
        parameters = initialize_network_parameters(sizes, key)
        flattend_parameters = [
            weight_bias for parameter in parameters
            for weight_bias in parameter
        ]
        stacked_parameters = jnp.vstack(
            tuple(
                parameter.reshape((-1, 1))
                for parameter in flattend_parameters))
        unstacked_parameters = unstack_parameters(sizes, stacked_parameters)
        for (w, b), (expected_w, expected_b) in zip(unstacked_parameters,
                                                    parameters):
            self.assert_trees_all_close(w, expected_w)
            self.assert_trees_all_close(b, expected_b)


if __name__ == '__main__':
    absltest.main()
