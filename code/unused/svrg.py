from keras.datasets import mnist
import tensorflow as tf


class SVRG(tf.keras.optimizers.Optimizer):

    def __init__(self, learning_rate=0.01, name="SVRG", **kwargs):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate",
                        kwargs.get("lr",
                                   learning_rate))  # handle lr=learning_rate
        self._is_first = True

    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "pv")  #previous variable i.e. weight or bias
        for var in var_list:
            self.add_slot(var, "pg")  #previous gradient

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable
        """
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)  # handle learning rate decay
        new_var_m = var - grad * lr_t
        mu_var = self.get_slot(var, "mu")

        if self._is_first:
            self._is_first = False
            new_var = new_var_m
        else:
            cond = grad * pg_var >= 0
            print(cond)
            avg_weights = (pv_var + var) / 2.0
            new_var = tf.where(cond, new_var_m, avg_weights)
        pv_var.assign(var)
        pg_var.assign(grad)
        var.assign(new_var)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate":
            self._serialize_hyperparameter("learning_rate"),
        }

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate":
            self._serialize_hyperparameter("learning_rate"),
            "decay":
            self._serialize_hyperparameter("decay"),
            "momentum":
            self._serialize_hyperparameter("momentum"),
        }


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5)

    print(model)


if __name__ == "__main__":
    main()
