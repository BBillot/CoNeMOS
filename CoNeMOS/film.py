import keras
from keras.layers import Layer, Dense
from ext.lab2im.edit_tensors import expand_dims


class FiLM(Layer):
    """FiLM layers adapted from Perez et al., 2017"""

    def __init__(self, n_dims, wt_decay=1e-5, **kwargs):
        """
        :param n_dims: dimension of the inputs (2D/3D)
        :param wt_decay: L2 penalty on FiLM projection.
        """

        self.n_dims = n_dims
        self.wt_decay = wt_decay
        self.channels = None
        self.fc = None
        super(FiLM, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["n_dims"] = self.n_dims
        config["wt_decay"] = self.wt_decay
        return config

    def build(self, input_shape):
        self.channels = input_shape[0][-1]  # input_shape: [x, z].
        self.fc = Dense(int(2 * self.channels),
                        kernel_regularizer=L2Regularizer(l2=self.wt_decay),
                        bias_regularizer=L2Regularizer(l2=self.wt_decay),
                        name=self.name + '_dense')
        self.built = True
        super(FiLM, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x, z = inputs
        z = self.fc(z)
        gamma = z[..., :self.channels]
        beta = z[..., self.channels:]
        return (1. + expand_dims(gamma, [1] * self.n_dims)) * x + expand_dims(beta, [1] * self.n_dims)


class L2Regularizer(keras.regularizers.Regularizer):
    """re-implement this here for backwards compatibility with earlier versions of Keras"""

    def __init__(self, l2):
        self.l2 = l2

    def __call__(self, x):
        return self.l2 * keras.backend.sum(keras.backend.square(x))

    def get_config(self):
        return {'l2': self.l2}
