import tensorflow as tf
import numpy as np
import sys
from rbm_utils import sample_bernoulli, sample_gaussian, trans


class ARBM:
    def __init__(
        self,
        n_visible,
        n_hidden,
        n_adaptive,
        sample_visible=False,
        sigma=1,
        learning_rate=0.01,
        momentum=0.95,
        cdk_level=1,
    ):
        """
        Creates a new Gaussian-Bernoulli Restricted Boltzmann Machine.

        Args:
            n_visible: The number of visible units.
            n_hidden: The number of hidden units.
            sample_visible: True is the reconstructed visible units should be sampled during each
                Gibbs step. False otherwise. Defaults to False.
            sigma: The standard deviation the gaussian visible units have. Defaults to 1.
            learning_rate: The learning rate applied on the gradient of the weights and biases.
            momentum: The momentum applied on the gradient of the weights and biases.
            cdk_level: The number of Gibbs steps to be used by the contrastive divergence algorithm.
        """
        if not 0.0 <= momentum < 1.0:
            raise ValueError('momentum should be in range [0, 1)')
        assert cdk_level > 0
        assert learning_rate > 0

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_adaptive = n_adaptive
        self.sample_visible = sample_visible
        self.sigma = sigma
        self.epsilon = learning_rate
        self.momentum = momentum
        self.cdk_level = cdk_level

        # initialise tensorflow variables
        self.input = None
        self.label = None
        self.hidden = None

        self.bar_weights = None
        self.bar_v_bias = None
        self.bar_h_bias = None

        self.adaptive_weights = None
        self.adaptive_v_bias = None
        self.adaptive_h_bias = None

        self.delta_bar_weights = None
        self.delta_bar_v_bias = None
        self.delta_bar_h_bias = None

        self.delta_adaptive_weights = None
        self.delta_adaptive_v_bias = None
        self.delta_adaptive_h_bias = None

        self._initialise_variables()

        assert self.input is not None
        assert self.label is not None
        assert self.hidden is not None

        assert self.bar_weights is not None
        assert self.bar_v_bias is not None
        assert self.bar_h_bias is not None

        assert self.adaptive_weights is not None
        assert self.adaptive_v_bias is not None
        assert self.adaptive_h_bias is not None

        assert self.delta_bar_weights is not None
        assert self.delta_bar_v_bias is not None
        assert self.delta_bar_h_bias is not None

        assert self.delta_adaptive_weights is not None
        assert self.delta_adaptive_v_bias is not None
        assert self.delta_adaptive_h_bias is not None

        # initialise tensorflow computational variables
        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.reconstruct_visible = None
        self.compute_error = None

        self._initialise_calculations()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.reconstruct_visible is not None
        assert self.compute_error is not None

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialise_variables(self):
        # unit layers of the machine
        self.input = tf.placeholder(tf.float32, [None, self.n_visible])
        self.label = tf.placeholder(tf.float32, [self.n_adaptive])
        self.hidden = tf.placeholder(tf.float32, [None, self.n_hidden])

        # independent weights and biases
        self.bar_weights = tf.Variable(
            tf.truncated_normal(
                [self.n_visible, self.n_hidden],
                stddev=0.1),
            dtype=tf.float32,
        )
        self.bar_v_bias = tf.Variable(
            tf.zeros([self.n_visible]),
            dtype=tf.float32,
        )
        self.bar_h_bias = tf.Variable(
            tf.zeros([self.n_hidden]),
            dtype=tf.float32,
        )

        # adaptive weights and biases
        self.adaptive_weights = tf.Variable(
            tf.truncated_normal(
                [self.n_adaptive, self.n_visible, self.n_visible],
                stddev=0.1),
            dtype=tf.float32,
        )
        self.adaptive_v_bias = tf.Variable(
            tf.zeros([self.n_adaptive, self.n_visible]),
            dtype=tf.float32,
        )
        self.adaptive_h_bias = tf.Variable(
            tf.zeros([self.n_adaptive, self.n_hidden]),
            dtype=tf.float32,
        )

        # gradient steps for independent variables
        self.delta_bar_weights = tf.Variable(
            tf.zeros([self.n_visible, self.n_hidden]),
            dtype=tf.float32,
        )
        self.delta_bar_v_bias = tf.Variable(
            tf.zeros([self.n_visible]),
            dtype=tf.float32,
        )
        self.delta_bar_h_bias = tf.Variable(
            tf.zeros([self.n_hidden]),
            dtype=tf.float32,
        )

        # gradient steps for adaptive variables
        self.delta_adaptive_weights = tf.Variable(
            tf.zeros([self.n_adaptive, self.n_visible, self.n_visible]),
            dtype=tf.float32,
        )
        self.delta_adaptive_v_bias = tf.Variable(
            tf.zeros([self.n_adaptive, self.n_visible]),
            dtype=tf.float32,
        )
        self.delta_adaptive_h_bias = tf.Variable(
            tf.zeros([self.n_adaptive, self.n_hidden]),
            dtype=tf.float32,
        )

    def _initialise_calculations(self):
        weights, v_bias, h_bias = self._compute_weights()
        (
            bar_weights_gradient,
            bar_v_bias_gradient,
            bar_h_bias_gradient,
            adaptive_weights_gradient,
            adaptive_v_bias_gradient,
            adaptive_h_bias_gradient,
        ) = self._compute_gradients(weights, v_bias, h_bias)

        # the momentum method for updating parameters
        def f(x_old, x_new):
            return self.momentum * x_old + \
                   self.epsilon * x_new * (1 - self.momentum) / (tf.to_float(tf.shape(x_new)[0])*100000)

        delta_bar_weights_new = f(self.delta_bar_weights, bar_weights_gradient)
        delta_bar_v_bias_new = f(self.delta_bar_v_bias, bar_v_bias_gradient)
        delta_bar_h_bias_new = f(self.delta_bar_h_bias, bar_h_bias_gradient)
        delta_adaptive_weights_new = f(self.delta_adaptive_weights, adaptive_weights_gradient)
        delta_adaptive_v_bias_new = f(self.delta_adaptive_v_bias, adaptive_v_bias_gradient)
        delta_adaptive_h_bias_new = f(self.delta_adaptive_h_bias, adaptive_h_bias_gradient)

        update_delta_bar_weights = self.delta_bar_weights.assign(delta_bar_weights_new)
        update_delta_bar_v_bias = self.delta_bar_v_bias.assign(delta_bar_v_bias_new)
        update_delta_bar_h_bias = self.delta_bar_h_bias.assign(delta_bar_h_bias_new)
        update_delta_adaptive_weights = self.delta_adaptive_weights.assign(delta_adaptive_weights_new)
        update_delta_adaptive_v_bias = self.delta_adaptive_v_bias.assign(delta_adaptive_v_bias_new)
        update_delta_adaptive_h_bias = self.delta_adaptive_h_bias.assign(delta_adaptive_h_bias_new)

        update_bar_weights = self.bar_weights.assign_add(delta_bar_weights_new)
        update_bar_v_bias = self.bar_v_bias.assign_add(delta_bar_v_bias_new)
        update_bar_h_bias = self.bar_h_bias.assign_add(delta_bar_h_bias_new)
        update_adaptive_weights = self.adaptive_weights.assign_add(delta_adaptive_weights_new)
        update_adaptive_v_bias = self.adaptive_v_bias.assign_add(delta_adaptive_v_bias_new)
        update_adaptive_h_bias = self.adaptive_h_bias.assign_add(delta_adaptive_h_bias_new)

        # tensorflow computations
        self.update_deltas = [
            update_delta_bar_weights,
            update_delta_bar_v_bias,
            update_delta_bar_h_bias,
            update_delta_adaptive_weights,
            update_delta_adaptive_v_bias,
            update_delta_adaptive_h_bias,
        ]
        self.update_weights = [
            update_bar_weights,
            update_bar_v_bias,
            update_bar_h_bias,
            update_adaptive_weights,
            update_adaptive_v_bias,
            update_adaptive_h_bias,
        ]

        self.compute_hidden = self._compute_hidden(self.input, h_bias, weights)
        self.compute_visible = self._compute_visible(self.hidden, v_bias, weights)
        self.reconstruct_visible = self._compute_visible(self.compute_hidden, v_bias, weights)
        self.compute_error = tf.reduce_mean(tf.square(self.input - self.reconstruct_visible))

    def _compute_weights(self):
        weights = tf.tensordot(
            self.adaptive_weights,
            self.label,
            axes=[[0], [0]],
        ) @ self.bar_weights

        v_bias = tf.tensordot(
            self.adaptive_v_bias,
            self.label,
            axes=[[0], [0]],
        ) + self.bar_v_bias

        h_bias = tf.tensordot(
            self.adaptive_h_bias,
            self.label,
            axes=[[0], [0]],
        ) + self.bar_h_bias

        return weights, v_bias, h_bias

    def _compute_hidden(self, v_layer, h_bias, weights):
        return (tf.nn.sigmoid((v_layer // self.sigma**2) @ weights + h_bias)) * 3.0

    def _compute_visible(self, h_layer, v_bias, weights):
        v = (sample_bernoulli(h_layer) @ trans(weights) + v_bias) / 3.0
        if self.sample_visible:
            v = sample_gaussian(v, self.sigma)
        return v

    def _compute_gradients(self, weights, v_bias, h_bias):
        v_sample, h_sample = self._gibbs_iter(weights, v_bias, h_bias)
        v0_sample = self.input
        h0_sample = self._compute_hidden(v0_sample, h_bias, weights)
        v_sample = v_sample / self.sigma**2
        v0_sample = v0_sample / self.sigma**2

        bar_weights_gradient = trans(v0_sample @ tf.tensordot(
            self.adaptive_weights,
            self.label,
            axes=[[0], [0]],
        )) @ h0_sample - trans(v_sample @ tf.tensordot(
            self.adaptive_weights,
            self.label,
            axes=[[0], [0]],
        )) @ h_sample

        bar_v_bias_gradient = tf.reduce_sum(v0_sample - v_sample, 0)
        bar_h_bias_gradient = tf.reduce_sum(h0_sample - h_sample, 0)

        index = tf.squeeze(tf.where(tf.not_equal(self.label, tf.constant(0, dtype=tf.float32))))

        adaptive_weights_gradient = tf.scatter_add(
            tf.Variable(tf.zeros([self.n_adaptive, self.n_visible, self.n_visible], dtype=tf.float32)),
            index,
            trans(self.bar_weights @ (trans(h0_sample) @ v0_sample - trans(h_sample) @ v_sample)),
        )
        adaptive_v_bias_gradient = tf.scatter_add(
            tf.Variable(tf.zeros([self.n_adaptive, self.n_visible], dtype=tf.float32)),
            index,
            tf.reduce_sum(v0_sample - v_sample, 0),
        )
        adaptive_h_bias_gradient = tf.scatter_add(
            tf.Variable(tf.zeros([self.n_adaptive, self.n_hidden], dtype=tf.float32)),
            index,
            tf.reduce_sum(h0_sample - h_sample, 0),
        )

        return (
            bar_weights_gradient,
            bar_v_bias_gradient,
            bar_h_bias_gradient,
            adaptive_weights_gradient,
            adaptive_v_bias_gradient,
            adaptive_h_bias_gradient,
        )

    def _gibbs_iter(self, weights, v_bias, h_bias):
        v_sample = self.input
        h_sample = self._compute_hidden(v_sample, h_bias, weights)

        # the Gibbs steps
        for i in range(self.cdk_level):
            v_sample = self._compute_visible(h_sample, v_bias, weights)
            h_sample = self._compute_hidden(v_sample, h_bias, weights)

        return v_sample, h_sample

    def _get_error(
            self,
            batch,
            batch_label,
    ):
        return self.sess.run(
            self.compute_error,
            feed_dict={self.input: batch, self.label: batch_label},
        )

    def _partial_fit(
            self,
            batch,
            batch_label,
    ):
        self.sess.run(
            self.update_weights + self.update_deltas,
            feed_dict={self.input: batch, self.label: batch_label},
        )

    def fit(
        self,
        data,
        n_epochs=30,
        batch_size=100,
        shuffle=True,
        verbose=True,
    ):
        """
        Fits the given data into the model of the RBM.

        Args:
            data: The training data to be used for learning
            n_epochs: The number of epochs. Defaults to 10.
            batch_size: The size of the data batch per epoch. Defaults to 10.
            shuffle: True if the data should be shuffled before learning.
                False otherwise. Defaults to True.
            verbose: True if the progress should be displayed on
                the standard output during training.

        Returns: An array of the mean square errors of each batch.

        """
        assert n_epochs > 0
        data_cpy = {}
        for label in data:
            data_cpy[label] = np.copy(np.transpose(data[label]))

        errs = []

        for e in range(n_epochs):
            if verbose:
                print('Epoch: {:d}'.format(e + 1))

            epoch_errs = np.array([])

            for label, features in data_cpy.items():
                if shuffle:
                    np.random.shuffle(features)

                onehot = np.zeros([self.n_adaptive], dtype=float)
                onehot[label] = 1

                for batch_nr in range(batch_size, features.shape[0], batch_size):
                    batch = features[batch_nr - batch_size:batch_nr]
                    self._partial_fit(batch, onehot)
                    batch_err = self._get_error(batch, onehot)
                    assert np.isnan(batch_err).any() == False
                    epoch_errs = np.append(epoch_errs, batch_err)

                if verbose:
                    err_mean = epoch_errs.mean()
                    print('Train error: {:.4f}'.format(err_mean))
                    print()
                    sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])

        return errs

    def reconstruct(
            self,
            speaker_label,
            speaker_data,
    ):
        onehot = np.zeros([self.n_adaptive], dtype=float)
        onehot[speaker_label] = 1
        return np.transpose(self.sess.run(
            self.reconstruct_visible,
            feed_dict={self.input: np.transpose(speaker_data), self.label: onehot},
        ))

    def convert(
            self,
            source_label,
            source_data,
            target_label,
    ):

        onehot = np.zeros([self.n_adaptive], dtype=float)
        onehot[source_label] = 1
        hidden = self.sess.run(
            self.compute_hidden,
            feed_dict={self.input: np.transpose(source_data), self.label: onehot},
        )
        onehot[source_label] = 0
        onehot[target_label] = 1
        conversion = self.sess.run(
            self.compute_visible,
            feed_dict={self.hidden: hidden, self.label: onehot},
        )
        return np.transpose(conversion)

    def add_speaker(
            self,
            speaker_data,
    ):
        # return
        pass

    def save(
            self,
            filename,
    ):
        pass

    def load(
            self,
            filename,
    ):
        pass

    @staticmethod
    def save_model(filename):
        pass

    @staticmethod
    def load_model(filename):
        pass
