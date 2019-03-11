import tensorflow as tf
import numpy as np
import sys
from rbm_utils import sample_softmax, sample_gaussian, trans


class ARBM:
    def __init__(
        self,
        n_visible,
        n_hidden,
        n_adaptive,
        sample_visible=False,
        sigma=1,
        learning_rate=0.001,
        momentum=0,
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

        self._index = 0
        self._source = 0
        self._target = 0

        # initialise tensorflow variables
        self.input = None

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
        self.reconstruct_visible = None
        self.compute_conversion = None
        self.compute_error = None

        self._initialise_calculations()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.reconstruct_visible is not None
        assert self.compute_conversion is not None
        assert self.compute_error is not None

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialise_variables(self):
        # unit layers of the machine
        self.input = tf.placeholder(tf.float32, [None, self.n_visible])

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
        weights, v_bias, h_bias = self._compute_weights(self._index)
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
            # I still don't understand why do I have to do that division at the end...
            return self.momentum * x_old + \
                   self.epsilon * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

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
        self.reconstruct_visible = self._compute_visible(self.compute_hidden, v_bias, weights)
        self.compute_conversion = self._compute_conversion()
        self.compute_error = tf.reduce_mean(tf.square(self.input - self.reconstruct_visible))

    def _compute_weights(self, index):
        weights = self.adaptive_weights[index] @ self.bar_weights
        v_bias = self.adaptive_v_bias[index] + self.bar_v_bias
        h_bias = self.adaptive_h_bias[index] + self.bar_h_bias
        return weights, v_bias, h_bias

    def _compute_hidden(self, v_layer, h_bias, weights):
        return sample_softmax(tf.nn.softmax(v_layer @ weights + h_bias))

    def _compute_visible(self, h_layer, v_bias, weights):
        v = h_layer @ trans(weights) + v_bias
        if self.sample_visible:
            v = sample_gaussian(v, self.sigma)
        return v

    def _compute_gradients(self, weights, v_bias, h_bias):
        v_sample, h_sample = self._gibbs_iter(weights, v_bias, h_bias)
        v0_sample = self.input
        h0_sample = self._compute_hidden(v0_sample, h_bias, weights)
        v_sample = v_sample / self.sigma**2
        v0_sample = v0_sample / self.sigma**2

        bar_weights_gradient = trans(v0_sample @ self.adaptive_weights[self._index]) @ h0_sample -\
                               trans(v_sample @ self.adaptive_weights[self._index]) @ h_sample
        bar_v_bias_gradient = tf.reduce_sum(v0_sample - v_sample, 0)
        bar_h_bias_gradient = tf.reduce_sum(h0_sample - h_sample, 0)
        adaptive_weights_gradient = tf.scatter_add(
            tf.Variable(tf.zeros([self.n_adaptive, self.n_visible, self.n_visible], dtype=tf.float32)),
            self._index,
            trans(self.bar_weights @ (trans(h0_sample) @ v0_sample - trans(h_sample) @ v_sample)),
        )
        adaptive_v_bias_gradient = tf.scatter_add(
            tf.Variable(tf.zeros([self.n_adaptive, self.n_visible], dtype=tf.float32)),
            self._index,
            tf.reduce_sum(v0_sample - v_sample, 0),
        )
        adaptive_h_bias_gradient = tf.scatter_add(
            tf.Variable(tf.zeros([self.n_adaptive, self.n_hidden], dtype=tf.float32)),
            self._index,
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

    def _compute_conversion(self):
        forward_weights, _, forward_h_bias = self._compute_weights(self._source)
        backward_weights, backward_v_bias, _ = self._compute_weights(self._target)
        h_sample = self._compute_hidden(self.input, forward_h_bias, forward_weights)
        v_sample = self._compute_visible(h_sample, backward_v_bias, backward_weights)
        return v_sample

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
        self._index = batch_label
        return self.sess.run(self.compute_error, feed_dict={self.input: batch})

    def _partial_fit(
            self,
            batch,
            batch_label,
    ):
        self._index = batch_label
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.input: batch})

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

                for batch_nr in range(batch_size, features.shape[0], batch_size):
                    batch = features[batch_nr - batch_size:batch_nr]
                    self._partial_fit(batch, label)
                    batch_err = self._get_error(batch, label)
                    print(batch_err)
                    assert np.isnan(batch_err) is False
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
        self._index = speaker_label
        return np.transpose(self.sess.run(
            self.reconstruct_visible,
            feed_dict={self.input: np.transpose(speaker_data)},
        ))

    def convert(
            self,
            source_label,
            source_data,
            target_label,
    ):
        self._source = source_label
        self._target = target_label
        return np.transpose(self.sess.run(
            self.compute_conversion,
            feed_dict={self.input: np.transpose(source_data)},
        ))

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
