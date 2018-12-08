import tensorflow as tf
import numpy as np
import sys
from rbm_utils import sample_bernoulli, sample_gaussian


class RBM:
    def __init__(
        self,
        n_visible,
        n_hidden,
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
            sample_visible: True is the reconstructed visible units should be sampled during each Gibbs step.
                False otherwise. Defaults to False.
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
        self.sample_visible = sample_visible
        self.sigma = sigma
        self.epsilon = learning_rate
        self.momentum = momentum
        self.cdk_level = cdk_level

        self.v_layer = tf.placeholder(tf.float32, [None, self.n_visible])
        self.h_layer = tf.placeholder(tf.float32, [None, self.n_hidden])

        self.w = tf.Variable(tf.truncated_normal([n_visible, n_hidden], stddev=0.1), dtype=tf.float32)
        self.vb = tf.Variable(tf.zeros([n_visible]), dtype=tf.float32)
        self.hb = tf.Variable(tf.zeros([n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([n_visible, n_hidden]), dtype=tf.float32)
        self.delta_vb = tf.Variable(tf.zeros([n_visible]), dtype=tf.float32)
        self.delta_hb = tf.Variable(tf.zeros([n_hidden]), dtype=tf.float32)

        self.update_weights = None
        self.update_deltas = None
        self.compute_hidden = None
        self.compute_visible = None
        self.reconstruct_visible = None

        self._initialize_vars()

        assert self.update_weights is not None
        assert self.update_deltas is not None
        assert self.compute_hidden is not None
        assert self.compute_visible is not None
        assert self.reconstruct_visible is not None

        self.compute_err = tf.reduce_mean(tf.square(self.v_layer - self.reconstruct_visible))

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_vars(self):
        v_sample = self.v_layer
        h_sample = sample_bernoulli(tf.nn.sigmoid(tf.matmul(v_sample, self.w) + self.hb))

        # the Gibbs steps
        for i in range(self.cdk_level):
            v_sample = tf.matmul(h_sample, tf.transpose(self.w)) + self.vb
            if self.sample_visible:
                v_sample = sample_gaussian(v_sample, self.sigma)
            h_sample = sample_bernoulli(tf.nn.sigmoid(tf.matmul(v_sample, self.w) + self.hb))

        # recalculate the first value of the hidden units and compute the positive and negative gradients.
        h0_sample = sample_bernoulli(tf.nn.sigmoid(tf.matmul(self.v_layer, self.w) + self.hb))
        positive_grad = tf.matmul(tf.transpose(self.v_layer), h0_sample)
        negative_grad = tf.matmul(tf.transpose(v_sample), h_sample)

        # the momentum method for updating parameters
        def f(x_old, x_new):
            # I still don't understand why do I have to do that division at the end...
            return self.momentum * x_old + self.epsilon * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

        delta_w_new = f(self.delta_w, positive_grad - negative_grad)
        delta_vb_new = f(self.delta_vb, tf.reduce_mean(self.v_layer - v_sample, 0))
        delta_hb_new = f(self.delta_hb, tf.reduce_mean(h0_sample - h_sample, 0))

        update_delta_w = self.delta_w.assign(delta_w_new)
        update_delta_vb = self.delta_vb.assign(delta_vb_new)
        update_delta_hb = self.delta_hb.assign(delta_hb_new)

        update_w = self.w.assign_add(delta_w_new)
        update_vb = self.vb.assign_add(delta_vb_new)
        update_hb = self.hb.assign_add(delta_hb_new)

        # tensorflow computations
        self.update_deltas = [update_delta_w, update_delta_vb, update_delta_hb]
        self.update_weights = [update_w, update_vb, update_hb]

        self.compute_hidden = tf.nn.sigmoid(tf.matmul(self.v_layer, self.w) + self.hb)
        self.compute_visible = tf.matmul(self.h_layer, tf.transpose(self.w)) + self.vb
        self.reconstruct_visible = tf.matmul(self.compute_hidden, tf.transpose(self.w)) + self.vb

    def _get_err(self, batch):
        return self.sess.run(self.compute_err, feed_dict={self.v_layer: batch})

    def _partial_fit(self, batch):
        self.sess.run(self.update_weights + self.update_deltas, feed_dict={self.v_layer: batch})

    def fit(
        self,
        data,
        n_epochs=10,
        batch_size=10,
        shuffle=True,
        verbose=True,
    ):
        """
        Fits the given data into the model of the RBM.

        Args:
            data: The training data to be used for learning
            n_epochs: The number of epochs. Defaults to 10.
            batch_size: The size of the data batch per epoch. Defaults to 10.
            shuffle: True if the data should be shuffled before learning. False otherwise. Defaults to True.
            verbose: True if the progress should be displayed on the standard output during training.

        Returns: An array of the mean square errors of each batch.

        """
        assert n_epochs > 0

        n_data = data.shape[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        data_cpy = data.copy()

        errs = []

        for e in range(n_epochs):
            if verbose:
                print('Epoch: {:d}'.format(e + 1))

            epoch_errs = np.array([])

            if shuffle:
                np.random.shuffle(data_cpy)

            for batch_nr in range(n_batches):
                batch = data_cpy[batch_nr * batch_size:(batch_nr + 1) * batch_size]
                self._partial_fit(batch)
                batch_err = self._get_err(batch)
                epoch_errs = np.append(epoch_errs, batch_err)

            if verbose:
                err_mean = epoch_errs.mean()
                print('Train error: {:.4f}'.format(err_mean))
                print()
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])

        return errs

    def transform(self, batch):
        """
        Computes the values of the hidden units given the visible units.

        Args:
            batch: The values of the visible units. Multiple input vectors can be given at once.

        Returns: The values of the hidden units. Returns one output vector for each input vector in the batch.

        """
        return self.sess.run(self.compute_hidden, feed_dict={self.v_layer: batch})

    def transform_inv(self, batch):
        """
        Computes the values of the visible units given the hidden units.

        Args:
            batch: The values of the hidden units. Multiple input vectors can be given at once.

        Returns: The values of the visible units. Returns one output vector for each input vector in the batch.

        """
        return self.sess.run(self.compute_visible, feed_dict={self.h_layer: batch})

    def reconstruct(self, batch):
        """
        Reconstructs the given data by doing one Gibbs step with it as the original visible layer.

        Args:
            batch: The data to be reconstructed. Multiple input vectors can be given at once.

        Returns: The reconstructed data. Returns one output vector for each input vector in the batch.

        """
        return self.sess.run(self.reconstruct_visible, feed_dict={self.v_layer: batch})

    def get_weights(self):
        """
        Gets the weights matrix and bias vectors currently in the model.

        Returns: A tuple (weights matrix, visible units biases, hidden units biases).

        """
        return self.sess.run(self.w), self.sess.run(self.vb), self.sess.run(self.hb)

    def save_weights(self, filename, name):
        """
        Saves the current weights and biases in the specified file.

        Args:
            filename: The file in which to save the model.
            name: The name of the model.

        Returns: The output of tf.train.Saver.save()

        """
        saver = tf.train.Saver({
            name + '_w': self.w,
            name + '_v': self.vb,
            name + '_h': self.hb,
        })
        return saver.save(self.sess, filename)

    def set_weights(self, w, visible_bias, hidden_bias):
        """
        Sets the weights matrix and bias vectors to the given values.
        Args:
            w: The matrix to which the weights to be set.
            visible_bias: The vector to which the biases of the visible units to be set.
            hidden_bias: The vector to which the biases of the hidden units to be set.

        """
        self.sess.run(self.w.assign(w))
        self.sess.run(self.vb.assign(visible_bias))
        self.sess.run(self.hb.assign(hidden_bias))

    def load_weights(self, filename, name):
        """
        Assigns to the weights matrix and bias vectors the values found in the specified file under the specified name.
        Args:
            filename: The file where the model is stored.
            name: The name of the model.

        """
        saver = tf.train.Saver({
            name + '_w': self.w,
            name + '_v': self.vb,
            name + '_h': self.hb,
        })
        saver.restore(self.sess, filename)
