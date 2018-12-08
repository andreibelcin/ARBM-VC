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
        self.n_adaptive = n_adaptive
        self.sample_visible = sample_visible
        self.sigma = sigma
        self.epsilon = learning_rate
        self.momentum = momentum
        self.cdk_level = cdk_level

        # unit layers of the machine
        self.v_layer = tf.placeholder(tf.float32, [None, self.n_visible])
        self.h_layer = tf.placeholder(tf.float32, [None, self.n_hidden])
        self.a_layer = tf.placeholder(tf.float32, [None, self.n_adaptive])

        # independent weights and biases
        self.w_bar = tf.Variable(tf.truncated_normal([n_visible, n_hidden], stddev=0.1), dtype=tf.float32)
        self.vb_bar = tf.Variable(tf.zeros([n_visible]), dtype=tf.float32)
        self.hb_bar = tf.Variable(tf.zeros([n_hidden]), dtype=tf.float32)

        # adaptive weights and biases
        self.aw = tf.Variable(tf.truncated_normal(
            [n_adaptive, n_visible, n_visible],
            stddev=0.1), dtype=tf.float32)
        self.avb = tf.Variable(tf.zeros([n_adaptive, n_visible]), dtype=tf.float32)
        self.ahb = tf.Variable(tf.zeros([n_adaptive, n_hidden]), dtype=tf.float32)

        # gradient steps for independent variables
        self.delta_w_bar = tf.Variable(tf.zeros([n_visible, n_hidden]), dtype=tf.float32)
        self.delta_vb_bar = tf.Variable(tf.zeros([n_visible]), dtype=tf.float32)
        self.delta_hb_bar = tf.Variable(tf.zeros([n_hidden]), dtype=tf.float32)

        # gradient steps for adaptive variables
        self.delta_aw = tf.Variable(tf.zeros([n_adaptive, n_visible, n_visible]), dtype=tf.float32)
        self.delta_avb = tf.Variable(tf.zeros([n_adaptive, n_visible]), dtype=tf.float32)
        self.delta_ahb = tf.Variable(tf.zeros([n_adaptive, n_hidden]), dtype=tf.float32)

        # tensorflow computational variables
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

    def _h_prob(self, v_sample, hb, w):
        return tf.nn.softmax(tf.squeeze(tf.reshape(v_sample, shape=[-1, 1, self.n_visible]) @ w) + hb)

    def _v_prob(self, h_sample, vb, w):
        return tf.squeeze(tf.reshape(h_sample, shape=[-1, 1, self.n_hidden]) @ trans(w, perm=[0, 2, 1])) + vb

    def _compute_hidden(self, v, hb, w):
        return sample_softmax(self._h_prob(v, hb, w))

    def _compute_visible(self, h, vb, w):
        v = self._v_prob(h, vb, w)
        if self.sample_visible:
            v = sample_gaussian(v, self.sigma)
        return v

    def _compute_weights(self):
        w = tf.reshape(
            tf.reshape(
                tf.tensordot(
                    self.a_layer, self.aw,
                    axes=[[1], [0]],
                ),
                shape=[-1, self.n_visible],
            ) @ self.w_bar,
            shape=[-1, self.n_visible, self.n_hidden],
        )
        vb = self.a_layer @ self.avb + self.vb_bar
        hb = self.a_layer @ self.ahb + self.hb_bar
        return w, vb, hb

    def _gibbs_iter(self):
        w, vb, hb = self._compute_weights()
        v_sample = self.v_layer
        h_sample = h0_sample = self._compute_hidden(v_sample, hb, w)

        # the Gibbs steps
        for i in range(self.cdk_level):
            v_sample = self._compute_visible(h_sample, vb, w)
            h_sample = self._compute_hidden(v_sample, hb, w)

        return v_sample, h_sample, h0_sample

    def _compute_gradients(self):
        v_sample, h_sample, h0_sample = self._gibbs_iter()
        v0_sample = self.v_layer

        vb_bar_grad = tf.reduce_sum(v0_sample - v_sample, axis=[0])
        avb_grad = trans(self.a_layer) @ v0_sample - trans(self.a_layer) @ v_sample
        hb_bar_grad = tf.reduce_sum(h0_sample - h_sample, axis=[0])
        ahb_grad = trans(self.a_layer) @ h0_sample - trans(self.a_layer) @ h_sample

        adaptive_matrix = tf.reshape(
            tf.tensordot(
                tf.reshape(self.a_layer, [-1, 1, self.n_adaptive]),
                trans(self.aw),
                axes=[[2], [2]],
            ),
            shape=[-1, self.n_visible, self.n_visible],
        )
        v_sample = tf.reshape(v_sample, shape=[-1, self.n_visible, 1])
        v0_sample = tf.reshape(v0_sample, shape=[-1, self.n_visible, 1])
        h_sample = tf.reshape(h_sample, shape=[-1, 1, self.n_hidden])
        h0_sample = tf.reshape(h0_sample, shape=[-1, 1, self.n_hidden])

        w_bar_grad = tf.squeeze(
            tf.tensordot(
                adaptive_matrix @ v0_sample,
                h0_sample,
                axes=[[0], [0]],
            ) - tf.tensordot(
                adaptive_matrix @ v_sample,
                h_sample,
                axes=[[0], [0]],
            )
        )
        aw_grad = tf.tensordot(
            self.a_layer,
            tf.reshape(
                tf.reshape(
                    v0_sample @ h0_sample,
                    shape=[-1, self.n_hidden]
                ) @ trans(self.w_bar),
                shape=[-1, self.n_visible, self.n_visible],
            ),
            axes=[[0], [0]],
        ) - tf.tensordot(
            self.a_layer,
            tf.reshape(
                tf.reshape(
                    v_sample @ h_sample,
                    shape=[-1, self.n_hidden]
                ) @ trans(self.w_bar),
                shape=[-1, self.n_visible, self.n_visible],
            ),
            axes=[[0], [0]],
        )

        return (
            w_bar_grad,
            aw_grad,
            vb_bar_grad,
            avb_grad,
            hb_bar_grad,
            ahb_grad,
        )

    def _initialize_vars(self):
        w_bar_grad, aw_grad, vb_bar_grad, avb_grad, hb_bar_grad, ahb_grad = self._compute_gradients()

        # the momentum method for updating parameters
        def f(x_old, x_new):
            # I still don't understand why do I have to do that division at the end...
            return self.momentum * x_old + self.epsilon * x_new * (1 - self.momentum) / tf.to_float(tf.shape(x_new)[0])

        delta_w_bar_new = f(self.delta_w_bar, w_bar_grad)
        delta_aw_new = f(self.delta_aw, aw_grad)
        delta_vb_bar_new = f(self.delta_vb_bar, vb_bar_grad)
        delta_avb_new = f(self.delta_avb, avb_grad)
        delta_hb_bar_new = f(self.delta_hb_bar, hb_bar_grad)
        delta_ahb_new = f(self.delta_ahb, ahb_grad)

        update_delta_w_bar = self.delta_w_bar.assign(delta_w_bar_new)
        update_delta_aw = self.delta_aw.assign(delta_aw_new)
        update_delta_vb_bar = self.delta_vb_bar.assign(delta_vb_bar_new)
        update_delta_avb = self.delta_avb.assign(delta_avb_new)
        update_delta_hb_bar = self.delta_hb_bar.assign(delta_hb_bar_new)
        update_delta_ahb = self.delta_ahb.assign(delta_ahb_new)

        update_w_bar = self.w_bar.assign_add(delta_w_bar_new)
        update_aw = self.aw.assign_add(delta_aw_new)
        update_vb_bar = self.vb_bar.assign_add(delta_vb_bar_new)
        update_avb = self.avb.assign_add(delta_avb_new)
        update_hb_bar = self.hb_bar.assign_add(delta_hb_bar_new)
        update_ahb = self.ahb.assign_add(delta_ahb_new)

        # tensorflow computations
        self.update_deltas = [
            update_delta_w_bar,
            update_delta_aw,
            update_delta_vb_bar,
            update_delta_avb,
            update_delta_hb_bar,
            update_delta_ahb,
        ]
        self.update_weights = [
            update_w_bar,
            update_aw,
            update_vb_bar,
            update_avb,
            update_hb_bar,
            update_ahb,
        ]

        w, vb, hb = self._compute_weights()
        self.compute_hidden = self._compute_hidden(self.v_layer, hb, w)
        self.compute_visible = self._compute_visible(self.h_layer, vb, w)
        self.reconstruct_visible = self._compute_visible(self.compute_hidden, vb, w)

    def _get_err(self, batch, batch_adapt):
        return self.sess.run(
            self.compute_err,
            feed_dict={
                self.v_layer: batch,
                self.a_layer: batch_adapt,
            },
        )

    def _partial_fit(self, batch, batch_adapt):
        self.sess.run(
            self.update_weights + self.update_deltas,
            feed_dict={
                self.v_layer: batch,
                self.a_layer: batch_adapt,
            },
        )

    def fit(
        self,
        data,
        adaptation_labels,
        n_epochs=10,
        batch_size=10,
        shuffle=True,
        verbose=True,
    ):
        """
        Fits the given data into the model of the RBM.

        Args:
            data: The training data to be used for learning
            adaptation_labels: The labels of the speakers from which each data point came from. Must be a vector with
                the same size as the number of training examples.
            n_epochs: The number of epochs. Defaults to 10.
            batch_size: The size of the data batch per epoch. Defaults to 10.
            shuffle: True if the data should be shuffled before learning. False otherwise. Defaults to True.
            verbose: True if the progress should be displayed on the standard output during training.

        Returns: An array of the mean square errors of each batch.

        """
        assert n_epochs > 0

        n_data = np.shape(data)[0]

        if batch_size > 0:
            n_batches = n_data // batch_size + (0 if n_data % batch_size == 0 else 1)
        else:
            n_batches = 1

        data_cpy = data.copy()
        adaptation_onehot = np.zeros([n_data, self.n_adaptive])
        for i in range(n_data):
            adaptation_onehot[i][adaptation_labels[i]] = 1
        indexes = np.arange(n_data)

        errs = []

        for e in range(n_epochs):
            if verbose:
                print('Epoch: {:d}'.format(e + 1))

            epoch_errs = np.array([])

            if shuffle:
                np.random.shuffle(indexes)
                data_cpy = data[indexes]
                adaptation_onehot = adaptation_onehot[indexes]

            for batch_nr in range(n_batches):
                batch = data_cpy[batch_nr * batch_size:(batch_nr + 1) * batch_size]
                batch_adapt = adaptation_onehot[batch_nr * batch_size:(batch_nr + 1) * batch_size]
                self._partial_fit(batch, batch_adapt)
                batch_err = self._get_err(batch, batch_adapt)
                epoch_errs = np.append(epoch_errs, batch_err)

            if verbose:
                err_mean = epoch_errs.mean()
                print('Train error: {:.4f}'.format(err_mean))
                print()
                sys.stdout.flush()

            errs = np.hstack([errs, epoch_errs])

        return errs

    def reconstruct(self, v, a):
        onehots = np.zeros([1, self.n_adaptive])
        onehots[0][a] = 1
        return self.sess.run(self.reconstruct_visible, feed_dict={self.v_layer: v, self.a_layer: onehots})


'''
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
'''
'''
    def get_independent_weights(self):
        """
        Gets the weights matrix and bias vectors currently in the model.

        Returns: A tuple (weights matrix, visible units biases, hidden units biases).

        """
        return self.sess.run(self.w_bar), self.sess.run(self.vb_bar), self.sess.run(self.hb_bar)

    def get_adaptive_weights(self):
        """

        Returns:

        """
        return self.sess.run(self.aw), self.sess.run(self.avb), self.sess.run(self.ahb)

    def save_independent_weights(self, filename, name):
        """
        Saves the current weights and biases in the specified file.

        Args:
            filename: The file in which to save the model.
            name: The name of the model.

        Returns: The output of tf.train.Saver.save()

        """
        saver = tf.train.Saver({
            name + '_w_bar': self.w_bar,
            name + '_vb_bar': self.vb_bar,
            name + '_hb_bar': self.hb_bar,
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
        self.sess.run(self.w_bar.assign(w))
        self.sess.run(self.vb_bar.assign(visible_bias))
        self.sess.run(self.hb_bar.assign(hidden_bias))

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
'''
