import tensorflow as tf


def sample_bernoulli(probs):
    """
    Samples a binary tensor based no the given element-wise probabilities.

    Args:
        probs: A tensor representing the probabilities of each element to be equal to 1.

    Returns: A tensor with the same shape as probs with the sampled elements.

    """
    return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))


def sample_gaussian(x, sigma):
    """
    Samples a real valued tensor based on gaussian distributions with
        the give element-wise means and global sigma.
    Args:
        x: A tensor representing the means of the gaussian distributions of each element.
        sigma: The standard deviation of all the gaussian distributions.

    Returns: A tensor of samples with the same shape as x.

    """
    return x + tf.random_normal(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)


def sample_softmax(probs):
    return tf.one_hot(tf.argmax(probs, axis=-1), tf.shape(probs)[-1])


def trans(tensor, **kwargs):
    return tf.transpose(tensor, **kwargs)
