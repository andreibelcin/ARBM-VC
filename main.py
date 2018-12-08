import matplotlib.pyplot as plt
from arbm import ARBM
from rbm import RBM
import tensorflow as tf
import numpy as np


def show_digit(x):
    plt.imshow(x.reshape((28, 28)))
    plt.show()


DSIZE = 2000
EPOCHS = 10
BATCH_S = 10
IMAGES = [1, 3, 5, 10]

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


arbm = ARBM(
    n_visible=784,
    n_hidden=1024,
    n_adaptive=1,
    learning_rate=0.001,
    momentum=0.95,
    cdk_level=1,
    sample_visible=True,
)
errs = arbm.fit(np.reshape(x_train[:DSIZE], [-1, 784]), np.zeros([DSIZE], dtype=np.int32), n_epochs=EPOCHS, batch_size=BATCH_S)
plt.plot(errs)
plt.show()

for image_id in IMAGES:
    image = x_train[image_id]
    image_rec = arbm.reconstruct(image.reshape(1, -1), 0)
    show_digit(image)
    show_digit(image_rec)

rbm = RBM(
    n_visible=784,
    n_hidden=1024,
    learning_rate=0.001,
    momentum=0.95,
    cdk_level=1,
    sample_visible=True
)
errs = rbm.fit(np.reshape(x_train[:DSIZE], [-1, 784]), n_epochs=EPOCHS, batch_size=BATCH_S)
plt.plot(errs)
plt.show()

for image_id in IMAGES:
    image = x_train[image_id]
    image_rec = rbm.reconstruct(image.reshape(1, -1))
    show_digit(image)
    show_digit(image_rec)
