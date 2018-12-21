import matplotlib.pyplot as plt
from arbm import ARBM
import numpy as np
from data_pipeline import get_features

EPOCHS = 10
BATCH_S = 10

data = get_features("data/vcc2016_training/SF1", 40)

arbm = ARBM(
    n_visible=len(data[0]),
    n_hidden=256,
    n_adaptive=1,
    learning_rate=0.001,
    momentum=0.95,
    cdk_level=1,
    sample_visible=True,
)
errs = arbm.fit(data, np.zeros(len(data), dtype=np.int32), n_epochs=EPOCHS, batch_size=BATCH_S)
plt.plot(errs)
plt.show()
