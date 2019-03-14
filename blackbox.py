import numpy as np
from rbm import RBM

class ARBM:
    def __init__(
            self,
            n_visible,
            n_hidden,
            n_adaptive,
            sample_visible=False,
            learning_rate=0.01,
            momentum=0.95,
            cdk_level=1,
    ):
        print("ARBM:")
        print(
            "\tNumber of visible units: {}".format(n_visible),
            "\tNumber of hidden units: {}".format(n_hidden),
            "\tNumber of adaptive units: {}".format(n_adaptive),
            sep="\n",
        )

    def fit(
            self,
            data,
            n_epochs=10,
            batch_size=10,
            shuffle=True,
            verbose=True,
    ) -> np.ndarray:
        print("\tFitting data:")
        for key, val in data.items():
            print("\t\t{}: {}".format(key, val.shape))
        return np.linspace(1, 0, n_epochs)**2

    def convert(
            self,
            source_label,
            source_data,
            target_label,

    ):
        return source_data

    def add_speaker(
            self,
            speaker_data,
    ):
        print("\tAdding speaker using data {}".format(speaker_data.shape))
        return -1

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
    def load_model(filename):
        return ARBM(0, 0, 0)


class ARBM_RBM:
    def __init__(
            self,
            n_visible,
            n_hidden,
            n_adaptive,
            sample_visible=False,
            learning_rate=0.01,
            momentum=0.95,
            cdk_level=1,
    ):
        print("ARBM_RBM")
        print("\tThis class mimics the normal RBM.")
        self.rbm = RBM(
            n_visible=n_visible,
            n_hidden=n_hidden,
            learning_rate=learning_rate,
            momentum=momentum,
            cdk_level=cdk_level,
        )

    def fit(
            self,
            data,
            n_epochs=10,
            batch_size=10,
            shuffle=True,
            verbose=True,
    ) -> np.ndarray:
        return self.rbm.fit(
            data=np.transpose(data[0]),
            n_epochs=n_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            verbose=verbose,
        )

    def convert(
            self,
            source_label,
            source_data,
            target_label,

    ):
        return np.transpose(self.rbm.reconstruct(np.transpose(source_data)))

    def add_speaker(
            self,
            speaker_data,
    ):
        print("\tCannot add speaker.")
        return -1

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
    def load_model(filename):
        return ARBM_RBM(0, 0, 0)

