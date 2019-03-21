from os import listdir
from os.path import isfile, join
from typing import Dict, List

import numpy as np
import scipy.io.wavfile as wav
from plot_utils import Plotter
from scipy.signal import stft, istft
from scipy.fftpack import dct, idct

from arbm import ARBM
from librosa.feature import mfcc


FEATURE_VECTOR_LENGTH = 32
FEATURE_TYPE = "mfcc"


class Dirs:
    SF1 = "data/vcc2016_training/SF1"
    SF2 = "data/vcc2016_training/SF2"
    SF3 = "data/vcc2016_training/SF3"
    SM1 = "data/vcc2016_training/SM1"
    SM2 = "data/vcc2016_training/SM2"
    TF1 = "data/vcc2016_training/TF1"
    TF2 = "data/vcc2016_training/TF2"
    TM1 = "data/vcc2016_training/TM1"
    TM2 = "data/vcc2016_training/TM2"
    TM3 = "data/vcc2016_training/TM3"
    TRAIN = [SF1, SF2, SM1, SM2, TF1, TM1]
    TEST = [SF1, SF2, SM2]
    TEST_LABELS = [0, 1, 3]


def load_model(
        filename: str,
) -> ARBM:
    return ARBM.load_model(filename)


def read_files_from_dir(
        dirname: str,
        nr_files: int = 0,
) -> (List[int], List[np.ndarray]):
    wav_files = []
    sample_rates = []
    filenames = np.array([join(dirname, filename)
                          for filename in listdir(dirname) if isfile(join(dirname, filename))])
    filenames.sort()
    if nr_files:
        filenames_slice = filenames[:nr_files]
    else:
        filenames_slice = filenames
    for filename in filenames_slice:
        sample_rate, samples = wav.read(filename)
        wav_files.append(samples.astype(float))
        sample_rates.append(sample_rate)
    return sample_rates, wav_files


def extract_features(
        sample_rate: int,
        wav_file: np.ndarray,
) -> np.ndarray:
    def mfcc_features():
        return mfcc(wav_file, sr=sample_rate, n_mfcc=FEATURE_VECTOR_LENGTH+2)[2:]

    def psc_features():
        pass

    def pcc_features():
        pass

    def pstc_features():
        pass

    def freq_features():
        _, _, Zxx = stft(wav_file, fs=sample_rate)
        return np.abs(Zxx)
    return {
        "mfcc": mfcc_features(),
        "psc": psc_features(),
        "pcc": pcc_features(),
        "pstc": pstc_features(),
        "freq": freq_features(),
    }[FEATURE_TYPE]


def extract_features_from_dataset(
        sample_rates: List[int],
        wav_files: List[np.ndarray],
) -> np.ndarray:
    data = extract_features(sample_rates[0], wav_files[0])
    for sample_rate, wav_file in zip(sample_rates[1:], wav_files[1:]):
        data = np.hstack((data, extract_features(sample_rate, wav_file)))
    return data


def reconstruct_speech(
        speech_features: np.ndarray,
        sample_rate: int,
) -> np.ndarray:
    def mfcc_reconstruct():
        return speech_features

    def psc_reconstruct():
        pass

    def pcc_reconstruct():
        pass

    def pstc_reconstruct():
        pass

    def freq_reconstruct():
        _, x = istft(speech_features, sample_rate)
        return x
    return {
        "mfcc": mfcc_reconstruct(),
        "psc": psc_reconstruct(),
        "pcc": pcc_reconstruct(),
        "pstc": pstc_reconstruct(),
        "freq": freq_reconstruct(),
    }[FEATURE_TYPE]


def make_model(
        n_visible: int = FEATURE_VECTOR_LENGTH,
        n_hidden: int = 1,
        n_adaptive: int = 1,
        sample_visible: bool = False,
        learning_rate: float = 0.01,
        momentum: float = 0.95,
        cdk_level: int = 1,
        data_dict: Dict[int, np.ndarray] = None,
        n_epochs: int = 20,
        batch_size: int = 100,
        shuffle: bool = True,
        verbose: bool = True,
) -> (ARBM, np.ndarray):
    assert data_dict is not None
    arbm = ARBM(
        n_visible,
        n_hidden,
        n_adaptive,
        sample_visible=sample_visible,
        learning_rate=learning_rate,
        momentum=momentum,
        cdk_level=cdk_level,
    )
    errors = arbm.fit(
        data=data_dict,
        n_epochs=n_epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=verbose,
    )
    return arbm, errors


def convert(
        model: ARBM,
        source_speech: np.ndarray,
        source_label: int,
        target_label: int,
) -> np.ndarray:
    source_data = np.transpose(extract_features_from_dataset([44100], [source_speech]))
    target_data = model.convert(
        source_label=source_label,
        source_data=source_data,
        target_label=target_label,
    )
    target_speech = reconstruct_speech(np.transpose(target_data), 44100)
    return target_speech


def add_speaker(
        model: ARBM,
        data_dir: str,
        nr_files: int,
) -> int:
    sample_rates, wav_files = read_files_from_dir(data_dir, nr_files=nr_files)
    data = extract_features_from_dataset(sample_rates, wav_files)
    return model.add_speaker(np.transpose(data))


def get_train_data() -> Dict[int, np.ndarray]:
    data_dict = {}
    label = 0
    for data_dir in Dirs.TRAIN:
        sfs, wavs = read_files_from_dir(data_dir, 50)
        data = extract_features_from_dataset(sfs, wavs)
        data_dict[label] = data
        label += 1
    return data_dict


def get_test_data() -> Dict[int, np.ndarray]:
    data_dict = {}
    label_index = 0
    for data_dir in Dirs.TEST:
        sfs, wavs = read_files_from_dir(data_dir, 1)
        data = extract_features_from_dataset(sfs, wavs)
        data_dict[Dirs.TEST_LABELS[label_index]] = data
        label_index += 1
    return data_dict


def main():
    p = Plotter()
    train_data = get_train_data()
    arbm, errors = make_model(
        n_visible=FEATURE_VECTOR_LENGTH,
        n_hidden=FEATURE_VECTOR_LENGTH*8,
        n_adaptive=len(train_data),
        data_dict=train_data,
        sample_visible=False,
        n_epochs=60,
        batch_size=150,
    )
    p.plot_line(errors, axes_title="Error rate")
    test_data = get_test_data()
    target_test_data_1 = arbm.convert(0, test_data[0], 0)
    p.plot_heatmap_comp(test_data[0], target_test_data_1, "Target", "Conversion", "Conversion Comparison T")
    p.plot_heatmap_comp(test_data[1], target_test_data_1, "Source", "Conversion", "Conversion Comparison S")
    target_test_data_2 = arbm.reconstruct(1, test_data[1])
    p.plot_heatmap_comp(test_data[1], target_test_data_2, "Original", "Reconstruction", "Reconstruction Comparison")
    p.show()


def playground():
    pass


if __name__ == "__main__":
    main()
