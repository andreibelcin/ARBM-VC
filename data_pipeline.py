import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import rfft, irfft
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join


def _read_files_from_dir(
    dirname,
    nr_files=1,
):
    wav_files = []
    sample_rates = []
    filenames = [join(dirname, filename)
                 for filename in listdir(dirname) if isfile(join(dirname, filename))]
    filenames.sort()
    for filename in filenames[:nr_files]:
        sample_rate, samples = wav.read(filename)
        wav_files.append(samples)
        sample_rates.append(sample_rate)
    return sample_rates, wav_files


def _get_frames_from_samples(
    samples,
    sample_rate,
    frame_size=0.1,
    frame_stride=None,
):
    if frame_stride is None:
        frame_stride = frame_size
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(samples)
    num_frames = int(np.ceil(float(signal_length) / frame_step))

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros([pad_signal_length - signal_length])
    pad_signal = np.append(samples, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    return frames


def get_features(dirname, nr_files):
    sample_rates, wav_files = _read_files_from_dir(dirname, nr_files)
    frames = []
    for i in range(len(wav_files)):
        frames.extend(rfft(_get_frames_from_samples(wav_files[i], sample_rates[i])))
    return frames


def __test__():
    nr_files = 4
    sample_rates, wav_files = _read_files_from_dir("data/vcc2016_training/SF1", nr_files=nr_files)
    plt.plot(wav_files[0][:10000])
    plt.show()
    frames = _get_frames_from_samples(wav_files[0], sample_rates[0])
    plt.plot(frames[0])
    plt.show()
    plt.plot(rfft(frames[0]))
    plt.show()


if __name__ == "__main__":
    __test__()
