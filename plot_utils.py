from matplotlib import pyplot as plt
import numpy as np
from typing import List, Optional, Tuple


class Plotter:
    def __init__(
            self,
    ):
        self._figures: List[plt.Figure] = []

    def get_figures(self) -> List[plt.Figure]:
        return self._figures

    def get_figure(
            self,
            index: int,
    ) -> Optional[plt.Figure]:
        try:
            return self._figures[index]
        except IndexError as e:
            print(e)
            return None

    def add_figure(
            self,
            gridspec: Tuple[int, int] = (1, 1),
            title: str = None,
            tracking: bool = True,
    ) -> plt.Figure:
        if title is not None:
            fig = plt.figure(title)
        elif tracking:
            fig = plt.figure("Figure {}".format(len(self._figures)))
        else:
            fig = plt.figure("Figure")
        gs = fig.add_gridspec(gridspec[0], gridspec[1])
        for row in range(gridspec[0]):
            for col in range(gridspec[1]):
                fig.add_subplot(gs[row, col])
        if tracking:
            self._figures.append(fig)
        return fig

    def plot_time_domain(
            self,
            samples: np.ndarray,
            sample_rate: float,
            fig_index: int = None,
            axes_index: int = 0,
            axes_title: str = "",
    ):
        assert samples.ndim == 1
        if fig_index is None:
            fig = self.add_figure(tracking=False)
        else:
            fig = self.get_figure(fig_index)
        assert fig is not None
        assert 0 <= axes_index < len(fig.axes)
        sound_duration = len(samples) / sample_rate
        time_axis = np.linspace(0, sound_duration, num=len(samples))
        fig.axes[axes_index].plot(time_axis, samples)
        fig.axes[axes_index].set_title(axes_title)
        fig.axes[axes_index].set_xlabel("Time")
        fig.axes[axes_index].set_ylabel("Amplitude")

    def plot_frequency_domain(
            self,
            samples: np.ndarray,
            sample_rate: float,
            fig_index: int = None,
            axes_index: int = 0,
            axes_title: str = "",
    ):
        assert samples.ndim == 1
        if fig_index is None:
            fig = self.add_figure(tracking=False)
        else:
            fig = self.get_figure(fig_index)
        assert fig is not None
        assert 0 <= axes_index < len(fig.axes)
        fig.axes[axes_index].magnitude_spectrum(samples, Fs=sample_rate)
        fig.axes[axes_index].set_title(axes_title)

    def plot_power_spectral_density(
            self,
            samples: np.ndarray,
            sample_rate: float,
            frame_size: int = 256,
            fig_index: int = None,
            axes_index: int = 0,
            axes_title: str = "",
    ):
        assert samples.ndim == 1
        if fig_index is None:
            fig = self.add_figure(tracking=False)
        else:
            fig = self.get_figure(fig_index)
        assert fig is not None
        assert 0 <= axes_index < len(fig.axes)
        fig.axes[axes_index].psd(samples, NFFT=frame_size, Fs=sample_rate)
        fig.axes[axes_index].set_title(axes_title)

    def plot_spectrogram(
            self,
            samples: np.ndarray,
            sample_rate: float,
            frame_size: int = 256,
            fig_index: int = None,
            axes_index: int = 0,
            axes_title: str = "",
    ):
        assert samples.ndim == 1
        if fig_index is None:
            fig = self.add_figure(tracking=False)
        else:
            fig = self.get_figure(fig_index)
        assert fig is not None
        assert 0 <= axes_index < len(fig.axes)
        fig.axes[axes_index].specgram(samples, NFFT=frame_size, Fs=sample_rate)
        fig.axes[axes_index].set_title(axes_title)
        fig.axes[axes_index].set_xlabel("Time")
        fig.axes[axes_index].set_ylabel("Frequency Amplitudes")

    def plot_line(
            self,
            data: np.ndarray,
            fig_index: int = None,
            axes_index: int = 0,
            axes_title: str = "",
            xlabel: str = "",
            ylabel: str = "",
    ):
        assert data.ndim == 1
        if fig_index is None:
            fig = self.add_figure(tracking=False)
        else:
            fig = self.get_figure(fig_index)
        assert fig is not None
        assert 0 <= axes_index < len(fig.axes)
        fig.axes[axes_index].plot(data)
        fig.axes[axes_index].set_title(axes_title)
        fig.axes[axes_index].set_xlabel(xlabel)
        fig.axes[axes_index].set_ylabel(ylabel)

    def plot_heatmap(
            self,
            data: np.ndarray,
            fig_index: int = None,
            axes_index: int = 0,
            axes_title: str = "",
            xlabel: str = "",
            ylabel: str = "",
    ):
        assert data.ndim == 2
        if fig_index is None:
            fig = self.add_figure(tracking=False)
        else:
            fig = self.get_figure(fig_index)
        assert fig is not None
        assert 0 <= axes_index < len(fig.axes)
        fig.axes[axes_index].imshow(data)
        fig.axes[axes_index].set_title(axes_title)
        fig.axes[axes_index].set_xlabel(xlabel)
        fig.axes[axes_index].set_ylabel(ylabel)

    def plot_mfcc(
            self,
            mfcc: np.ndarray,
            fig_index: int = None,
            axes_index: int = 0,
            axes_title: str = "",
    ):
        self.plot_heatmap(
            data=mfcc,
            fig_index=fig_index,
            axes_index=axes_index,
            axes_title=axes_title,
            xlabel="Time",
            ylabel="MFCC",
        )

    def plot_line_comp(
            self,
            data_1: np.ndarray,
            data_2: np.ndarray,
            data_1_title: str = None,
            data_2_title: str = None,
            title: str = None,
    ):
        assert data_1.ndim == data_2.ndim == 1
        if data_1_title is None:
            data_1_title = "data 1"
        if data_2_title is None:
            data_2_title = "data 2"
        if title is None:
            title = "Line Comparison"
        fig = self.add_figure((3, 1), title, tracking=False)
        fig.axes[0].plot(data_1)
        fig.axes[0].set_title(data_1_title)
        fig.axes[1].plot(data_2)
        fig.axes[1].set_title(data_2_title)
        lim = min(data_1.shape[0], data_2.shape[0])
        fig.axes[2].plot(data_1[:lim] - data_2[:lim])
        fig.axes[2].set_title("difference")

    def plot_heatmap_comp(
            self,
            data_1: np.ndarray,
            data_2: np.ndarray,
            data_1_title: str = None,
            data_2_title: str = None,
            title: str = None,
    ):
        assert data_1.ndim == data_2.ndim == 2
        if data_1_title is None:
            data_1_title = "data 1"
        if data_2_title is None:
            data_2_title = "data 2"
        if title is None:
            title = "Heatmap Comparison"
        xlim = min(data_1.shape[1], data_2.shape[1])
        ylim = min(data_1.shape[0], data_2.shape[0])
        if ylim < xlim:
            fig = self.add_figure((3, 1), title, tracking=False)
        else:
            fig = self.add_figure((1, 3), title, tracking=False)
        fig.axes[0].imshow(data_1)
        fig.axes[0].set_title(data_1_title)
        fig.axes[1].imshow(data_2)
        fig.axes[1].set_title(data_2_title)
        fig.axes[2].imshow(data_1[:ylim, :xlim] - data_2[:ylim, :xlim])
        fig.axes[2].set_title("difference")

    def show(self):
        for fig in self._figures:
            fig.tight_layout()
        plt.show()

    def save_figures(
            self,
            filename_base: str = "figure",
    ):
        for index, fig in enumerate(self._figures):
            fig.savefig("{}_{}.png".format(filename_base, index))
