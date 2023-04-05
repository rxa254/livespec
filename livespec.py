"""
Tested on Linux with python 3.7
Must have portaudio installed (e.g. dnf install portaudio-devel)

pip install pyqtgraph pyaudio

---
forked from https://gist.github.com/boylea/1a0b5442171f9afbf372
March 2023

modified by rxa254 3/2023

* conda install pyqt

"""


import numpy as np
import pyqtgraph as pg
from pyqtgraph import AxisItem
import pyaudio
from PyQt5 import QtCore, QtGui
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

import signal
import sys

def signal_handler(sig, frame):
    print('\n')
    print('SIGINT detected - shutting down gravefully...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


parser = argparse.ArgumentParser()
parser.add_argument('-fs', '--sample_freq', type=int,
                    help='Desired audio sampling frequency',
                    default=44100)
parser.add_argument('-nfft', '--fft_length', type=int,
                    help='FFT length in # of samples. Default = 1024',
                    default=1024)
parser.add_argument('-cmax',
                    help="max z in dB",
                    default=50)
parser.add_argument('-cmin',
                    help="min z in dB",
                    default=-20)
parser.add_argument('--whiten',
                    help="True or False",
                    default=True)
parser.add_argument('--colormap', type=str,
                    help="matplotlib colormap name. default = rainbow",
                    default='rainbow')
parser.add_argument('--freq_scale', type=str,
                    help="log or lin frequency scale. default = lin",
                    default='lin')
parser.add_argument('--specgram_weight', type=float,
                    help="exponential or flat. Default = 0",
                    default=0)
parser.add_argument('--normalize', type=int,
                    help="Gather PSD normalization data on N segments. Default = 0.",
                    default=0)

args = parser.parse_args()



FS = args.sample_freq #Hz
CHUNKSZ = args.fft_length #samples

class MicrophoneRecorder():
    def __init__(self, signal):
        self.signal = signal
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=FS,
                            input=True,
                            frames_per_buffer=CHUNKSZ)

    def read(self):
        data = self.stream.read(CHUNKSZ, exception_on_overflow=False)
        y = np.frombuffer(data, 'int16')
        self.signal.emit(y)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class LogAxisItem(AxisItem):
    def __init__(self, *args, **kwargs):
        AxisItem.__init__(self, *args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        return [f"{10**value:.0f}" for value in values]


class SpectrogramWidget(pg.PlotWidget):
    read_collected = QtCore.pyqtSignal(np.ndarray)
    def __init__(self):
        super(SpectrogramWidget, self).__init__()

        self.img = pg.ImageItem()
        self.addItem(self.img)

        # how many FFTs to display
        N_timeslices = 2**10
        self.img_array  = np.zeros((N_timeslices, int(CHUNKSZ/2+1)))
        self.data_array = np.zeros((N_timeslices, int(CHUNKSZ/2+1)))
        self.iterator = 0

        # colormap
        colormap = mpl.colormaps[args.colormap]
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([args.cmin, args.cmax])

        if args.freq_scale == 'lin':
            # setup the scaling for linear freq y-axis
            freq = np.arange((CHUNKSZ/2)+1)/(float(CHUNKSZ)/FS)
            # setup the scaling for log freq y-axis
            freq = np.logspace(np.log10(20), np.log10(FS/2), int(CHUNKSZ/2)+1)
            yscale = 1.0/(self.img_array.shape[1]/freq[-1])
            self.img.scale((1./FS)*CHUNKSZ, yscale)
            self.setLabel('left', 'Frequency', units='Hz')

        elif args.freq_scale == 'log':
            # setup the correct scaling for log y-axis
            freq = np.logspace(np.log10(1), np.log10(FS/2), CHUNKSZ//2+1, base=10)
            yscale = 1.0/(self.img_array.shape[1]/np.log10(freq[-1]/freq[0]))
            self.img.scale((1./FS)*CHUNKSZ, yscale)
            self.getAxis('left').setScale(LogAxisItem(orientation='left'))
            self.getAxis('left').setLabel('Frequency', units='Hz')


        self.setLabel('bottom', 'Time', units='s')

        # prepare window for later use
        self.win = np.hanning(CHUNKSZ)
        self.show()

    def update(self, chunk):
        # normalized, windowed frequencies in data chunk
        spec = np.fft.rfft(chunk*self.win) / CHUNKSZ
        # get magnitude
        psd = abs(spec)
        asd = np.sqrt(psd)
        # convert to dB scale
        z = 20 * np.log10(psd)

        #smoo_const = 0.5
        #z_ema = self.img_array[-1:]
        #z_ema = (z - z_ema)*smoo_const + z_ema

        # data array for calculating the moving average
        whiten = args.whiten
        self.data_array = np.roll(self.data_array, -1, 0)
        self.data_array[-1:] = psd

        ### make an exponentially averaging window
        nx, ny = np.shape(self.data_array)
        wm = np.ones_like(self.data_array)
        wx = np.flipud(np.arange(nx))

        if args.normalize > 0:
            while self.iterator < args.normalize:
                self.iterator += 1

            psd_ave = np.mean(self.data_array, axis=0)



        # make the weighting matrix for the specgram
        # 0 = no weighting
        # > 0 means do some exponential averaging
        wv = np.exp(-1 * args.specgram_weight * wx)

        # make an exponentially weighted matrix
        wm = np.broadcast_to(wv,(ny, len(wv)))
        weighted_data_array = wm.T * self.data_array
        psd_ave = np.mean(weighted_data_array, axis=0)

        if whiten:
            z_ave = 20*np.log10(psd_ave)
        else:
            z_ave = 0

        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = z - z_ave

        self.img.setImage(self.img_array, autoLevels=False)

if __name__ == '__main__':
    app = QtGui.QApplication([])
    w = SpectrogramWidget()
    w.read_collected.connect(w.update)

    mic = MicrophoneRecorder(w.read_collected)

    # time (seconds) between reads
    interval = CHUNKSZ/FS
    t = QtCore.QTimer()
    t.timeout.connect(mic.read)
    t.start(int(1000 * interval)) #QTimer takes ms

    app.exec_()
    mic.close()
