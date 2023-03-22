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
import pyaudio
from PyQt5 import QtCore, QtGui
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-fs', '--sample_freq', type=int,
                    help='Desired audio sampling frequency',
                    default=44100)
parser.add_argument('-nfft', '--fft_length', type=int,
                    help='FFT length in # of samples. Default = 1024',
                    default=1024)
parser.add_argument('-cmax',
                    help="max z in dB",
                    default=20)
parser.add_argument('-cmin',
                    help="min z in dB",
                    default=-20)
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

class SpectrogramWidget(pg.PlotWidget):
    read_collected = QtCore.pyqtSignal(np.ndarray)
    def __init__(self):
        super(SpectrogramWidget, self).__init__()

        self.img = pg.ImageItem()
        self.addItem(self.img)

        self.img_array  = np.zeros((1024, int(CHUNKSZ/2+1)))
        self.data_array = np.zeros( (1024, int(CHUNKSZ/2+1)))

        # bipolar colormap
        #pos = np.array([0., 1., 0.5, 0.25, 0.75])
        #color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        #cmap = pg.ColorMap(pos, color)
        #lut = cmap.getLookupTable(0.0, 1.0, 256)
        # colormap
        colormap = mpl.colormaps['rainbow']
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-20,20])


        # setup the correct scaling for y-axis
        freq = np.arange((CHUNKSZ/2)+1)/(float(CHUNKSZ)/FS)
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        self.img.scale((1./FS)*CHUNKSZ, yscale)

        self.setLabel('left', 'Frequency', units='Hz')
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

        # roll down one and replace leading edge with new data

        # data array for calculating the moving average
        whiten = True
        self.data_array = np.roll(self.data_array, -1, 0)
        self.data_array[-1:] = psd
        psd_ave = np.mean(self.data_array, axis=0)
        if whiten:
            z_ave = 20*np.log10(psd_ave)
        else:
            z_ave = 0

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
