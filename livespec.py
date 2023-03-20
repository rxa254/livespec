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

FS = 44100 #Hz
CHUNKSZ = 1024 #samples

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

        self.img_array = np.zeros((1000, int(CHUNKSZ/2+1)))

        # bipolar colormap
        pos = np.array([0., 1., 0.5, 0.25, 0.75])
        color = np.array([[0,255,255,255], [255,255,0,255], [0,0,0,255], (0, 0, 255, 255), (255, 0, 0, 255)], dtype=np.ubyte)
        cmap = pg.ColorMap(pos, color)
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        # colormap
        colormap = mpl.colormaps['inferno']
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-50,40])

        # set colormap
        self.img.setLookupTable(lut)
        self.img.setLevels([-40, 40])

        # setup the correct scaling for y-axis
        freq = np.arange((CHUNKSZ/2)+1)/(float(CHUNKSZ)/FS)
        yscale = 1.0/(self.img_array.shape[1]/freq[-1])
        self.img.scale((1./FS)*CHUNKSZ, yscale)

        self.setLabel('left', 'Frequency', units='Hz')

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
        z = 20 * np.log10(asd)

        # roll down one and replace leading edge with new data
        self.img_array = np.roll(self.img_array, -1, 0)
        self.img_array[-1:] = z

        self.img.setImage(self.img_array, autoLevels=True)

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
