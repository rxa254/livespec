import pyaudio
import numpy as np
import matplotlib.pyplot as plt

p = pyaudio.PyAudio()

# Parameters for the spectrogram
NFFT = 1024  # Number of points for the FFT
Fs = 44100   # Sampling frequency

# Initialize the plot
fig, ax = plt.subplots()
img = ax.imshow(np.zeros((NFFT // 2 + 1, 100)), extent=[0, 1, 0, Fs/2], cmap='inferno', aspect='auto')

# Function to update the spectrogram
def update_specgram(in_data, frame_count, time_info, status):
    # Compute the spectrogram
    spec = np.abs(np.fft.rfft(np.frombuffer(in_data, dtype=np.int16), n=NFFT))
    spec = 20 * np.log10(spec)
    
    # Update the plot data
    img.set_data(np.hstack([img.get_array()[:, 1:], np.atleast_2d(spec).T]))

    #img.setImage(img_array, autoLevels=True)
    # Redraw the plot
    fig.canvas.draw()

# Start the microphone input stream and update the spectrogram

stream = p.open(format=pyaudio.paInt16, channels=1, rate=Fs, input=True, frames_per_buffer=NFFT, stream_callback=update_specgram)

stream.start_stream()

# Continuous update loop
plt.show(block=False)
while stream.is_active():
    plt.pause(0.01)

# Stop the stream and close the audio device
stream.stop_stream()
stream.close()
p.terminate()


