import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from modules.functions import *

# this scripts takes an audio file as an input, divides it into parts of 5 secs having the right format
# it then plots the spectrogram of up to the 10 first parts

filename = 'test_feu_8'
names = prepare_audiofile(filename, 0, label='')
i = 0
for name in names:
    waveform, label = get_waveform_and_label(name)
    spectrogram = get_spectrogram_0(waveform)
    print(spectrogram.shape)
    fig, axes = plt.subplots(1, figsize=(12, 8))
    timescale = np.arange(waveform.shape[0])

    plot_spectrogram(spectrogram.numpy(), axes)
    axes.set_title('Spectrogram')
    axes.set_xlabel('time in secs')
    axes.set_ylabel('Frequency in Hz')
    plt.show()
    i += 1
    if i == 10:
        break

delete_files(names)
