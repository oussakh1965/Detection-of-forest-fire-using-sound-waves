import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import wave
from pydub import AudioSegment
from math import ceil
import pathlib
tf.config.run_functions_eagerly(True)


states = ['fire', 'other']


# re using the functions used to prepare audio files before calling the AI
def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Since all the data is single channel (mono), drop the `channels`
    # axis from the array.
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


AUTOTUNE = tf.data.AUTOTUNE


def get_spectrogram_0(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 80000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [80000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def get_spectrogram_1(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 80000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [80000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=1023, frame_step=512)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

# different stft are used by different AI models


def get_spectrogram_and_label_id_0(audio, label):
    spectrogram = get_spectrogram_0(audio)
    print('Waveform shape:', audio)
    print('Spectrogram shape:', spectrogram)
    label_id = tf.argmax(label == states)
    return spectrogram, label_id


def get_spectrogram_and_label_id_1(audio, label):
    spectrogram = get_spectrogram_1(audio)
    print('Waveform shape:', audio)
    print('Spectrogram shape:', spectrogram)
    label_id = tf.argmax(label == states)
    return spectrogram, label_id


def preprocess_dataset(files, mode):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)
    if mode == 0:
        output_ds = output_ds.map(
            map_func=get_spectrogram_and_label_id_0,
            num_parallel_calls=AUTOTUNE)
    if mode == 1:
        output_ds = output_ds.map(
            map_func=get_spectrogram_and_label_id_1,
            num_parallel_calls=AUTOTUNE)
    return output_ds

# fucntion plot_data.py to draw the spectrogram


def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    XX = [x/16000 for x in X]
    Y = range(height)
    YY = [y*16000/129 for y in Y]
    ax.pcolormesh(XX, YY, log_spec)


def compose_path(filename):
    return str('files\\audio\\'+filename+'.wav')


def file_from_dataset(filename, label):
    return str('datasets\\samples\\'+label+'\\'+filename+'.wav')


def prepare_audiofile(filename, amp=0, label=None):
    if label in ['fire', 'other']:
        complete_path = file_from_dataset(filename, label)
    else:
        complete_path = compose_path(filename)
    print(complete_path)
    with wave.open(complete_path, "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        nframes = infile.getnframes()
        duration = nframes/float(framerate)
        nparts = ceil(duration/5)
        print('number of slices: '+str(nparts))
        Data = [0]*nparts
        names = [0]*nparts

        for k in range(0, nparts):
            # set position in wave to start of segment
            infile.setpos(int(5 * k * framerate))
            # extract data
            Data[k] = infile.readframes(int(5 * framerate))

    print('done getting data')
    for m in range(0, nparts):
        # write the extracted data to a new file
        new_name = filename+'_'+str(m+1)
        final_path = compose_path(new_name)
        with wave.open(final_path, 'w') as outfile:
            outfile.setnchannels(nchannels)
            outfile.setsampwidth(2)
            outfile.setframerate(framerate)
            outfile.setnframes(int(len(Data[m]) / sampwidth))
            outfile.writeframes(Data[m])
        sound = AudioSegment.from_wav(final_path)
        sound = sound + amp
        sound = sound.set_channels(1)
        sound = sound.set_frame_rate(16000)
        sound.export(final_path, format="wav")
        names[m] = final_path
    print('done preparing file')
    return names


def delete_files(names):
    for name in names:
        os.remove(name)
    print('files deleted')
