from __future__ import print_function
import wave
from pydub import AudioSegment
from math import ceil
import random


def compose_path(filename, label='fire'):
    return str('datasets/untouched/'+label+'/'+filename+'.wav')


def compose_final(filename, label='fire'):
    return str('datasets/brut/'+label+'/'+filename+'.wav')


for i in range(1, 2224):
    # script to divide audio files and convert them to the right formats
    # break in the beggining to not run the script
    break
    filename = 'other_' + str(i)

    complete_path = compose_path(filename, 'other')

    # file to extract the snippet from
    with wave.open(complete_path, "rb") as infile:
        # get file data
        nchannels = infile.getnchannels()
        sampwidth = infile.getsampwidth()
        framerate = infile.getframerate()
        nframes = infile.getnframes()
        duration = nframes/float(framerate)
        nparts = ceil(duration/5)
        print(duration)
        print(framerate, nframes, nparts)
        print('number of slices: '+str(nparts))
        Data = [0]*nparts

        for k in range(0, nparts):
            # set position in wave to start of segment
            infile.setpos(int(5 * k * framerate))
            # extract data
            Data[k] = infile.readframes(int(5 * framerate))

    print('done getting data')
    for m in range(0, nparts):
        # write the extracted data to a new file
        new_name = filename+'_'+str(m)
        final_path = compose_final(new_name, 'other')
        with wave.open(final_path, 'w') as outfile:
            outfile.setnchannels(nchannels)
            outfile.setsampwidth(sampwidth)
            outfile.setframerate(framerate)
            outfile.setnframes(int(len(Data[m]) / sampwidth))
            outfile.writeframes(Data[m])
        sound = AudioSegment.from_wav(final_path)
        sound = sound.set_channels(1)
        sound = sound.set_frame_rate(16000)
        sound.export(final_path, format="wav")
        print(new_name+' done !')


def match_target_amplitude(sound, target_dBFS):
    # match a sound's amplitude to the target amplitude
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def mix_sounds(name1, gain2, name2=None, label='fire'):
    # mixes two sounds and and save mixed sound in a directory matching the labels
    # if name2 is not specified, copies to the diroctory matching the label
    path1 = compose_path(name1, label)
    sound1 = AudioSegment.from_file(path1, "wav")
    normalized_sound1 = match_target_amplitude(sound1, -9)
    if name2 == None:
        normalized_sound1.export(compose_final(name1, label), format="wav")
        print('picked file : '+name1)
    else:
        path2 = compose_path(name2, 'combine')
        combined_path = compose_final(name1+'+'+name2, label)
        sound2 = AudioSegment.from_file(path2, "wav")
        normalized_sound2 = match_target_amplitude(sound2, gain2)
        overlay = normalized_sound1.overlay(normalized_sound2, position=0)
        overlay.export(combined_path, format="wav")
        print('combined : '+name1+' and '+name2)
