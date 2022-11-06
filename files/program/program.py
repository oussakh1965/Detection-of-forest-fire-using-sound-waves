# IMPORTANT
# Before running this file, make sure to set \projet-feux-de-foret\ as the current work directory


import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from modules.functions import *
from tensorflow.keras.models import load_model
import os
print(os.path.dirname(os.path.realpath(__file__)))
tf.config.run_functions_eagerly(True)


# trained models: model_0 and model_1 being the first two models made, and model_noise_0 and model_noise_1 include noise in the training datased
models = ['model_0', 'model_1', 'model_noise_0',
          'model_noise_1']

# specify the file name in audio repository
filename = 'record'
# specify the id of the model you want to use (0 for model_0 all the way to 3 for model_noise_1)
model_id = 2


def call_IA(filename, model_id):
    model = load_model('files/program/'+models[model_id])
    # loads the model
    names = prepare_audiofile(filename, 0)
    # prepare audiofile by dividing in into 5 sec parts and make a list with all the names of the parts
    try:
        data = preprocess_dataset(names, model_id % 2)
        # models use different stft, model_id % 2 will select the right stft that works with the model
        audio_segments = []
        for audio, label in data:
            print(audio.shape)
            audio_segments.append(audio)
        prediction = model(np.array(audio_segments))
        print(prediction)
        # prediction is a matrix specifying the probability of the predictionfor each segment and each label
        results = []
        for element in prediction:
            result = states[np.argmax(element)]
            results.append(result)
        print(results)
        # results is an array containing the prediction for each segment.
    except Exception as e:
        print(e)

    # try is put here to delate to new made files even if something goes wrong with the program
    delete_files(names)
    return results


call_IA(filename, model_id)
