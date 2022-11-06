# convert a tensorflow 2 model into a tensorflow lite model

from tensorflow.keras.models import load_model
import tensorflow as tf
print(tf.__version__)

# model = load_model('files/program/model_1')
converter = converter = tf.lite.TFLiteConverter.from_saved_model(
    'files/program/model_2')
tflite_model = converter.convert()

# Save the model.
with open('files/program/model_2.tflite', 'wb') as f:
    f.write(tflite_model)
