import os

import json

import tensorflowjs as tfjs
import tensorflow as tf
from tensorflow import keras
from scipy.io import wavfile
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)




wav_file_path = '/tf/dataset/bubbling/02.wav'
#wav_file_path = '/tmp/tfjs-sc-model/audio_sample_one_male_adult.wav'

print(tf.version.VERSION)
preproc_model_path = '/tmp/tfjs-sc-model/sc_preproc_model'
#preproc_model_path = '/tf/checkpoint/aaaamymodel_52000'

tfjs_model_json_path = '/tmp/tfjs-sc-model/model.json'
# This is the main classifier model.
model = tfjs.converters.load_keras_model(tfjs_model_json_path)



preproc_model = tf.keras.models.load_model(preproc_model_path)
preproc_model.summary()
input_length = preproc_model.input_shape[-1]
print("Input audio length = %d" % input_length)


combined_model = tf.keras.Sequential(name='combined_model')
combined_model.add(preproc_model)
combined_model.add(model)
combined_model.build([None, input_length])
combined_model.summary()





# fs: sample rate in Hz; xs: the audio PCM samples.
fs, xs = wavfile.read(wav_file_path)

if len(xs) >= input_length:
    xs = xs[:input_length]
else:
    raise ValueError("Audio from .wav file is too short")


input_tensor = tf.constant(xs, shape=(1, input_length), dtype=tf.float32) / 32768.0
# The model outputs the probabilties for the classes (`probs`).
probs = combined_model.predict(input_tensor)

# Read class labels of the model.
metadata_json_path = '/tmp/tfjs-sc-model/metadata.json'

with open(metadata_json_path, 'r') as f:
    metadata = json.load(f)
    class_labels = metadata["words"]

# Get sorted probabilities and their corresponding class labels.
probs_and_labels = list(zip(probs[0].tolist(), class_labels))
# Sort the probabilities in descending order.
probs_and_labels = sorted(probs_and_labels, key=lambda x: -x[0])
probs_and_labels
# len(probs_and_labels)

# Print the top-5 labels:
print('top-5 class probabilities:')
for i in range(5):
    prob, label = probs_and_labels[i]
    print('%20s: %.4e' % (label, prob))