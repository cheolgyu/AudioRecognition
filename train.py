#!/usr/bin/python
# -*- coding: utf-8 -*-

import glob
import json
import os
import random
import librosa 
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
import tensorflow as tf
import tensorflowjs as tfjs
import tqdm

import json
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

print(tf.__version__)
print(tfjs.__version__)


preproc_model_path = '/tmp/tfjs-sc-model/sc_preproc_model'
preproc_model = tf.keras.models.load_model(preproc_model_path)
preproc_model.summary()
preproc_model.input_shape



# Create some constants to be used later.

# Target sampling rate. It is required by the audio preprocessing model.
TARGET_SAMPLE_RATE = 44100
# The specific audio tensor length expected by the preprocessing model.
EXPECTED_WAVEFORM_LEN = preproc_model.input_shape[-1]

# Where the Speech Commands v0.02 dataset has been downloaded.
DATA_ROOT = "/tmp/speech_commands_v0.02"

WORDS = ("_background_noise_snippets_", "bubbling")



noise_wav_paths = glob.glob(os.path.join(DATA_ROOT, "_background_noise_", "*.wav"))
snippets_dir = os.path.join(DATA_ROOT, "_background_noise_snippets_")
os.makedirs(snippets_dir, exist_ok=True)


def extract_snippets(wav_path, snippet_duration_sec=1.0):
  basename = os.path.basename(os.path.splitext(wav_path)[0])
  sample_rate, xs = wavfile.read(wav_path)
  assert xs.dtype == np.int16
  n_samples_per_snippet = int(snippet_duration_sec * sample_rate)
  i = 0
  while i + n_samples_per_snippet < len(xs):
    snippet_wav_path = os.path.join(snippets_dir, "%s_%.5d.wav" % (basename, i))
    snippet = xs[i : i + n_samples_per_snippet].astype(np.int16)
    wavfile.write(snippet_wav_path, sample_rate, snippet)
    i += n_samples_per_snippet

for noise_wav_path in noise_wav_paths:
  print("Extracting snippets from %s..." % noise_wav_path)
  extract_snippets(noise_wav_path, snippet_duration_sec=1.0)

def resample_wavs(dir_path, target_sample_rate=44100):
  """Resample the .wav files in an input directory to given sampling rate.
  
  The resampled waveforms are written to .wav files in the same directory with
  file names that ends in "_44100hz.wav".

  44100 Hz is the sample rate required by the preprocessing model. It is also
  the most widely supported sample rate among web browsers and mobile devices.
  For example, see:
  https://developer.mozilla.org/en-US/docs/Web/API/AudioContextOptions/sampleRate
  https://developer.android.com/ndk/guides/audio/sampling-audio

  Args:
    dir_path: Path to a directory that contains .wav files.
    target_sapmle_rate: Target sampling rate in Hz.
  """
  wav_paths = glob.glob(os.path.join(dir_path, "*.wav"))
  resampled_suffix = "_%shz.wav" % target_sample_rate
  for i, wav_path in tqdm.tqdm(enumerate(wav_paths)):
    if wav_path.endswith(resampled_suffix):
      continue
    sample_rate, xs = wavfile.read(wav_path)
    xs = xs.astype(np.float32)
    xs = librosa.resample(xs, sample_rate, TARGET_SAMPLE_RATE).astype(np.int16)
    resampled_path = os.path.splitext(wav_path)[0] + resampled_suffix
    wavfile.write(resampled_path, target_sample_rate, xs)


for word in WORDS:
  word_dir = os.path.join(DATA_ROOT, word)
  assert os.path.isdir(word_dir)
  #resample_wavs(word_dir, target_sample_rate=TARGET_SAMPLE_RATE)


@tf.function
def read_wav(filepath):
  file_contents = tf.io.read_file(filepath)
  return tf.expand_dims(tf.squeeze(tf.audio.decode_wav(
      file_contents, 
      desired_channels=-1,
      desired_samples=TARGET_SAMPLE_RATE).audio, axis=-1), 0)


@tf.function
def filter_by_waveform_length(waveform, label):
  return tf.size(waveform) > EXPECTED_WAVEFORM_LEN


@tf.function
def crop_and_convert_to_spectrogram(waveform, label):
  cropped = tf.slice(waveform, begin=[0, 0], size=[1, EXPECTED_WAVEFORM_LEN])
  return tf.squeeze(preproc_model(cropped), axis=0), label


@tf.function
def spectrogram_elements_finite(spectrogram, label):
  return tf.math.reduce_all(tf.math.is_finite(spectrogram))


def get_dataset(input_wav_paths, labels):
  """Get a tf.data.Dataset given input .wav files and their labels.

  The returned dataset emits 2-tuples of `(spectrogram, label)`, wherein
  - `spectrogram` is a tensor of dtype tf.float32 and shape [43, 232, 1].
    It is z-normalized (i.e., have a mean of ~0.0 and variance of ~1.0).
  - `label` is a tensor of dtype tf.int32 and shape [] (scalar).
  
  Args:
    input_wav_paths: Input audio .wav file paths as a list of string.
    labels: integer labels (class indices) of the input .wav files. Must have
      the same lengh as `input_wav_paths`.

  Returns:
    A tf.data.Dataset object as described above.
  """
  ds = tf.data.Dataset.from_tensor_slices(input_wav_paths)
  # Read audio waveform from the .wav files.
  ds = ds.map(read_wav)
  ds = tf.data.Dataset.zip((ds, tf.data.Dataset.from_tensor_slices(labels)))
  # Keep only the waveforms longer than `EXPECTED_WAVEFORM_LEN`.
  ds = ds.filter(filter_by_waveform_length)
  # Crop the waveforms to `EXPECTED_WAVEFORM_LEN` and convert them to
  # spectrograms using the preprocessing layer.
  ds = ds.map(crop_and_convert_to_spectrogram)
  # Discard examples that contain infinite or NaN elements.
  ds = ds.filter(spectrogram_elements_finite)
  return ds
  
# input_wav_paths_and_labels = []
# for i, word in enumerate(WORDS):
#   wav_paths = glob.glob(os.path.join(DATA_ROOT, word, "*_%shz.wav" % TARGET_SAMPLE_RATE))
#   print("Found %d examples for class %s" % (len(wav_paths), word))
#   labels = [i] * len(wav_paths)
#   input_wav_paths_and_labels.extend(zip(wav_paths, labels))
# random.shuffle(input_wav_paths_and_labels)
  
# input_wav_paths, labels = ([t[0] for t in input_wav_paths_and_labels],
#                            [t[1] for t in input_wav_paths_and_labels])
# dataset = get_dataset(input_wav_paths, labels)

# # Show some example spectrograms for inspection.
# fig = plt.figure(figsize=(40, 100))
# dataset_iter = iter(dataset)
# num_spectrograms_to_show = 10
# for i in range(num_spectrograms_to_show):
#   ax = fig.add_subplot(1, num_spectrograms_to_show, i + 1)
#   spectrogram, label = next(dataset_iter)
#   spectrogram = spectrogram.numpy()
#   label = label.numpy()
#   plt.imshow(np.flipud(np.squeeze(spectrogram, -1).T), aspect=0.2)
#   ax.set_title("Example of \"%s\"" % WORDS[label])
#   ax.set_xlabel("Time frame #")
#   if i == 0:
#     ax.set_ylabel("Frequency bin #")


input_wav_paths_and_labels = []
for i, word in enumerate(WORDS):
  wav_paths = glob.glob(os.path.join(DATA_ROOT, word, "*_%shz.wav" % TARGET_SAMPLE_RATE))
  print("Found %d examples for class %s" % (len(wav_paths), word))
  labels = [i] * len(wav_paths)
  input_wav_paths_and_labels.extend(zip(wav_paths, labels))
random.shuffle(input_wav_paths_and_labels)
  
input_wav_paths, labels = ([t[0] for t in input_wav_paths_and_labels],
                           [t[1] for t in input_wav_paths_and_labels])
dataset = get_dataset(input_wav_paths, labels)



print(
    "Loading dataset and converting data to numpy arrays. "
    "This may take a few minutes...")
xs_and_ys = list(dataset)
xs = np.stack([item[0] for item in xs_and_ys])
ys = np.stack([item[1] for item in xs_and_ys])
print("Done.")





tfjs_model_json_path = '/tmp/tfjs-sc-model/model.json'

# Load the Speech Commands model. Weights are loaded along with the topology,
# since we train the model from scratch. Instead, we will perform transfer
# learning based on the model.
orig_model = tfjs.converters.load_keras_model(tfjs_model_json_path, load_weights=True)

# Remove the top Dense layer and add a new Dense layer of which the output
# size fits the number of sound classes we care about.
model = tf.keras.Sequential(name="TransferLearnedModel")
for layer in orig_model.layers[:-1]:
  model.add(layer)
model.add(tf.keras.layers.Dense(units=len(WORDS), activation="softmax"))

# Freeze all but the last layer of the model. The last layer will be fine-tuned
# during transfer learning.
for layer in model.layers[:-1]:
  layer.trainable = False
from tensorflow import keras
callbacks = [
    keras.callbacks.ModelCheckpoint(
        # Path where to save the model
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath="checkpoint/mymodel_{epoch}",
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
        period=500
    )
]

model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["acc"])
model.summary()


# Train the model.
history = model.fit(xs, ys, batch_size=256, validation_split=0.3, shuffle=True, epochs=50 , callbacks=callbacks)
model.save('model.h5')


print(history.history)

print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)

# # Convert the model to TensorFlow.js Layers model format.

# tfjs_model_dir = "/tmp/tfjs-model"
# tfjs.converters.save_keras_model(model, tfjs_model_dir)

# # Create the metadata.json file.
# metadata = {"words": ["_background_noise_"] + WORDS[1:], "frameSize": model.input_shape[-2]}
# with open(os.path.join(tfjs_model_dir, "metadata.json"), "w") as f:
#   json.dump(metadata, f)


combined_model = tf.keras.Sequential(name='CombinedModel')
combined_model.add(preproc_model)
combined_model.add(model)
combined_model.build([None, EXPECTED_WAVEFORM_LEN])
combined_model.summary()

tflite_output_path = '/tmp/tfjs-sc-model/combined_model.tflite'
converter = tf.lite.TFLiteConverter.from_keras_model(combined_model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
]
with open(tflite_output_path, 'wb') as f:
    f.write(converter.convert())
print("Saved tflite file at: %s" % tflite_output_path)

