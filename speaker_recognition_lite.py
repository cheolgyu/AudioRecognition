
import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow import keras
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


from pathlib import Path
from IPython.display import display, Audio

from tensorflow import keras 

# speaker_model.tflite
# ======================================
# [{'name': 'input', 'index': 0, 'shape': array([   1, 8000,    1], dtype=int32), 'shape_signature': array([  -1, 8000,    1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
# ======================================
# [{'name': 'Identity', 'index': 125, 'shape': array([1, 1], dtype=int32), 'shape_signature': array([-1,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
def speaker_model():
  model = keras.models.load_model('speaker_model.h5')
  tflite_output_path = 'speaker_model.tflite'
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS
  ]
  with open(tflite_output_path, 'wb') as f:
    f.write(converter.convert())
  print("Saved tflite file at: %s" % tflite_output_path)


  tflite_model =   converter.convert()
  # Run the model with TensorFlow Lite
  interpreter = tf.lite.Interpreter(model_content=tflite_model)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  print("======================================")
  print(input_details)
  print("======================================")
  print(output_details)

# conv_actions_frozen
# ======================================
# [{'name': 'decoded_sample_data', 'index': 14, 'shape': array([16000,     1], dtype=int32), 'shape_signature': array([16000,     1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'decoded_sample_data:1', 'index': 15, 'shape': array([1], dtype=int32), 'shape_signature': array([1], dtype=int32), 'dtype': <class 'numpy.int32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
# ======================================
# [{'name': 'labels_softmax', 'index': 16, 'shape': array([ 1, 12], dtype=int32), 'shape_signature': array([ 1, 12], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]

def conv_actions_frozen():
  tflite_path = 'conv_actions_frozen.tflite'
  interpreter = tf.lite.Interpreter(model_path=tflite_path)
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  print("======================================")
  print(input_details)
  print("======================================")
  print(output_details)

conv_actions_frozen()  