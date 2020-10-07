# https://github.com/tensorflow/docs/blob/r1.13/site/en/api_docs/python/tf/lite/TFLiteConverter.md
# # Converting a GraphDef from session.
# converter = lite.TFLiteConverter.from_session(sess, in_tensors, out_tensors)
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)

# # Converting a GraphDef from file.
# converter = lite.TFLiteConverter.from_frozen_graph(
#   graph_def_file, input_arrays, output_arrays)
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)

# # Converting a SavedModel.
# converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
# tflite_model = converter.convert()

# # Converting a tf.keras model.
# converter = lite.TFLiteConverter.from_keras_model_file(keras_model)
# tflite_model = converter.convert()

import tensorflow as tf
import tensorflow.lite as lite

graph_def_file="/workspace/AudioRecognition/output/graph/my_frozen_graph_1_13_1.pb"
input_arrays=""
output_arrays=""
converted_model="/workspace/AudioRecognition/output/tflite/my_frozen_graph_1_13_1.tflite"

gf = tf.GraphDef()   
m_file = open(graph_def_file,'rb')
gf.ParseFromString(m_file.read())

print( "==============================")
for n in gf.node:
    print( n.name+" , " )

print( "==============================")
tensor = n.op
print( tensor)
print( "==============================")
converter = lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open(converted_model, "wb").write(tflite_model)