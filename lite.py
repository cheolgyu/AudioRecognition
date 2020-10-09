import tensorflow as tf
print(tf.__version__)
def detail(interpreter):
  interpreter.allocate_tensors()
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  print("======================================")
  print(input_details)
  print("======================================")
  print(output_details)

def log_tflite():
  frozen_graph_path = 'output/graph/my_frozen_graph.pb'
  tflite_path = 'output/tflite/my_frozen_graph.tflite'
  interpreter = tf.lite.Interpreter(model_path=tflite_path)
  detail(interpreter)

def tolite_from_frozen_graph():
    graph_def_file = 'output/graph/my_frozen_graph_6700.pb'
    tflite_path = 'output/lite/my_frozen_graph_6700.pb'
    input_arrays = ['wav_data']
    output_arrays = ['labels_softmax']
#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/python_api.md#complex-examples-
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
        graph_def_file=graph_def_file,
        input_arrays=input_arrays,
        input_shapes={'wav_data' : [None,3920]},
        output_arrays=output_arrays,
    )
    tflite_model = converter.convert()
    open(tflite_path, "wb").write(tflite_model)

def from_saved_model():
  # Convert the model
  saved_model_dir='output/train'
  tflite_path = 'output/tflite/my_frozen_graph.tflite'
  converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model(saved_model_dir) # path to the SavedModel directory
  tflite_model = converter.convert()

  # Save the model.
  open(tflite_path, "wb").write(tflite_model)
  


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def tolite_from_session():
    graph_def_file = 'output/graph/my_frozen_graph.pb'
    tflite_path = 'output/tflite/my_frozen_graph.tflite'
    in_tensors = ['wav_data']
    out_tensors = ['labels_softmax']

    load_graph(graph_def_file)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    output_layer_name="wav_data:0"
    input_layer_name="labels_softmax:0"

    with tf.compat.v1.Session(config=config) as sess:
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        #predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})


        out_tensors = softmax_tensor

        converter = tf.lite.TFLiteConverter.from_session(sess, in_tensors, out_tensors)
        tflite_model = converter.convert()
        open(tflite_path, "wb").write(tflite_model)


def log_pb():
    graph_def_file = 'output/graph/my_frozen_graph.pb'
    import tensorflow as tf
    gf = tf.GraphDef()   
    m_file = open(graph_def_file,'rb')
    gf.ParseFromString(m_file.read())

    with open('somefile.txt', 'a') as the_file:
        for n in gf.node:
            the_file.write(n.name+'\n')

    file = open('somefile.txt','r')
    data = file.readlines()
    print("output name = ")
    print(data[len(data)-1])

    print("Input name = ")
    file.seek ( 0 )
    print(file.readline())



tolite_from_frozen_graph()    