# 물끓는 소리나 불타는 소리 인식하기.   
    1. 일단 링크의 에제 돌려보기.
    2. 물소리 불타는 소리 모으기
    3. 학습시키기
    4. 배포하기-안드로이드 
##
    1.13.1 안드(라이브러리 1.13.1) 성공

###
    유튜브 다운로드 MP3 https://youtube-cutter.org/ 

# [예제]( https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md    )   
    tensorflow 1.13.1
# [kreas-lite-js]( https://github.com/tensorflow/tfjs-models/blob/master/speech-commands/training/browser-fft/training_custom_audio_model_in_python.ipynb   )   

# 설치
    ```
    # 설치
    1.13.1
    git clone https://github.com/tensorflow/tensorflow.git --branch r1.13 --single-branch --depth 1
    docker build --pull --rm -f "dockerfile" -t tf-audio:latest .
    docker run  -d  -it --runtime=nvidia  --name audio -v ~/workspace/AudioRecognition:/workspace/AudioRecognition: -p 8888:8888 -p 6006:6006 tf-audio:latest

    # 훈련
    python /workspace/AudioRecognition/tensorflow/tensorflow/examples/speech_commands/train.py \
        --data_dir=/workspace/AudioRecognition/dataset \
        --wanted_words=bubbling \
        --summaries_dir=/workspace/AudioRecognition/output/logs \
        --start_checkpoint=/workspace/AudioRecognition/output/train/conv.ckpt-6700 \
        --train_dir=/workspace/AudioRecognition/output/train 


    # tensorboard
    tensorboard --logdir /workspace/AudioRecognition/output/logs

    # Training Finished
    python /workspace/AudioRecognition/tensorflow/tensorflow/examples/speech_commands/freeze.py \
        --wanted_words=bubbling \
        --start_checkpoint=/workspace/AudioRecognition/output/train/conv.ckpt-6700 \
        --output_file=/workspace/AudioRecognition/output/graph/my_frozen_graph_6700.pb

        python /workspace/AudioRecognition/tensorflow/tensorflow/examples/speech_commands/label_wav.py \
        --graph=/workspace/AudioRecognition/output/graph/my_frozen_graph_6700.pb \
        --labels=/workspace/AudioRecognition/output/train/conv_labels.txt \
        --wav=/workspace/AudioRecognition/00.wav

        python /workspace/AudioRecognition/tensorflow/tensorflow/examples/speech_commands/label_wav.py \
        --graph=/workspace/AudioRecognition/output/graph/my_frozen_graph.pb \
        --labels=/workspace/AudioRecognition/output/train/conv_labels.txt \
        --wav=/workspace/AudioRecognition/dataset/dog/fe1916ba_nohash_1.wav

    ```

    ```
    js- 2.3.1
    cd ~/worksapce/AudioRecognition 
    docker build -f "dockerfile-2.3" -t tf-audio:latest .
    docker run  -d  -it --runtime=nvidia   --name tf-audio -v  $(pwd):/tf  -p 8888:8888 -p 6006:6006 tf-audio:latest 
    ```
# 실행
## container
### tensorboard
    tensorboard --logdir /workspace/AudioRecognition/output/logs

rm -rf /tmp/speech_commands_v0.02/bubbling
mkdir /tmp/speech_commands_v0.02/bubbling
 cp -r ./dataset/bubbling/sound* /tmp/speech_commands_v0.02/bubbling
    
```

# speaker_model.tflite <== 케라스 예제 
# ======================================
# [{'name': 'input', 'index': 0, 'shape': array([   1, 8000,    1], dtype=int32), 'shape_signature': array([  -1, 8000,    1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
# ======================================
# [{'name': 'Identity', 'index': 125, 'shape': array([1, 1], dtype=int32), 'shape_signature': array([-1,  1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]

# conv_actions_frozen <== 안드로이드 다운 모델
# ======================================
# [{'name': 'decoded_sample_data', 'index': 14, 'shape': array([16000,     1], dtype=int32), 'shape_signature': array([16000,     1], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}, {'name': 'decoded_sample_data:1', 'index': 15, 'shape': array([1], dtype=int32), 'shape_signature': array([1], dtype=int32), 'dtype': <class 'numpy.int32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
# ======================================
# [{'name': 'labels_softmax', 'index': 16, 'shape': array([ 1, 12], dtype=int32), 'shape_signature': array([ 1, 12], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]

my_model <== js 예제
======================================
[{'name': 'audio_preproc_input', 'index': 0, 'shape': array([    1, 44032], dtype=int32), 'shape_signature': array([   -1, 44032], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]
======================================
[{'name': 'Identity', 'index': 72, 'shape': array([1, 2], dtype=int32), 'shape_signature': array([-1,  2], dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0), 'quantization_parameters': {'scales': array([], dtype=float32), 'zero_points': array([], dtype=int32), 'quantized_dimension': 0}, 'sparsity_parameters': {}}]



```