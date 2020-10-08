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
    cd ~/worksapce/AudioRecognition 
    docker build -f "dockerfile" -t tf-audio:latest .
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