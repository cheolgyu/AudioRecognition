# 물끓는 소리나 불타는 소리 인식하기.   
    1. 일단 링크의 에제 돌려보기.
    2. 물소리 불타는 소리 모으기
    3. 학습시키기
    4. 배포하기-안드로이드 
##
    
    1.13.1 안드(라이브러리 1.13.1) 성공


# [예제]( https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md    )   
    tensorflow 1.15 

# 설치
    ```
    cd ~/worksapce/AudioRecognition 
    git clone https://github.com/tensorflow/tensorflow.git --branch r1.15 --single-branch --depth 1

    docker build --pull --rm -f "dockerfile" -t audio-recognition:latest .
    docker run  -d  -it --runtime=nvidia  --name audio -v ~/workspace/AudioRecognition:/workspace/AudioRecognition: -p 8888:8888 -p 6006:6006 audio-recognition:latest 
    
    유튜브 다운로드 MP3 https://youtube-cutter.org/ 

    ```
# 실행
## container
### train
   python /workspace/AudioRecognition/tensorflow/tensorflow/examples/speech_commands/train.py  \
    --data_dir=/workspace/AudioRecognition/dataset \
    --wanted_words=bubbling \
    --summaries_dir=/workspace/AudioRecognition/output/logs \
    --train_dir=/workspace/AudioRecognition/output/train

### tensorboard
    tensorboard --logdir /workspace/AudioRecognition/output/logs

### Training Finished
    python /workspace/AudioRecognition/tensorflow/tensorflow/examples/speech_commands/freeze.py \
    --start_checkpoint=/workspace/AudioRecognition/output/train/conv.ckpt-18000 \
    --wanted_words=bubbling \
    --output_file=/workspace/AudioRecognition/output/graph/my_frozen_graph.pb

    python /workspace/AudioRecognition/tensorflow/tensorflow/examples/speech_commands/label_wav.py \
    --graph=/workspace/AudioRecognition/output/graph/my_frozen_graph.pb \
    --labels=/workspace/AudioRecognition/output/train/conv_labels.txt \
    --wav=/workspace/AudioRecognition/dataset/bubbling/02.wav

    python /workspace/AudioRecognition/tensorflow/tensorflow/examples/speech_commands/label_wav.py \
    --graph=/workspace/AudioRecognition/output/graph/my_frozen_graph.pb \
    --labels=/workspace/AudioRecognition/output/train/conv_labels.txt \
    --wav=/workspace/AudioRecognition/dataset/dog/fe1916ba_nohash_1.wav
    
