# 물끓는 소리나 불타는 소리 인식하기.   

    1. 일단 링크의 에제 돌려보기.
    2. 물소리 불타는 소리 모으기
    3. 학습시키기
    4. 배포하기-안드로이드 

# [예제]( https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/sequences/audio_recognition.md    )   
    tensorflow 1.15 

# 설치
    ```
    cd ~/worksapce/AudioRecognition 
    docker build --pull --rm -f "dockerfile" -t audio-recognition:latest "."
    docker run  -d  -it --runtime=nvidia  --name audio -v ~/workspace/AudioRecognition:/tf/notebooks -p 8888:8888 -p 6006:6006 audio-recognition:latest 
    

    ```
# 실행
## container
### train
    python /tensorflow/tensorflow/examples/speech_commands/train.py
    python /tensorflow/tensorflow/examples/speech_commands/train.py \
    --start_checkpoint=/tmp/speech_commands_train/conv.ckpt-17300 
### tensorboard
    tensorboard --logdir /tmp/retrain_logs

### Training Finished
    python /tensorflow/tensorflow/examples/speech_commands/freeze.py \
    --start_checkpoint=/tmp/speech_commands_train/conv.ckpt-18000 \
    --output_file=/tmp/my_frozen_graph.pb

    python /tf/notebooks/label_wav.py \
    --graph=/tmp/my_frozen_graph.pb \
    --labels=/tmp/speech_commands_train/conv_labels.txt \
    --wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav
    

python tensorflow/examples/speech_commands/train.py