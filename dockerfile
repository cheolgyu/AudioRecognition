
FROM tensorflow/tensorflow:2.3.1-gpu-jupyter
RUN apt-get install -y libsndfile1
RUN mkdir -p /tmp/tfjs-sc-model
RUN curl -o /tmp/tfjs-sc-model/metadata.json -fsSL https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/metadata.json 
RUN curl -o /tmp/tfjs-sc-model/model.json -fsSL https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/model.json
RUN curl -o /tmp/tfjs-sc-model/group1-shard1of2 -fSsL https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/group1-shard1of2
RUN curl -o /tmp/tfjs-sc-model/group1-shard2of2 -fsSL https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.3/browser_fft/18w/group1-shard2of2
RUN curl -o /tmp/tfjs-sc-model/sc_preproc_model.tar.gz -fSsL https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/conversion/sc_preproc_model.tar.gz
RUN cd /tmp/tfjs-sc-model/ && tar xzvf sc_preproc_model.tar.gz

RUN mkdir -p /tmp/speech_commands_v0.02
RUN curl -o /tmp/speech_commands_v0.02/speech_commands_v0.02.tar.gz -fSsL http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
RUN cd  /tmp/speech_commands_v0.02 && tar xzf speech_commands_v0.02.tar.gz

RUN pip install librosa tensorflowjs tqdm
EXPOSE 6006
