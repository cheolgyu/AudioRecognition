from os import path
from pydub.utils import mediainfo
from pydub import AudioSegment
import soundfile as sf
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonAudio.html#PySoundFile
#https://github.com/jiaaro/pydub/blob/master/API.markdown
import glob

# files

def r1():
    for x in range(5):
        src = "dataset/bubbling/0"+str(x)+".wav"
        dst = "dataset/bubbling/0"+str(x)+".wav"

        # convert wav to mp3                                                            
        sound = AudioSegment.from_wav(src)
        sound.export(dst, format="wav")
        info = mediainfo(dst)
        print(info)

def r2():
    for x in range(5):
        src = "dataset/bubbling/0"+str(x)+".wav"
        dst = "dataset/bubbling/0"+str(x)+".wav"    
        sound = AudioSegment.from_file(src, format="wav")
        sound.set_channels(1)
        sound = sound.set_frame_rate(16000)
        
        for i, chunk in enumerate(sound[::100]):
            with open("dataset/bubbling/sound"+str(x)+"%s.wav" % i, "wb") as f:
                chunk.export(f, format="wav")    
def r3():
    src = "dataset/bubbling/*.wav"
    file_list = glob.glob(src)   
    for  file in file_list:
        print(file)
        sound = AudioSegment.from_file(file, format="wav")   
        # 44100
        # 16000
        sound.set_channels(1)
        sound = sound.set_frame_rate(16000)
        sound.export(file, format="wav")
def r4():
    from scipy.io.wavfile import read as read_wav
    import os
    sampling_rate, data=read_wav("dataset/bubbling/00.wav") # enter your filename
    print(sampling_rate)
r2()
# for f in *.mp3; do mv -- "$f" "${f%.prog}.wav" 
# rename 's/.mp3/.wav/' *.mp3