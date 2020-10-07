from os import path
from pydub.utils import mediainfo
from pydub import AudioSegment
import soundfile as sf
# https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonAudio.html#PySoundFile
#https://github.com/jiaaro/pydub/blob/master/API.markdown

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
        
        for i, chunk in enumerate(sound[::1000]):
            with open("dataset/bubbling/sound_+"+str(x)+"_-%s.mp3" % i, "wb") as f:
                chunk.export(f, format="wav")    
r2()            