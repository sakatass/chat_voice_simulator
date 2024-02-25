import soundcard as sc
import soundfile as sf
import sounddevice as sd
import base64
import wave
import io
import marshal
import numpy as np

OUTPUT_FILE_NAME = "out.wav"    # file name.
SAMPLE_RATE = 16000              # [Hz]. sampling rate.
RECORD_SEC = 15                  # [sec]. duration recording audio.

def record_audio(OUTPUT_FILE_NAME = "out.wav", SAMPLE_RATE = 16000, RECORD_SEC = 15):
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
        # record audio with loopback from default speaker.
        data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
        
        sf.write(file=OUTPUT_FILE_NAME, data=data[:, 0], samplerate=SAMPLE_RATE)

def record_audio_CODER(SAMPLE_RATE = 16000, RECORD_SEC = 15):
    with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
        # record audio with loopback from default speaker.
        data = mic.record(numframes=SAMPLE_RATE*RECORD_SEC)
        print(type(data))
        print(data.shape)
        bytecode = base64.b64encode(data.tobytes()).decode('utf-8')
        dtype = str(data.dtype)
        original_shape = data.shape
        
        return bytecode, dtype, original_shape

def record_microphone(SAMPLE_RATE=SAMPLE_RATE, RECORD_SEC=RECORD_SEC):
    print('Recording...')
    myrecording = sd.rec(int(RECORD_SEC * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=2)
    sd.wait()
    print('Recording complete.')
    bytecode = base64.b64encode(myrecording.tobytes()).decode('utf-8')
    dtype = str(myrecording.dtype)
    original_shape = myrecording.shape
    
    return bytecode, dtype, original_shape
