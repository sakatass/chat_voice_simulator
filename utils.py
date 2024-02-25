import numpy as np
from scipy.io.wavfile import write
import os
from capture import record_audio_CODER, record_microphone
import soundcard as sc
import soundfile as sf
import os 
import marshal
import requests
from datetime import datetime
import winsound
import uuid
import base64

def decode_bytecode(base64_bytecode, dtype, original_shape):
    # Decode base64-encoded bytecode
    decoded_bytes = base64.b64decode(base64_bytecode)

    # Convert the decoded bytes to a NumPy array
    decoded_array = np.frombuffer(decoded_bytes, dtype=np.dtype(dtype))
    decoded_array = decoded_array.reshape(original_shape)

    return decoded_array

def wite_gen_wav(normalized_data, file_dir, rate=16000): # file_dir with .wav
  write(file_dir, rate, normalized_data)

def generate_random_name():
    # Генерация UUID-4 (случайный UUID)
    random_name = str(uuid.uuid4())
    return random_name

def voice_chat(API_URL):
    while True:
        bytecode, dtype, original_shape = record_microphone(RECORD_SEC=6)
        r_data = {'bytecode': bytecode, 'dtype': dtype, 'original_shape': original_shape,
                  'ngrok-skip-browser-warning': 'asd21'}
        
        # send requests
        response = requests.post(API_URL + '/voice_response', json=r_data)
        if response.status_code == 200:
            
            # get response params
            wav_bytecode = response.json().get('wav_bytecode', '')
            shape = response.json().get('shape', '')
            dtype = response.json().get('dtype', '')
            wav_data = decode_bytecode(wav_bytecode, dtype, shape)

            # write a .wav file
            f_name = generate_random_name()
            file_dir = os.path.join('wav_sessions',f'{f_name}.wav')
            wite_gen_wav(wav_data, file_dir)

            # play .wav file
            winsound.PlaySound(file_dir, winsound.SND_FILENAME)
            
        else:
            print("POST request failed with status code:", response.status_code)

def text_chat(API_URL):
    while True:

        # User write a some text
        human_message = str(input('Write a some text: '))

        # send requests
        response = requests.post(API_URL + '/text_response', json={'human_message': human_message})
        if response.status_code == 200:
            
            # get response params
            ai_message = response.json().get('response', '')
            
            # show a AI message
            print('AI: {}'.format(ai_message))
            
        else:
            print("POST request failed with status code:", response.status_code)
