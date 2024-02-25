#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install -q -U google-generativeai')
get_ipython().system('pip install flask-ngrok')
get_ipython().system('pip install pyngrok')
get_ipython().system('pip install transformers huggingsound')
get_ipython().system('pip install -q --upgrade transformers accelerate')
get_ipython().system('pip install -q -U google-generativeai')


# In[15]:


from pyngrok import ngrok
publick_url = ngrok.connect(5000).public_url


# In[ ]:


from flask import Flask, request, jsonify
import os
import marshal
import soundfile as sf
import uuid
import base64
import numpy as np
from transformers import VitsModel, AutoTokenizer
import torch
from scipy.io.wavfile import write
from huggingsound import SpeechRecognitionModel
from pyngrok import ngrok
from flask_ngrok import run_with_ngrok
import google.generativeai as genai

ngrok.set_auth_token('************************')

TTS_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
TTS_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

STT_model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")

history_human = []
history_ai = []

gemini_chat_prompt = f"""
    You are an old friend of mine, today we meet after a long time apart.

    ** Step-by-step instructions:**
    0. The answer should be in one to two sentences.
    1. Listen to the user's question or message and analyze it carefully.
    2. Generate a response that is appropriate to the content of the question and the topic of the conversation. The response should be clear, knowledgeable and informative.
    3. Keep the history of the dialog, relying on previous answers and questions, if any.
    """
history = [
      {
        'role': "user",
        'parts': [{ 'text': gemini_chat_prompt}],
      },
      {
        'role': "model",
        'parts': [{ 'text': "Understood."}],
      } ]

genai.configure(api_key='AIzaSyCa_xSzyPqbDXwPHCox0xz-LoyBFtKzxzQ')
gemini_model = genai.GenerativeModel('gemini-pro')
gemini_long_memory = gemini_model.start_chat(history=history)

if not os.path.exists('input_wavs'):
    # Создать папку, если она не существует
    os.makedirs('input_wavs')
    print(f'Папка input_wavs успешно создана.')
else:
    print(f'Папка input_wavs уже существует.')

def gemini_chat_V2(model, human_text):
    response = model.send_message(human_text)
    print(f'AI message: {response.text}')
    return response.text

def gemini_chat(history_human, history_ai, human_text):
    dialog_history = ''
    prompt_summ = '''
    You're a tool for summarizing dialogue history into two or three sentences. Dialogue record format: AI: <some_text>, H: <some_text>. Your task is to keep the entire narrative of the dialog, but not to go into the details of the lines.

    Dialog history:{dialog_history}
    '''
    #human_text = str(input('Enter a human QA: '))
    history_human.append(human_text)
    for n, elem in enumerate(history_human[-5:]):
        dialog_history += '\nHuman message: ' + elem
        try:
            dialog_history += '\nAI message: ' + history_ai[n]
        except:
            pass

    if len(history_human) + len(history_ai) >= 2:
      print(prompt_summ.format(dialog_history=dialog_history))
      response = gemini_model.generate_content(prompt_summ.format(dialog_history=dialog_history))
      dialog_history = response.text
      print(dialog_history)
    prompt = f"""
    You are an old friend of mine, today we meet after a long time apart.

    ** Step-by-step instructions:**
    1. The answer should be in one to two sentences.
    2. Listen to the user's question or message and analyze it carefully.
    3. Generate a response that is appropriate to the content of the question and the topic of the conversation. The response should be clear, knowledgeable and informative.
    4. Keep the history of the dialog, relying on previous answers and questions, if any.

    **Dialogue History:**
    {dialog_history}
    """
    print(dialog_history)
    response = gemini_model.generate_content(prompt)
    history_ai.append(response.text)
    print(f'AI message: {history_ai[-1]}')
    return history_ai[-1]

def generate_audio(text: str, TTS_model, TTS_tokenizer):
  #TTS_model = VitsModel.from_pretrained("facebook/mms-tts-eng")
  #TTS_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

  #text_example = "some example text in the English language"
  inputs = TTS_tokenizer(text, return_tensors="pt")

  with torch.no_grad():
      output = TTS_model(**inputs).waveform

  audio_data = output.numpy().T

  # Specify the sampling rate
  sampling_rate = 16000

  # Normalize the audio data to the range [-32768, 32767] for 16-bit PCM encoding
  normalized_data = np.int16(audio_data * 32767)

  return normalized_data

def generate_random_name():
    # Генерация UUID-4 (случайный UUID)
    random_name = str(uuid.uuid4())

    return random_name

def decode_bytecode(base64_bytecode, dtype, original_shape):
    # Decode base64-encoded bytecode
    decoded_bytes = base64.b64decode(base64_bytecode)

    # Convert the decoded bytes to a NumPy array
    decoded_array = np.frombuffer(decoded_bytes, dtype=np.dtype(dtype))
    decoded_array = decoded_array.reshape(original_shape)

    return decoded_array


app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    return 'Hello, World!'

@app.route('/voice_response', methods=['POST'])
def voice_response():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Check if the 'bytecode' key is present in the JSON data
        if 'bytecode' not in data or 'dtype' not in data:
            return jsonify({'error': 'Bytecode or dtype not found in the request'}), 400

        # Extract the bytecode from the JSON data
        bytecode = data['bytecode']
        dtype = data['dtype']
        original_shape = data['original_shape']
        audio_data = decode_bytecode(bytecode, dtype, original_shape)


        # save .wav file to dir input_wavs
        fname = generate_random_name() + '.wav'
        fpath = os.path.join('input_wavs', fname)
        sf.write(file=fpath, data=audio_data[:, 0], samplerate=16000)

        # get transcribe
        result = STT_model.transcribe([fpath])
        print(result[0]['transcription'])

        # get gemini response
        ai_message = gemini_chat_V2(gemini_long_memory, result[0]['transcription'])

        # generate speech
        wav_data = generate_audio(ai_message, TTS_model, TTS_tokenizer)
        bytecode = base64.b64encode(wav_data.tobytes()).decode('utf-8')

        # Return a success response
        print(wav_data.shape, str(wav_data.dtype))
        return jsonify({'wav_bytecode': bytecode, 'shape': wav_data.shape, 'dtype': str(wav_data.dtype)}), 200

    except Exception as e:
        # Handle any exceptions that may occur
        return jsonify({'error': str(e)}), 500

@app.route('/text_response', methods=['POST'])
def text_response():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Check if the 'bytecode' key is present in the JSON data
        if 'human_message' not in data:
            return jsonify({'error': 'human_message not found in the request'}), 400

        # Extract the bytecode from the JSON data
        human_message = data['human_message']

        # get gemini response
        ai_message = gemini_chat_V2(gemini_long_memory, human_message)

        # Return a success response
        return jsonify({'response': ai_message}), 200

    except Exception as e:
        # Handle any exceptions that may occur
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    run_with_ngrok(app)
    app.run()

