from utils import decode_bytecode, wite_gen_wav, generate_random_name, voice_chat, text_chat


simulator_mode = ''
while simulator_mode not in ['chat', 'voice']:
    simulator_mode = str(input('Select simulator mode (voice or chat): '))

# set the api url
API_URL = str(input('Paste the API link: '))

# running the selected simulator
if simulator_mode == 'chat':
    text_chat(API_URL)
elif simulator_mode == 'voice':
    voice_chat(API_URL)
