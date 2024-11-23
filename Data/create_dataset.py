import sys
import os
sys.path.append(os.path.abspath('../'))
import subprocess
import torchaudio

def create_dataset(quality, train='train'):
    for file in os.listdir(f'Data/HQ Light/{train}/'):
        subprocess.run(['python3', '-m', 'encodec', '-b', str(quality), '--hq', '-f',  f'Data/HQ Light/{train}/{file}', f'Data/LQ/{quality}/{train}/{file}'], stdout=subprocess.PIPE)
        print(f'Encoded {file} with quality {quality}')
        audio, sr = torchaudio.load(f'Data/LQ/{quality}/{train}/{file}')
        # resample at 44.1 kHz
        audio = torchaudio.transforms.Resample(sr, 44100)(audio)
        torchaudio.save(f'Data/LQ/{quality}/{train}/{file}', audio, 44100)
        print(f'Resampled {file} to 44100 kHz')
    print('Done!')


create_dataset(24, 'test')
    



        
