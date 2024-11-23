from imports import *
from misc import log_mel_spectrogram, mel_spectrogram

def split_data(lq_data_path, hq_data_path, cfg):
    # Ensure the chunk directories exist
    os.makedirs(os.path.join(lq_data_path, 'Chunks'), exist_ok=True)
    os.makedirs(os.path.join(hq_data_path, 'Chunks'), exist_ok=True)
    
    # Get list of LQ and HQ files, excluding folders
    lq_files = [f for f in os.listdir(lq_data_path) if os.path.isfile(os.path.join(lq_data_path, f))]
    hq_files = [f for f in os.listdir(hq_data_path) if os.path.isfile(os.path.join(hq_data_path, f))]

    n = 1
    j = 0
    for lq_file, hq_file in zip(lq_files, hq_files):
        print(f'Processing the {n}th file')
        n += 1
        # Load LQ and HQ audio
        if not lq_file.endswith('.wav'):
            continue
        lq, _ = torchaudio.load(os.path.join(lq_data_path, lq_file))
        hq, _ = torchaudio.load(os.path.join(hq_data_path, hq_file))
        
        # Convert to mono by averaging channels
        lq = torch.mean(lq, dim=0, keepdim=True)
        hq = torch.mean(hq, dim=0, keepdim=True)


        # Split into segment_size sample chunks with a hop_length sample overlap
        length = len(hq[0])
        for i in range(0, length, cfg['hop_length']):
            if i + cfg['segment_size'] > length:
                break
            
            # Extract chunks
            hq_chunk = hq[:, i:i+cfg['segment_size']]
            lq_chunk = lq[:, i:i+cfg['segment_size']]            
            # Compute mel spectrogram for the LQ chunk
            #lq_chunk_mel = log_mel_spectrogram(cfg, lq_chunk)
            #spec = log_mel_spectrogram(cfg, lq_chunk)
            spec = mel_spectrogram(cfg, lq_chunk)
            
            # Save the HQ chunk as a .wav file
            torchaudio.save(os.path.join(hq_data_path, 'Chunks', f'{j}.wav'), hq_chunk, cfg['sample_rate'])
            torchaudio.save(os.path.join(lq_data_path, 'Chunks', f'{j}.wav'), lq_chunk, cfg['sample_rate'])
            
            # Save the LQ mel spectrogram as a .pt file
            torch.save(spec, os.path.join(lq_data_path, 'Chunks', f'{j}.pt'))
            
            j += 1
    return

# Config for mel-spectrogram
cfg = {
    'segment_size': 16384,
    'sample_rate': 44100,
    'n_fft': 4096,
    'win_length': 1024,
    'hop_length': 520,
    'n_mels': 256,
    'center': True,
    'device': 'cuda'
}

split_data('Data/LQ/24/test/', 'Data/HQ Light/test/', cfg)
