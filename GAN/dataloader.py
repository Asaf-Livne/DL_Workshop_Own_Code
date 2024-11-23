import os
import torchaudio
import torch
from torch.utils.data import Dataset

class PairedWavDataset(Dataset):
    def __init__(self, lq_folder, hq_folder, chunk_size=16000):
        self.lq_folder = lq_folder
        self.hq_folder = hq_folder
        self.chunk_size = chunk_size
        self.lq_chunks = []
        self.hq_chunks = []
  
        
        self.files = [f for f in os.listdir(lq_folder) if f.endswith('.wav')]
        for f in self.files:
            lq_file, _ = torchaudio.load(os.path.join(lq_folder, f))
            hq_file, _ = torchaudio.load(os.path.join(hq_folder, f))
            length = len(lq_file[0]) // chunk_size
            assert len(hq_file[0]) // chunk_size == length, "LQ and HQ files have different lengths!"
            lq_file = lq_file[0] + lq_file[1]
            hq_file = hq_file[0] + hq_file[1]
            for i in range(length):
                self.lq_chunks.append(lq_file[i*chunk_size:(i+1)*chunk_size])
                self.hq_chunks.append(hq_file[i*chunk_size:(i+1)*chunk_size])

        
    def __len__(self):
        return len(self.lq_chunks)

    def __getitem__(self, idx):
        return self.lq_chunks[idx].unsqueeze(0), self.hq_chunks[idx].unsqueeze(0)
