from imports import *
from misc import log_mel_spectrogram, mel_spectrogram

class HFDataSet(torch.utils.data.Dataset):
    def __init__(self, lq_data_path, hq_data_path, cfg):
        self.lq_data_path = lq_data_path
        self.hq_data_path = hq_data_path
        self.hq_data, self.lq_data = self.load_data(hq_data_path, lq_data_path)
        self.cfg = cfg
    
    
    def load_data(self, hq_data_path, lq_data_path):
        lq_data = []
        hq_data = []
        length = len(os.listdir(hq_data_path))
        for i in range(3000):
            file = f'{i}.wav'
            hq, _ = torchaudio.load(os.path.join(hq_data_path, file))
            hq_data.append(hq)
            lq, _ = torchaudio.load(os.path.join(lq_data_path, file))
            lq_data.append(lq)
        return hq_data, lq_data
            
    
    def __len__(self):
        return len(self.hq_data)
    
    def __getitem__(self, idx):
        audio = self.hq_data[idx]
        lq_audio = self.lq_data[idx]
        spec = mel_spectrogram(self.cfg, lq_audio)
        return spec, audio
    
class HFDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        super(HFDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)