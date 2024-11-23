from imports import *


class HFDataSet(torch.utils.data.Dataset):
    def __init__(self, lq_data_path, hq_data_path):
        self.lq_data_path = lq_data_path
        self.hq_data_path = hq_data_path
        self.lq_specs, self.hq_audio = self.load_data(lq_data_path, hq_data_path)
    
    
    def load_data(self, lq_data_path, hq_data_path):
        lq_data = []
        hq_data = []
        length = len(os.listdir(lq_data_path))
        for i in range(3000):
            file = f'{i}.pt'
            lq = torch.load(os.path.join(lq_data_path, file))
            lq = lq.squeeze(0)
            lq_data.append(lq)
            file = f'{i}.wav'
            hq, _ = torchaudio.load(os.path.join(hq_data_path, file))
            hq_data.append(hq)
        return lq_data, hq_data
            
    
    def __len__(self):
        return len(self.lq_specs)
    
    def __getitem__(self, idx):
        return self.lq_specs[idx], self.hq_audio[idx]
    
class HFDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        super(HFDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)