from torch.utils.data import Dataset
from typing import List, Callable, Any
from pathlib import Path
import torchaudio
import torch
import os

def list_files_with_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


class EncodecDataset(Dataset):
    
    def __init__(
            self, 
            data_files: List[Path],
            transform: Callable = None, 
            fs: int = 19531, 
            min_dur: int = 5
            ):
        super().__init__()
        """
        Dataset class for Neuralink Compression Challenge
        
        Args:
            data_files: list of paths to data files
            transform: optional transform to be applied to the data
            fs: sampling frequency of the data
            min_dur: minimum duration of the data in seconds
        """
        self.data = []
        self.fs = fs
        for file in data_files:
            self.data.append(torchaudio.load(file)[0].squeeze()[:min_dur*fs])

        self.data = torch.stack(self.data)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].unsqueeze(0)
        if self.transform:
            sample = self.transform(sample)
        return sample 
    

def get_random_chunk(x, chunk_size):
    rand_idx = torch.randint(0, len(x)-chunk_size, size=(1,))
    return x[rand_idx:rand_idx+chunk_size]


def composed_transform(x, transforms):
    for transform in transforms:
        x = transform(x)
    return x