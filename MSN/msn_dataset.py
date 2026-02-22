# msn_dataset.py

import os
import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")]

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        item = torch.load(self.data_files[idx])
        return item["data"], item["label"]
