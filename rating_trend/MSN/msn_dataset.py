# msn_dataset.py

import os
import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pt")
        ])
        if not self.files:
            raise RuntimeError(f"No .pt files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = torch.load(self.files[idx])
        return (
            item["data"],          # [C,T,J]
            item["label_n"],       # int {0,1,2}
            item["label_p"],       # int {0,1,2}
            item["subject_id"],    # str
        )
