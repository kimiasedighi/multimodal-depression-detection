# resres_datasets.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class PoseAsImageDataset(Dataset):
    """
    Loads .pt pose tensors [3,T,J] and turns them into [3,224,224] for ResNet.
    Expects keys in each .pt:
      - data: [3,T,J]
      - label_n: int (0/1/2)
      - label_p: int (0/1/2)
      - subject_id: str
    """
    def __init__(self, data_dir, key_name="body"):
        self.paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pt")
        ])
        self.key_name = key_name
        if not self.paths:
            raise RuntimeError(f"No .pt files found in {data_dir}")

    def __len__(self):
        return len(self.paths)

    def _pose_to_image(self, pose_ctj: torch.Tensor) -> torch.Tensor:
        x = pose_ctj.float()  # [3,T,J]
        mean = x.mean(dim=(1, 2), keepdim=True)
        std  = x.std(dim=(1, 2), keepdim=True) + 1e-6
        x = (x - mean) / std
        x = F.interpolate(
            x.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)  # [3,224,224]
        return x

    def __getitem__(self, idx):
        path = self.paths[idx]
        item = torch.load(path, map_location="cpu")

        img = self._pose_to_image(item["data"])
        return {
            self.key_name: img,
            "label_n": int(item["label_n"]),
            "label_p": int(item["label_p"]),
            "subject_id": str(item["subject_id"]),
            "name": os.path.basename(path),
        }


class FusionPoseDataset(Dataset):
    """
    Paired body+face dataset, matching by identical basename (excluding .pt).
    Expects same keys in both body and face .pt files:
      - data, label_n, label_p, subject_id
    """
    def __init__(self, body_dir, face_dir):
        self.pairs = []
        body_files = [f for f in os.listdir(body_dir) if f.endswith(".pt")]
        body_files = sorted(body_files)

        for bf in body_files:
            stem = os.path.splitext(bf)[0]
            bpath = os.path.join(body_dir, bf)
            fpath = os.path.join(face_dir, stem + ".pt")
            if os.path.exists(fpath):
                self.pairs.append((bpath, fpath))

        if not self.pairs:
            raise RuntimeError("No paired body/face .pt files found.")

    def __len__(self):
        return len(self.pairs)

    def _pose_to_image(self, pose_ctj: torch.Tensor) -> torch.Tensor:
        x = pose_ctj.float()  # [3,T,J] or [3,H,W]
        if x.dim() == 3 and (x.shape[-1] != 224 or x.shape[-2] != 224):
            # If it's [3,T,J] or not 224x224, resize to 224x224
            x = F.interpolate(
                x.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

        mean = x.mean(dim=(1, 2), keepdim=True)
        std  = x.std(dim=(1, 2), keepdim=True) + 1e-6
        x = (x - mean) / std
        return x

    def __getitem__(self, idx):
        bpath, fpath = self.pairs[idx]
        b = torch.load(bpath, map_location="cpu")
        f = torch.load(fpath, map_location="cpu")

        body_img = self._pose_to_image(b["data"])
        face_img = self._pose_to_image(f["data"])

        # Labels should match (coming from same trial)
        return {
            "body": body_img,
            "face": face_img,
            "label_n": int(b["label_n"]),
            "label_p": int(b["label_p"]),
            "subject_id": str(b["subject_id"]),
            "name": os.path.basename(bpath),
        }
