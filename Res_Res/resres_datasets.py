# resres_datasets.py

import os, torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class PoseAsImageDataset(Dataset):
    """
    Loads your .pt pose tensors [3,T,J] and turns them into [3,224,224] for ResNet.
    """
    def __init__(self, data_dir):
        self.paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pt")])

    def __len__(self): return len(self.paths)

    def _pose_to_image(self, pose_ctj):
        x = pose_ctj.float()  # [3,T,J]
        # standardize per channel
        mean = x.mean(dim=(1,2), keepdim=True)
        std  = x.std(dim=(1,2), keepdim=True) + 1e-6
        x = (x - mean) / std
        # resize (T,J) -> (224,224)
        x = F.interpolate(x.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False).squeeze(0)
        return x  # [3,224,224]

    def __getitem__(self, idx):
        item = torch.load(self.paths[idx], map_location="cpu")
        img  = self._pose_to_image(item["data"])
        label = int(item["label"])
        return {"image": img, "label": label, "name": os.path.basename(self.paths[idx])}

class FusionPoseDataset(Dataset):
    """
    If/when you have matching face tensors (same basenames) in face_dir.
    Face/body 'data' can be [3,224,224] already or [3,T,J] (we'll resize).
    """
    def __init__(self, body_dir, face_dir):
        self.pairs = []
        body_files = [f for f in os.listdir(body_dir) if f.endswith(".pt")]
        for bf in sorted(body_files):
            stem = os.path.splitext(bf)[0]
            bpath = os.path.join(body_dir, bf)
            fpath = os.path.join(face_dir, stem + ".pt")
            if os.path.exists(fpath):
                self.pairs.append((bpath, fpath))
        if not self.pairs:
            raise RuntimeError("No paired body/face .pt files found.")

    def __len__(self): return len(self.pairs)

    def _normalize(self, x):
        """Standardizes a [3,H,W] tensor."""
        mean = x.mean(dim=(1,2), keepdim=True)
        std  = x.std(dim=(1,2), keepdim=True) + 1e-6
        return (x - mean) / std

    def _maybe_to_img(self, x):
        """Resizes and normalizes the tensor."""
        if x.dim() == 3 and (x.shape[-1] != 224 or x.shape[-2] != 224):
             # Resize if it's not already 224x224
             x = F.interpolate(x.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False).squeeze(0)
    
        # Now, normalize it
        x = self._normalize(x)
        return x

    def __getitem__(self, idx):
        bpath, fpath = self.pairs[idx]
        b = torch.load(bpath, map_location="cpu")
        f = torch.load(fpath, map_location="cpu")
    
        # _maybe_to_img now handles both resizing AND normalization
        body_img = self._maybe_to_img(b["data"].float())
        face_img = self._maybe_to_img(f["data"].float())
    
        label = int(b["label"])
        return {"body": body_img, "face": face_img, "label": label, "name": os.path.basename(bpath)}

# import os, torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset

# def get_subject_id(fname: str):
#     # assumes filenames like: 004_t2_20230124_*.pt
#     return fname.split("_")[0]

# class PoseAsImageDataset(Dataset):
#     def __init__(self, data_dir):
#         self.paths = sorted([
#             os.path.join(data_dir, f)
#             for f in os.listdir(data_dir)
#             if f.endswith(".pt")
#         ])

#         self.subjects = [get_subject_id(os.path.basename(p)) for p in self.paths]

#     def __len__(self):
#         return len(self.paths)

#     def _pose_to_image(self, x):
#         x = x.float()  # [3,T,J]
#         mean = x.mean(dim=(1,2), keepdim=True)
#         std  = x.std(dim=(1,2), keepdim=True) + 1e-6
#         x = (x - mean) / std
#         x = F.interpolate(
#             x.unsqueeze(0), size=(224,224),
#             mode="bilinear", align_corners=False
#         ).squeeze(0)
#         return x

#     def __getitem__(self, idx):
#         item = torch.load(self.paths[idx], map_location="cpu")
#         img = self._pose_to_image(item["data"])
#         return {
#             "image": img,
#             "label": int(item["label"]),
#             "subject": self.subjects[idx],
#             "name": os.path.basename(self.paths[idx]),
#         }

# class FusionPoseDataset(Dataset):
#     def __init__(self, body_dir, face_dir):
#         self.samples = []

#         body_files = [f for f in os.listdir(body_dir) if f.endswith(".pt")]
#         for bf in sorted(body_files):
#             stem = os.path.splitext(bf)[0]
#             fpath = os.path.join(face_dir, stem + ".pt")
#             if os.path.exists(fpath):
#                 self.samples.append((
#                     os.path.join(body_dir, bf),
#                     fpath,
#                     get_subject_id(bf)
#                 ))

#         if not self.samples:
#             raise RuntimeError("No paired body/face files found.")

#     def __len__(self):
#         return len(self.samples)

#     def _normalize(self, x):
#         mean = x.mean(dim=(1,2), keepdim=True)
#         std  = x.std(dim=(1,2), keepdim=True) + 1e-6
#         return (x - mean) / std

#     def _to_img(self, x):
#         if x.shape[-2:] != (224,224):
#             x = F.interpolate(
#                 x.unsqueeze(0), size=(224,224),
#                 mode="bilinear", align_corners=False
#             ).squeeze(0)
#         return self._normalize(x)

#     def __getitem__(self, idx):
#         bpath, fpath, subject = self.samples[idx]
#         b = torch.load(bpath, map_location="cpu")
#         f = torch.load(fpath, map_location="cpu")

#         return {
#             "body": self._to_img(b["data"].float()),
#             "face": self._to_img(f["data"].float()),
#             "label": int(b["label"]),
#             "subject": subject,
#             "name": os.path.basename(bpath),
#         }
