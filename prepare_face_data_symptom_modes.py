# prepare_face_data_symptom_modes.py

# Face preprocessing with:
# - symptom mode (retardation / agitation)
# - pose mode (ei / training / coping / all)

import os
import json
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# --------------------------------------------------
# MODES
# --------------------------------------------------
POSE_MODE = os.environ.get("POSE_MODE", "all").lower()
SYMPTOM_MODE = os.environ.get("SYMPTOM_MODE", "retardation").lower()

LABEL_CSV_MAP = {
    "retardation": "labels/retardation_labels.csv",
    "agitation": "labels/agitation_labels.csv",
}

if SYMPTOM_MODE not in LABEL_CSV_MAP:
    raise ValueError(f"Unknown SYMPTOM_MODE: {SYMPTOM_MODE}")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
FACE_ROOT = "/home/janus/iwso-datasets/eiFaceLandmarks"
LABEL_CSV = LABEL_CSV_MAP[SYMPTOM_MODE]
OUTPUT_DIR = "./processed_face"
FRAME_LEN = 300
CHANNELS = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD LABELS
# --------------------------------------------------
labels_df = pd.read_csv(LABEL_CSV, dtype=str)
labels_df["subject_id"] = labels_df["subject_id"].str.zfill(4)
labels_df["label"] = labels_df["label"].str.lower().str.strip()

LABEL_MAP = {"healthy": 0, "depressed": 1}
labels_df["label_num"] = labels_df["label"].map(LABEL_MAP)

labels_dict = dict(zip(labels_df["subject_id"], labels_df["label_num"]))

print("✅ Labels loaded:")
print(labels_df["label"].value_counts())

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def normalize_joints(j):
    return j - j[0]

def extract_pts(frame):
    for k in ["face", "face_landmarks", "landmarks"]:
        if k in frame:
            return frame[k]
    return None

def file_matches_mode(fname):
    f = fname.lower()
    if POSE_MODE == "ei":
        return "app" in f
    if POSE_MODE == "training":
        return "result_training" in f
    if POSE_MODE == "coping":
        return "result_coping" in f
    return ("app" in f or "result_training" in f or "result_coping" in f)

# --------------------------------------------------
# PROCESS
# --------------------------------------------------
folders = sorted([f for f in os.listdir(FACE_ROOT) if "_t2_" in f])

for folder in tqdm(folders):
    sid = folder.split("_")[0].zfill(4)
    if sid not in labels_dict:
        continue

    json_files = [
        f for f in glob(os.path.join(FACE_ROOT, folder, "*.json"))
        if file_matches_mode(os.path.basename(f))
    ]

    if not json_files:
        continue

    frames_all = []

    for jf in sorted(json_files):
        with open(jf) as f:
            data = json.load(f)

        for k in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
            pts = extract_pts(data[k])
            if not pts:
                continue

            arr = np.array([[p["x"], p["y"], p["z"]] for p in pts])
            frames_all.append(normalize_joints(arr))

    if not frames_all:
        continue

    seq = np.stack(frames_all)

    if seq.shape[0] < FRAME_LEN:
        pad = np.zeros((FRAME_LEN - seq.shape[0], seq.shape[1], CHANNELS))
        seq = np.concatenate([seq, pad])
    else:
        seq = seq[:FRAME_LEN]

    seq = seq.transpose(2, 0, 1)

    torch.save(
        {"data": torch.tensor(seq, dtype=torch.float32),
         "label": labels_dict[sid]},
        os.path.join(OUTPUT_DIR, f"{folder}.pt")
    )

print("🎉 FACE preprocessing DONE.")
