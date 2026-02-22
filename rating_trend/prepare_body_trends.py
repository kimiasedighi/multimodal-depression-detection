# prepare_body_trends.py

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# ---------- CONFIG ----------
JSON_ROOT = "/home/janus/iwso-datasets/t2-3d-body-poses"
RAW_DIR = "/home/vault/empkins/tpD/D02/RCT/raw_data"
OUTPUT_DIR = "./rating_trend/processed_body_trends"

LABEL_CSV = "/home/hpc/iwso/iwso193h/rating_trend/rating_differences.csv"

FRAME_LEN = 300
NUM_JOINTS = 11
CHANNELS = 3

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

# ---------- SAFETY CHECKS ----------
assert os.path.exists(LABEL_CSV), f"CSV not found: {LABEL_CSV}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- LOAD LABELS ----------
df = pd.read_csv(LABEL_CSV)
df["subject_id"] = df["subject_id"].astype(str).str.zfill(3)

trial_label_map = {
    (r["folder_name"], r["trial_label"]): (
        LABEL_MAP[r["n_change_type"]],
        LABEL_MAP[r["p_change_type"]],
    )
    for _, r in df.iterrows()
}

def normalize_joints(j):
    return j - j[0]

# ---------- PROCESS ----------
for folder in sorted(os.listdir(JSON_ROOT)):
    folder_path = os.path.join(JSON_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    subject_id = folder.split("_")[0]
    app_dir = os.path.join(RAW_DIR, subject_id)

    poses_json = os.path.join(folder_path, "poses.json")
    if not os.path.exists(poses_json):
        continue

    if not os.path.isdir(app_dir):
        continue

    # ---- APP CSVs ----
    app_files = [
        f for f in os.listdir(app_dir)
        if f.lower().endswith(".csv") and "app" in f.lower()
    ]

    if not app_files:
        continue

    df_app = pd.concat(
        [pd.read_csv(os.path.join(app_dir, f)) for f in app_files],
        ignore_index=True
    )

    df_app["label"] = df_app["label"].astype(str).str.strip('"')
    df_app["timestamp"] = pd.to_datetime(df_app["timestamp"], errors="coerce")
    df_app = df_app.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # ---- KINECT TIMESTAMPS (SAFE) ----
    kinect_files = [
        f for f in os.listdir(app_dir)
        if f.lower().endswith(".txt") and "kinect" in f.lower()
    ]

    if not kinect_files:
        print(f"⚠️ No Kinect timestamp file for {folder}, skipping.")
        continue

    with open(os.path.join(app_dir, kinect_files[0]), "r") as f:
        frame_times = [
            datetime.strptime(l.strip(), "%Y-%m-%d %H:%M:%S.%f")
            for l in f if l.strip() and "Start time" not in l
        ]

    with open(poses_json, "r") as f:
        frames = json.load(f).get("frames", [])

    # ---- PROCESS EACH TRIAL ----
    for i, row in df_app.iterrows():
        trial = row["label"]
        key = (folder, trial)

        if key not in trial_label_map:
            continue

        if i + 1 >= len(df_app):
            continue

        start, end = row["timestamp"], df_app.iloc[i + 1]["timestamp"]

        try:
            s = next(i for i, t in enumerate(frame_times) if t >= start)
            e = next(i for i, t in reversed(list(enumerate(frame_times))) if t <= end)
        except StopIteration:
            continue

        seq = []
        for fr in frames[s:e + 1]:
            coords = np.zeros((NUM_JOINTS, CHANNELS))
            for j in fr.get("poses", []):
                jid = j.get("joint")
                if jid is not None and jid < NUM_JOINTS:
                    coords[jid] = [j["x_3d"], j["y_3d"], j["z_3d"]]
            seq.append(normalize_joints(coords))

        if not seq:
            continue

        seq = np.stack(seq)
        if seq.shape[0] < FRAME_LEN:
            pad = np.zeros((FRAME_LEN - seq.shape[0], NUM_JOINTS, CHANNELS))
            seq = np.concatenate([seq, pad])
        else:
            seq = seq[:FRAME_LEN]

        tensor = torch.tensor(seq.transpose(2, 0, 1), dtype=torch.float32)
        label_n, label_p = trial_label_map[key]

        out_path = os.path.join(OUTPUT_DIR, f"{folder}_{trial}.pt")
        torch.save(
            {
                "data": tensor,
                "label_n": label_n,
                "label_p": label_p,
                "trial": trial,
                "subject_id": subject_id
            },
            out_path
        )

        print(f"✅ Saved {out_path}")

print("🎉 BODY PREPROCESSING DONE")
