# prepare_face_trends.py

# import os, json, torch, numpy as np
# from glob import glob
# from tqdm import tqdm
# import pandas as pd

# # ---------- Paths ----------
# FACE_ROOT = "/home/janus/iwso-datasets/eiFaceLandmarks"
# OUTPUT_DIR = "./rating_trend/processed_face_trends"
# FRAME_LEN = 300
# LABEL_CSV = "./rating_trend/rating_differences.csv"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ---------- Load trial labels ----------
# df = pd.read_csv(LABEL_CSV)

# df["subject_id"] = df["subject_id"].astype(str).str.zfill(3)

# LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

# # folder_name → list of trials
# trial_map = {}
# for _, r in df.iterrows():
#     trial_map.setdefault(r["folder_name"], []).append(r)

# # ---------- Utils ----------
# def normalize_joints(j):
#     return j - j[0]

# # ---------- Process ----------
# folders = sorted([f for f in os.listdir(FACE_ROOT) if "_t2_" in f])

# for folder in tqdm(folders):
#     if folder not in trial_map:
#         continue

#     sid = folder.split("_")[0]
#     folder_path = os.path.join(FACE_ROOT, folder)

#     json_files = sorted([
#         f for f in glob(os.path.join(folder_path, "*.json"))
#         if "app" in os.path.basename(f).lower()
#     ])

#     if not json_files:
#         continue

#     all_frames = []

#     for jf in json_files:
#         with open(jf, "r") as f:
#             data = json.load(f)

#         for k in sorted(data.keys(), key=lambda x: int(x)):
#             frame = data[k]

#             pts = (
#                 frame.get("face")
#                 or frame.get("face_landmarks")
#                 or frame.get("landmarks")
#             )

#             if not pts:
#                 continue

#             arr = np.zeros((len(pts), 3))
#             try:
#                 for j, p in enumerate(pts):
#                     arr[j] = [p["x"], p["y"], p["z"]]
#             except KeyError:
#                 continue

#             all_frames.append(normalize_joints(arr))

#     if not all_frames:
#         continue

#     arr = np.stack(all_frames)
#     J = arr.shape[1]

#     if arr.shape[0] < FRAME_LEN:
#         pad = np.zeros((FRAME_LEN - arr.shape[0], J, 3))
#         arr = np.concatenate([arr, pad])
#     else:
#         arr = arr[:FRAME_LEN]

#     arr = arr.transpose(2, 0, 1)
#     tensor = torch.tensor(arr, dtype=torch.float32)

#     # ---------- SAVE ONE FILE PER TRIAL ----------
#     for r in trial_map[folder]:
#         trial = r["trial_label"]
#         n_label = LABEL_MAP[r["n_change_type"]]
#         p_label = LABEL_MAP[r["p_change_type"]]

#         out = {
#             "data": tensor,
#             "label_n": n_label,
#             "label_p": p_label,
#             "trial": trial,
#             "subject_id": sid
#         }

#         torch.save(out, f"{OUTPUT_DIR}/{folder}_{trial}.pt")

# print("✅ FACE DONE")


# Generates trial-level face tensors for rating trend classification

import os
import json
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

# ================= CONFIG =================

FACE_ROOT = "/home/janus/iwso-datasets/eiFaceLandmarksNew"
LABEL_CSV = "./rating_trend/rating_differences.csv"
OUTPUT_DIR = "./rating_trend/processed_face_trends"

FRAME_LEN = 300
CHANNELS = 3

LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD LABEL CSV =================

assert os.path.exists(LABEL_CSV), f"CSV not found: {LABEL_CSV}"

df = pd.read_csv(LABEL_CSV)
df["subject_id"] = df["subject_id"].astype(str).str.zfill(3)

# Create mapping:
# (folder_name, trial_label) -> (label_n, label_p)
trial_label_map = {
    (r["folder_name"], r["trial_label"]): (
        LABEL_MAP[r["n_change_type"]],
        LABEL_MAP[r["p_change_type"]],
    )
    for _, r in df.iterrows()
}

print(f"Loaded {len(trial_label_map)} labeled trials.")

# ================= HELPERS =================

def normalize_joints(joints):
    """Normalize face landmarks relative to first landmark."""
    return joints - joints[0]


def extract_face_points(frame: dict):
    """Robustly extract face landmarks from different JSON formats."""
    if "face" in frame:
        return frame["face"]
    if "face_landmarks" in frame:
        return frame["face_landmarks"]
    if "landmarks" in frame:
        return frame["landmarks"]
    return None


# ================= PROCESS FOLDERS =================

folders = sorted([f for f in os.listdir(FACE_ROOT) if "_t2_" in f])

for folder in tqdm(folders):

    folder_path = os.path.join(FACE_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    # Get trials belonging to this folder
    trials_for_folder = [
        (trial, labels)
        for (f_name, trial), labels in trial_label_map.items()
        if f_name == folder
    ]

    if not trials_for_folder:
        continue

    # Get all trial JSON files in folder
    json_files = glob(os.path.join(folder_path, "result_*.json"))

    if not json_files:
        print(f"⚠️ No trial JSON files in {folder}")
        continue

    for trial_label, (label_n, label_p) in trials_for_folder:

        # Match JSON file for this trial
        matching_json = [
            jf for jf in json_files
            if f"result_{trial_label}_" in os.path.basename(jf)
        ]

        if not matching_json:
            print(f"⚠️ No JSON for trial '{trial_label}' in {folder}")
            continue

        jf = matching_json[0]

        # -------- Load JSON --------
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to read {jf}: {e}")
            continue

        all_frames = []

        # Sort frame keys numerically if possible
        keys = sorted(
            data.keys(),
            key=lambda x: int(x) if str(x).isdigit() else x
        )

        for k in keys:
            frame = data[k]
            pts = extract_face_points(frame)

            if not pts:
                continue

            J = len(pts)
            arr = np.zeros((J, CHANNELS))
            valid = True

            for j, p in enumerate(pts):
                try:
                    arr[j] = [p["x"], p["y"], p["z"]]
                except KeyError:
                    valid = False
                    break

            if not valid:
                continue

            arr = normalize_joints(arr)
            all_frames.append(arr)

        if not all_frames:
            print(f"⚠️ No valid frames for {trial_label} in {folder}")
            continue

        # -------- Stack & Pad --------

        seq = np.stack(all_frames, axis=0)  # [T, J, 3]
        T = seq.shape[0]
        J = seq.shape[1]

        if T < FRAME_LEN:
            pad = np.zeros((FRAME_LEN - T, J, CHANNELS))
            seq = np.concatenate([seq, pad], axis=0)
        else:
            seq = seq[:FRAME_LEN]

        seq = seq.transpose(2, 0, 1)  # -> [3, T, J]

        # -------- Save --------

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{folder}_{trial_label}.pt"
        )

        torch.save(
            {
                "data": torch.tensor(seq, dtype=torch.float32),
                "label_n": label_n,
                "label_p": label_p,
                "trial": trial_label,
                "subject_id": folder.split("_")[0]
            },
            out_path
        )

        print(f"✅ Saved {os.path.basename(out_path)}")

print("\n🎉 FACE TREND PREPROCESSING DONE.")
