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
import argparse


# ==================================================
# ARGUMENTS
# ==================================================
parser = argparse.ArgumentParser()

parser.add_argument(
    "--pose_mode",
    type=str,
    default="all",
    choices=["all", "ei", "training", "coping"],
    help="Pose selection mode"
)

parser.add_argument(
    "--symptom_mode",
    type=str,
    default="retardation",
    choices=["retardation", "agitation"],
    help="Select symptom label set"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./symptom_classification/processed_face",
    help="Directory where processed .pt files will be saved"
)

args = parser.parse_args()

POSE_MODE = args.pose_mode.lower()
SYMPTOM_MODE = args.symptom_mode.lower()
OUTPUT_DIR = args.output_dir

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# CONFIG
# ==================================================
FACE_ROOT = "/home/janus/iwso-datasets/eiFaceLandmarksNew"
FRAME_LEN = 300
CHANNELS = 3

LABEL_CSV_MAP = {
    "retardation": "symptom_classification/labels/retardation_labels.csv",
    "agitation": "symptom_classification/labels/agitation_labels.csv",
}

if SYMPTOM_MODE not in LABEL_CSV_MAP:
    raise ValueError(f"Unknown SYMPTOM_MODE: {SYMPTOM_MODE}")

LABEL_CSV = LABEL_CSV_MAP[SYMPTOM_MODE]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# LOAD LABELS
# ==================================================
labels_df = pd.read_csv(LABEL_CSV, dtype=str)

labels_df["subject_id"] = labels_df["subject_id"].str.zfill(4)
labels_df["label"] = labels_df["label"].str.lower().str.strip()

LABEL_MAP = {"healthy": 0, "depressed": 1}
labels_df["label_num"] = labels_df["label"].map(LABEL_MAP)

labels_dict = dict(zip(labels_df["subject_id"], labels_df["label_num"]))

print("✅ Labels loaded:")
print(labels_df["label"].value_counts())


# ==================================================
# HELPERS
# ==================================================
def normalize_joints(joints):
    """Normalize landmarks relative to first landmark"""
    return joints - joints[0]


def extract_face_points(frame: dict):
    if "face" in frame:
        return frame["face"]
    if "face_landmarks" in frame:
        return frame["face_landmarks"]
    if "landmarks" in frame:
        return frame["landmarks"]
    return None


def file_matches_mode(fname: str) -> bool:
    fname = fname.lower()

    if POSE_MODE == "ei":
        return "app" in fname and "facial_landmarks" in fname

    if POSE_MODE == "training":
        return "result_training" in fname

    if POSE_MODE == "coping":
        return "result_coping" in fname

    # all
    return (
        ("app" in fname and "facial_landmarks" in fname)
        or "result_training" in fname
        or "result_coping" in fname
    )


# ==================================================
# PROCESS DATASET
# ==================================================
folders = sorted([f for f in os.listdir(FACE_ROOT) if "_t2_" in f])

for folder in tqdm(folders):

    sid = folder.split("_")[0].zfill(4)

    if sid not in labels_dict:
        print(f"Skipping {folder} (no label)")
        continue

    folder_path = os.path.join(FACE_ROOT, folder)

    json_files = [
        f for f in glob(os.path.join(folder_path, "*.json"))
        if file_matches_mode(os.path.basename(f))
    ]

    if not json_files:
        print(f"⚠️ No matching JSONs in {folder} for mode={POSE_MODE}")
        continue

    all_frames = []

    for jf in sorted(json_files):
        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to read {jf}: {e}")
            continue

        for k in sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else x):
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

    if len(all_frames) == 0:
        print(f"⚠️ No valid face frames in {folder}")
        continue

    # ================= STACK + PAD =================
    T = len(all_frames)
    J = all_frames[0].shape[0]

    seq = np.stack(all_frames, axis=0)  # [T, J, 3]

    if T < FRAME_LEN:
        pad = np.zeros((FRAME_LEN - T, J, CHANNELS))
        seq = np.concatenate([seq, pad], axis=0)
    else:
        seq = seq[:FRAME_LEN]

    seq = seq.transpose(2, 0, 1)  # [3, T, J]

    out_path = os.path.join(OUTPUT_DIR, f"{folder}.pt")

    torch.save(
        {
            "data": torch.tensor(seq, dtype=torch.float32),
            "label": labels_dict[sid]
        },
        out_path
    )

    print(f"✅ Saved {folder}")

print("🎉 FACE preprocessing DONE.")
