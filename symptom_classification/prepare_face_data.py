# prepare_face_data.py for depressed and not depressed

import os
import json
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pose_mode",
    type=str,
    default="all",
    choices=["all", "ei", "training", "coping"],
    help="Select pose mode"
)

parser.add_argument(
    "--pattern",
    type=str,
    default="CRADK|ADK",
    help="Regex pattern for group filtering (e.g. CR|ADK|SHAM)"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./symptom_classification/processed_face",
    help="Directory where processed .pt files will be saved"
)

args = parser.parse_args()

# ================= CONFIG =================
FACE_ROOT = "/home/janus/iwso-datasets/eiFaceLandmarksNew"
OUTPUT_DIR = args.output_dir
LABEL_FILE = "./symptom_classification/labels/20250110_Participant_list.xlsx"

FRAME_LEN = 300
CHANNELS = 3

# POSE_MODE = os.environ.get("POSE_MODE", "all").lower()
POSE_MODE = args.pose_mode.lower()

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LABEL LOADING (same logic as body) =================
sheet = pd.read_excel(
    LABEL_FILE,
    sheet_name="Sheet3",
    header=2,
    engine="openpyxl"
)

sheet.columns = (
    sheet.columns.astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# pattern = "CRADK|ADK"
pattern = args.pattern


def normalize_id(x):
    x = str(x).split(".")[0].strip()
    if x.isdigit():
        return x.zfill(3)
    return x

dep_ids = sheet.loc[
    sheet["Bedingung"].astype(str).str.contains(pattern, case=False, na=False),
    "ID"
].dropna().apply(normalize_id)

healthy_ids = sheet.loc[
    sheet["Bedingung.1"].astype(str).str.contains(pattern, case=False, na=False),
    "ID.1"
].dropna().apply(normalize_id)

labels = {
    **{sid: 1 for sid in dep_ids},
    **{sid: 0 for sid in healthy_ids}
}
labels.pop("ID", None)

# ================= HELPERS =================
def normalize_joints(joints):
    """Normalize relative to first landmark"""
    return joints - joints[0]

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

def extract_face_points(frame: dict):
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
    sid = folder.split("_")[0]

    if sid not in labels:
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
        print(f"Reading {os.path.basename(jf)}")

        try:
            with open(jf, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to read {jf}: {e}")
            continue

        # keys are usually frame indices as strings
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
            "label": int(labels[sid])
        },
        out_path
    )

    print(f"✅ Saved {folder}")

print("🎉 FACE PREPROCESSING DONE.")
