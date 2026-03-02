# prepare_body_data_symptom_modes.py


# Body pose preprocessing with:
# - symptom mode (retardation / agitation)
# - pose mode (ei / training / coping / all)

import os
import shutil
import pandas as pd
import json
import torch
import numpy as np
from datetime import datetime
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
    help="Pose filtering mode"
)

parser.add_argument(
    "--symptom_mode",
    type=str,
    default="retardation",
    choices=["retardation", "agitation"],
    help="Symptom label set"
)

parser.add_argument(
    "--output_dir",
    type=str,
    default="./symptom_classification/processed_body",
    help="Directory where processed .pt files will be saved"
)

args = parser.parse_args()

POSE_MODE = args.pose_mode.lower()
SYMPTOM_MODE = args.symptom_mode.lower()
OUTPUT_DIR = args.output_dir


# ==================================================
# CONFIG
# ==================================================
LABEL_CSV_MAP = {
    "retardation": "symptom_classification/labels/retardation_labels.csv",
    "agitation": "symptom_classification/labels/agitation_labels.csv",
}

if SYMPTOM_MODE not in LABEL_CSV_MAP:
    raise ValueError(f"Unknown SYMPTOM_MODE: {SYMPTOM_MODE}")

CONFIG = {
    "json_root_dir": "/home/janus/iwso-datasets/t2-3d-body-poses",
    "raw_data_dir": "/home/vault/empkins/tpD/D02/RCT/raw_data",
    "label_csv": LABEL_CSV_MAP[SYMPTOM_MODE],
    "num_joints": 11,
    "channels": 3,
    "frame_len": 300,
}


# ==================================================
# UTILS
# ==================================================
def fresh_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def normalize_joints(j):
    return j - j[0]


# ==================================================
# INIT OUTPUT DIRECTORY
# ==================================================
fresh_dir(OUTPUT_DIR)


# ==================================================
# LOAD LABELS
# ==================================================
labels_df = pd.read_csv(CONFIG["label_csv"], dtype=str)

labels_df["subject_id"] = labels_df["subject_id"].str.zfill(4)
labels_df["label"] = labels_df["label"].str.lower().str.strip()

LABEL_MAP = {"healthy": 0, "depressed": 1}
labels_df["label_num"] = labels_df["label"].map(LABEL_MAP)

labels_dict = dict(zip(labels_df["subject_id"], labels_df["label_num"]))

print("✅ Labels loaded:")
print(labels_df["label"].value_counts())
print("--------------------------------------------------")


# ==================================================
# PROCESS ONE PARTICIPANT
# ==================================================
def process_participant(folder_name):

    subject_id = folder_name.split("_")[0]

    if subject_id not in labels_dict:
        print(f"⚠️ Skipping {folder_name} (no label)")
        return False

    poses_json = os.path.join(CONFIG["json_root_dir"], folder_name, "poses.json")
    app_dir = os.path.join(CONFIG["raw_data_dir"], subject_id)

    if not os.path.exists(poses_json):
        print(f"⚠️ Skipping {folder_name} (no poses.json)")
        return False

    if not os.path.exists(app_dir):
        print(f"⚠️ Skipping {folder_name} (no raw data dir)")
        return False

    # ---------------- APP CSVs ----------------
    app_files = [
        f for f in os.listdir(app_dir)
        if f.lower().endswith(".csv") and "app" in f.lower()
    ]

    if not app_files:
        print(f"⚠️ Skipping {folder_name} (no app CSV)")
        return False

    dfs = []
    for f in app_files:
        df = pd.read_csv(os.path.join(app_dir, f))
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        dfs.append(df)

    app_df = (
        pd.concat(dfs, ignore_index=True)
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # ---------------- POSE MODE FILTER ----------------
    if POSE_MODE == "ei":
        app_df = app_df[app_df["label"].str.match(r"ei_\d+", na=False)]

    elif POSE_MODE == "training":
        app_df = app_df[app_df["label"].str.contains("training", case=False, na=False)]

    elif POSE_MODE == "coping":
        app_df = app_df[app_df["label"].str.contains("coping", case=False, na=False)]

    app_df = app_df.reset_index(drop=True)

    if len(app_df) < 2:
        print(f"⚠️ Skipping {folder_name} (not enough app segments)")
        return False

    # ---------------- KINECT TIMESTAMPS ----------------
    kinect_files = [
        f for f in os.listdir(app_dir)
        if "kinect" in f.lower() and f.endswith(".txt")
    ]

    if not kinect_files:
        print(f"⚠️ Skipping {folder_name} (no kinect file)")
        return False

    with open(os.path.join(app_dir, kinect_files[0])) as f:
        frame_times = [
            datetime.strptime(l.strip(), "%Y-%m-%d %H:%M:%S.%f")
            for l in f if "Start time" not in l
        ]

    with open(poses_json) as f:
        frames = json.load(f).get("frames", [])

    # ---------------- SELECT FRAMES ----------------
    selected = []

    for i in range(len(app_df) - 1):
        start = app_df.loc[i, "timestamp"]
        end = app_df.loc[i + 1, "timestamp"]

        try:
            s = next(j for j, t in enumerate(frame_times) if t >= start)
            e = next(j for j, t in reversed(list(enumerate(frame_times))) if t <= end)
            selected.extend(frames[s:e + 1])
        except StopIteration:
            continue

    if not selected:
        print(f"⚠️ Skipping {folder_name} (no frames selected)")
        return False

    # ---------------- BUILD SEQUENCE ----------------
    seq = []

    for frame in selected:
        coords = np.zeros((CONFIG["num_joints"], CONFIG["channels"]))

        for j in frame.get("poses", []):
            jid = j.get("joint")
            if jid is not None and jid < CONFIG["num_joints"]:
                coords[jid] = [j["x_3d"], j["y_3d"], j["z_3d"]]

        seq.append(normalize_joints(coords))

    seq = np.stack(seq)

    if seq.shape[0] < CONFIG["frame_len"]:
        pad = np.zeros(
            (CONFIG["frame_len"] - seq.shape[0],
             CONFIG["num_joints"],
             CONFIG["channels"])
        )
        seq = np.concatenate([seq, pad])
    else:
        seq = seq[:CONFIG["frame_len"]]

    tensor = torch.tensor(seq.transpose(2, 0, 1), dtype=torch.float32)

    torch.save(
        {"data": tensor, "label": labels_dict[subject_id]},
        os.path.join(OUTPUT_DIR, f"{folder_name}.pt")
    )

    print(f"✅ Saved {folder_name}")
    return True


# ==================================================
# RUN DATASET
# ==================================================
processed_count = 0
skipped_count = 0

folders = sorted(os.listdir(CONFIG["json_root_dir"]))

for folder in tqdm(folders, desc="Processing BODY"):

    folder_path = os.path.join(CONFIG["json_root_dir"], folder)

    if not os.path.isdir(folder_path):
        continue

    success = process_participant(folder)

    if success:
        processed_count += 1
    else:
        skipped_count += 1


print("\n==============================")
print(f"✅ Processed: {processed_count}")
print(f"⚠️ Skipped:   {skipped_count}")
print("🎉 BODY preprocessing DONE.")
print("==============================")
