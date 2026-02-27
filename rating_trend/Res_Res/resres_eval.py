# resres_eval.py

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from resres_model import ResRes18DecisionFusion
from resres_datasets import PoseAsImageDataset, FusionPoseDataset

NUM_CLASSES = 3


# ---------- SUBJECT SPLIT (STATIC - SAME AS TRAIN) ----------
def subject_split(ds, test_ratio=0.2, val_ratio=0.1, seed=42):
    subjects = defaultdict(list)
    for i in range(len(ds)):
        subjects[ds[i]["subject_id"]].append(i)

    rng = np.random.RandomState(seed)
    sids = list(subjects.keys())
    rng.shuffle(sids)

    n_test = int(len(sids) * test_ratio)
    n_val  = int((len(sids) - n_test) * val_ratio)

    test_sids = sids[:n_test]
    val_sids  = sids[n_test:n_test + n_val]
    train_sids = sids[n_test + n_val:]

    def collect(which):
        return [i for sid in which for i in subjects[sid]]

    return collect(train_sids), collect(val_sids), collect(test_sids)


# ---------- DATASET SELECTOR (SAME LOGIC AS TRAIN) ----------
def make_dataset(args):
    if args.input_type == "F+B":
        return FusionPoseDataset(args.body_dir, args.face_dir)
    elif args.input_type == "F":
        return PoseAsImageDataset(args.face_dir, key_name="face")
    elif args.input_type == "B":
        return PoseAsImageDataset(args.body_dir, key_name="body")
    else:
        raise ValueError("Invalid input_type")


# ---------- EVALUATION ----------
@torch.no_grad()
def evaluate(model, loader, device, input_type, label_key):
    model.eval()
    preds, labels = [], []

    for batch in loader:
        b = batch.get("body")
        f = batch.get("face")

        if b is not None:
            b = b.to(device)
        if f is not None:
            f = f.to(device)

        if input_type == "B":
            out = model(body_img=b)
            logits = out["logits_body"]

        elif input_type == "F":
            out = model(face_img=f)
            logits = out["logits_face"]

        else:  # F+B
            out = model(body_img=b, face_img=f)
            logits = out["logits_fused"]

        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        labels.extend(batch[label_key].cpu().tolist())

    acc = (np.array(preds) == np.array(labels)).mean() * 100.0
    f1  = f1_score(labels, preds, average="macro", zero_division=0)

    return acc, f1, preds, labels


# ---------- MAIN ----------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Input type:", args.input_type)

    ds = make_dataset(args)

    # STATIC split — identical to training
    _, _, te_idx = subject_split(ds, seed=args.seed)

    test_loader = DataLoader(
        Subset(ds, te_idx),
        batch_size=args.batch_size,
        shuffle=False
    )

    model = ResRes18DecisionFusion(
        num_classes=NUM_CLASSES,
        input_type=args.input_type,
        fusion=args.fusion,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    test_acc, test_f1, preds, labels = evaluate(
        model,
        test_loader,
        device,
        args.input_type,
        args.label
    )

    print("\n🎯 FINAL TEST")
    print(f"Accuracy: {test_acc:.2f}%   MacroF1: {test_f1:.3f}")

    names = ["negative", "neutral", "positive"]

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=names, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Evaluate ResRes Trend Model (STATIC SPLIT)")

    ap.add_argument("--input_type", default="B", choices=["B", "F", "F+B"])
    ap.add_argument("--body_dir", default="./rating_trend/processed_body_trends")
    ap.add_argument("--face_dir", default="./rating_trend/processed_face_trends")
    ap.add_argument("--label", default="label_n", choices=["label_n", "label_p"])

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fusion", default="avg_logit", choices=["avg_logit", "avg_prob"])
    ap.add_argument("--model_path", required=True)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    main(args)
