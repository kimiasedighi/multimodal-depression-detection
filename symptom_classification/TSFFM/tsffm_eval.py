# tsffm_eval.py

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.append("symptom_classification/Res_Res")

from tsffm_model import TSFFM
from resres_datasets import FusionPoseDataset


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        b = batch["body"].to(device)
        f = batch["face"].to(device)

        out = model(b, f)  # inference mode
        logits = out["logits_final"]

        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        labels.extend(batch["label"].tolist())

    acc = (np.array(preds) == np.array(labels)).mean() * 100
    return acc, preds, labels


def make_test_loader(args):
    ds = FusionPoseDataset(args.body_dir, args.face_dir)

    idx = list(range(len(ds)))
    y_all = [ds[i]["label"] for i in idx]

    # SAME split logic as training (only need test split for eval)
    _, te = train_test_split(
        idx,
        test_size=0.2,
        stratify=y_all,
        random_state=42
    )

    test_loader = DataLoader(
        Subset(ds, te),
        batch_size=args.batch_size,
        shuffle=False
    )

    return test_loader


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader = make_test_loader(args)

    model = TSFFM(num_classes=2, pretrained=args.pretrained).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    test_acc, preds, labels = evaluate(model, test_loader, device)

    print(f"\n🎯 TEST Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser("TSFFM Evaluation (binary)")

    ap.add_argument("--body_dir", default="symptom_classification/processed_body")
    ap.add_argument("--face_dir", default="symptom_classification/processed_face")

    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--pretrained", action="store_true")

    ap.add_argument("--model_path", required=True)

    args = ap.parse_args()
    main(args)
