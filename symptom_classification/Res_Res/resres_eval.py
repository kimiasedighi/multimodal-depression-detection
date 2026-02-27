# resres_eval.py - Evaluate Res-Res model on test set and print classification report + confusion matrix

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from resres_model import ResRes18DecisionFusion
from resres_datasets import PoseAsImageDataset, FusionPoseDataset

NUM_CLASSES = 2


def make_test_loader(args):
    if args.input_type == "F+B":
        ds = FusionPoseDataset(args.body_dir, args.face_dir)
    elif args.input_type == "F":
        ds = PoseAsImageDataset(args.face_dir)
    else:
        ds = PoseAsImageDataset(args.body_dir)

    idx = list(range(len(ds)))
    y_all = [ds[i]["label"] for i in idx]

    # SAME split logic as training
    _, test_idx = train_test_split(
        idx,
        test_size=0.2,
        random_state=42,
        stratify=y_all
    )

    test_loader = DataLoader(
        Subset(ds, test_idx),
        batch_size=args.batch_size,
        shuffle=False
    )

    return test_loader


@torch.no_grad()
def evaluate(model, loader, device, input_type):
    model.eval()
    preds, labels = [], []

    for batch in loader:
        y = batch["label"]

        if input_type == "B":
            out = model(body_img=batch["image"].to(device))
            logits = out["logits_body"]

        elif input_type == "F":
            out = model(face_img=batch["image"].to(device))
            logits = out["logits_face"]

        else:  # F+B
            out = model(
                body_img=batch["body"].to(device),
                face_img=batch["face"].to(device),
            )
            logits = out["logits_fused"]

        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        labels.extend(y.tolist())

    acc = (np.array(preds) == np.array(labels)).mean() * 100
    return acc, preds, labels


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_loader = make_test_loader(args)

    model = ResRes18DecisionFusion(
        num_classes=NUM_CLASSES,
        input_type=args.input_type,
        fusion=args.fusion,
        pretrained=args.pretrained,
        branch_loss_weight=args.branch_loss_w,
        fusion_loss_weight=args.fusion_loss_w,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))

    test_acc, preds, labels = evaluate(model, test_loader, device, args.input_type)

    print(f"\n🎯 TEST Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Evaluate Res-Res model")

    ap.add_argument("--input_type", type=str, default="B", choices=["B", "F", "F+B"])
    ap.add_argument("--body_dir", type=str, default="./symptom_classification/processed_body")
    ap.add_argument("--face_dir", type=str, default="./symptom_classification/processed_face")

    ap.add_argument("--batch_size", type=int, default=16)

    ap.add_argument("--fusion", type=str, default="avg_logit", choices=["avg_logit", "avg_prob"])
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--branch_loss_w", type=float, default=1.0)
    ap.add_argument("--fusion_loss_w", type=float, default=1.0)

    ap.add_argument("--model_path", type=str, required=True)

    args = ap.parse_args()
    main(args)
