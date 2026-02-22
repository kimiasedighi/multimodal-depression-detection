# msn_eval.py

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from msn_dataset import PoseDataset      # your dataset.py
from msn_body import MSNBody         # your msn_body.py

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for data, labels in loader:
        data  = data.to(device)          # [B, C, T, J]
        labels = labels.to(device)
        outputs = model(data)            # [B, num_classes]
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean() * 100.0
    return acc, all_preds.tolist(), all_labels.tolist()

def main():
    ap = argparse.ArgumentParser("Evaluate saved MSN model on a folder of .pt tensors")
    ap.add_argument("--data_dir", type=str, default="./processed_data",
                    help="Folder with .pt files (each has {'data':[C,T,J], 'label':int})")
    ap.add_argument("--model_path", type=str, default="MSN/best_msn_model.pth",
                    help="Path to saved MSN weights")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--class_names", type=str, default="Healthy,Depressed",
                    help="Comma-separated names for classes (for pretty report)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    ds = PoseDataset(args.data_dir)
    if len(ds) == 0:
        raise RuntimeError(f"No .pt files found in {args.data_dir}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Model
    model = MSNBody(in_channels=3, num_joints=11, num_classes=args.num_classes).to(device)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Eval
    acc, preds, labels = evaluate(model, loader, device)
    print(f"Accuracy: {acc:.2f}%")

    try:
        names = [s.strip() for s in args.class_names.split(",")] if args.class_names else None
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=names, digits=4))
    except Exception:
        print("\nClassification Report:")
        print(classification_report(labels, preds, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    main()
