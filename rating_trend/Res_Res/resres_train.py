# resres_train.py 

import os
import argparse
import torch
import numpy as np
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from resres_model import ResRes18DecisionFusion
from resres_datasets import PoseAsImageDataset, FusionPoseDataset

NUM_CLASSES = 3

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def seed_all(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def compute_class_weights(ds, indices, label_key, num_classes=NUM_CLASSES):
    labels = [ds[i][label_key] for i in indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32), counts


# --------------------------------------------------
# Dataset selector
# --------------------------------------------------

def make_dataset(args):
    if args.input_type == "F+B":
        return FusionPoseDataset(args.body_dir, args.face_dir)
    elif args.input_type == "F":
        return PoseAsImageDataset(args.face_dir, key_name="face")
    elif args.input_type == "B":
        return PoseAsImageDataset(args.body_dir, key_name="body")
    else:
        raise ValueError("Invalid input_type")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

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

        elif input_type == "F+B":
            out = model(body_img=b, face_img=f)
            logits = out["logits_fused"]

        else:
            raise ValueError("Invalid input_type")

        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        labels.extend(batch[label_key].cpu().tolist())

    acc = (np.array(preds) == np.array(labels)).mean() * 100.0
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, f1, preds, labels


# --------------------------------------------------
# Main training
# --------------------------------------------------

def main(args):
    seed_all(args.seed)

    # Force uppercase to avoid logic bugs
    args.input_type = args.input_type.upper()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print("Input type:", args.input_type)

    ds = make_dataset(args)
    tr_idx, va_idx, te_idx = subject_split(ds, seed=args.seed)

    train_loader = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(ds, va_idx), batch_size=args.batch_size)
    test_loader  = DataLoader(Subset(ds, te_idx), batch_size=args.batch_size)

    # class weights
    w, counts = compute_class_weights(ds, tr_idx, args.label, NUM_CLASSES)
    w = w.to(device)
    print("Train class counts:", counts.tolist())
    print("Class weights:", w.detach().cpu().tolist())

    model = ResRes18DecisionFusion(
        num_classes=NUM_CLASSES,
        input_type=args.input_type,
        fusion=args.fusion,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):

        model.train()
        running_loss = 0.0

        for batch in train_loader:

            b = batch.get("body")
            f = batch.get("face")

            if b is not None:
                b = b.to(device)
            if f is not None:
                f = f.to(device)

            y = batch[args.label].to(device)

            optimizer.zero_grad()

            if args.input_type == "B":
                out = model(body_img=b)
                logits = out["logits_body"]

            elif args.input_type == "F":
                out = model(face_img=f)
                logits = out["logits_face"]

            elif args.input_type == "F+B":
                out = model(body_img=b, face_img=f)
                logits = out["logits_fused"]

            else:
                raise ValueError("Invalid input_type")

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc, val_f1, _, _ = evaluate(model, val_loader, device, args.input_type, args.label)

        print(f"Epoch {epoch}/{args.epochs}  "
              f"Loss {running_loss/len(train_loader):.4f}  "
              f"ValAcc {val_acc:.2f}%  ValF1 {val_f1:.3f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print("✅ Saved best model")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("⛔ Early stopping triggered")
            break

    # ------------------ Final Test ------------------

    model.load_state_dict(torch.load(args.save_path, map_location=device))
    test_acc, test_f1, preds, labels = evaluate(model, test_loader, device, args.input_type, args.label)

    print("\n🎯 FINAL TEST")
    print(f"Accuracy: {test_acc:.2f}%   MacroF1: {test_f1:.3f}")

    names = ["negative", "neutral", "positive"]
    print(classification_report(labels, preds, target_names=names, digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


# --------------------------------------------------
# Argparse
# --------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Improved ResRes Training")

    ap.add_argument("--input_type", default="B", choices=["B", "F", "F+B"])
    ap.add_argument("--body_dir", default="./rating_trend/processed_body_trends")
    ap.add_argument("--face_dir", default="./rating_trend/processed_face_trends")
    ap.add_argument("--label", default="label_n", choices=["label_n", "label_p"])

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--fusion", default="avg_logit", choices=["avg_logit", "avg_prob"])
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--save_path", default="./rating_trend/Res_Res/best_resres18.pth")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    main(args)
