# msn_train.py

import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from msn_dataset import PoseDataset
from msn_body import MSNBody


# ---------- SUBJECT SPLIT ----------
def subject_split(dataset, test_ratio=0.2, val_ratio=0.1, seed=42):
    subjects = defaultdict(list)
    for i in range(len(dataset)):
        _, _, _, sid = dataset[i]
        subjects[sid].append(i)

    rng = np.random.RandomState(seed)
    subject_ids = list(subjects.keys())
    rng.shuffle(subject_ids)

    n_test = int(len(subject_ids) * test_ratio)
    n_val = int((len(subject_ids) - n_test) * val_ratio)

    test_sids = subject_ids[:n_test]
    val_sids = subject_ids[n_test:n_test + n_val]
    train_sids = subject_ids[n_test + n_val:]

    def collect(sids):
        return [i for sid in sids for i in subjects[sid]]

    return collect(train_sids), collect(val_sids), collect(test_sids)


# ---------- CLASS WEIGHTS ----------
def compute_class_weights(labels, num_classes=3):
    counts = np.bincount(labels, minlength=num_classes)
    weights = counts.sum() / (num_classes * counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)


# ---------- TRAIN ----------
def train_epoch(model, loader, crit_n, crit_p, opt, device):
    model.train()
    total = 0

    for x, y_n, y_p, _ in loader:
        x, y_n, y_p = x.to(device), y_n.to(device), y_p.to(device)

        opt.zero_grad()
        out_n, out_p = model(x)

        loss = crit_n(out_n, y_n) + crit_p(out_p, y_p)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        total += loss.item()

    return total / len(loader)


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    pn, pp, yn, yp = [], [], [], []

    for x, y_n, y_p, _ in loader:
        x = x.to(device)
        out_n, out_p = model(x)

        pn.extend(out_n.argmax(1).cpu())
        pp.extend(out_p.argmax(1).cpu())
        yn.extend(y_n)
        yp.extend(y_p)

    f1_n = f1_score(yn, pn, average="macro", zero_division=0)
    f1_p = f1_score(yp, pp, average="macro", zero_division=0)
    return f1_n, f1_p, pn, pp, yn, yp


# ---------- MAIN ----------
def main(args):

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU")

    dataset = PoseDataset(args.data_dir)
    tr_idx, va_idx, te_idx = subject_split(dataset)

    train_loader = DataLoader(Subset(dataset, tr_idx), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, va_idx), batch_size=args.batch_size)
    test_loader  = DataLoader(Subset(dataset, te_idx), batch_size=args.batch_size)

    # class weights
    labels_n = [dataset[i][1] for i in tr_idx]
    labels_p = [dataset[i][2] for i in tr_idx]

    w_n = compute_class_weights(labels_n).to(device)
    w_p = compute_class_weights(labels_p).to(device)

    model = MSNBody().to(device)

    crit_n = nn.CrossEntropyLoss(weight=w_n, label_smoothing=0.05)
    crit_p = nn.CrossEntropyLoss(weight=w_p, label_smoothing=0.05)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = 0.0

    for ep in range(args.epochs):
        loss = train_epoch(model, train_loader, crit_n, crit_p, opt, device)
        f1n, f1p, *_ = eval_model(model, val_loader, device)

        score = (f1n + f1p) / 2

        print(f"Epoch {ep+1:02d} | Loss {loss:.4f} | Val F1 N {f1n:.3f} | Val F1 P {f1p:.3f}")

        if score > best:
            best = score
            torch.save(model.state_dict(), args.save_path)
            print("✅ Saved best model")

    # ---------- TEST ----------
    model.load_state_dict(torch.load(args.save_path))
    f1n, f1p, pn, pp, yn, yp = eval_model(model, test_loader, device)

    names = ["negative", "neutral", "positive"]

    print("\n🎯 FINAL TEST RESULTS")
    print(f"F1 N: {f1n:.3f} | F1 P: {f1p:.3f}")

    print("\nReport (n_change_type)")
    print(classification_report(yn, pn, target_names=names, zero_division=0))
    print(confusion_matrix(yn, pn))

    print("\nReport (p_change_type)")
    print(classification_report(yp, pp, target_names=names, zero_division=0))
    print(confusion_matrix(yp, pp))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./rating_trend/processed_body_trends")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save_path", default="./rating_trend/MSN/best_msn_model.pth")
    args = ap.parse_args()

    main(args)
