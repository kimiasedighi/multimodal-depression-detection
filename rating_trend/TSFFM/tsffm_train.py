# tsffm_train.py

import os, argparse, torch, numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from tsffm_model import TSFFM
from resres_datasets import FusionPoseDataset


def seed_all(s=42):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def subject_split(ds, test_ratio=0.2, val_ratio=0.1, seed=42):
    subjects = defaultdict(list)
    for i in range(len(ds)):
        subjects[ds[i]["subject_id"]].append(i)

    rng = np.random.RandomState(seed)
    sids = list(subjects.keys())
    rng.shuffle(sids)

    n_test = int(len(sids) * test_ratio)
    n_val  = int((len(sids) - n_test) * val_ratio)

    test_sids  = sids[:n_test]
    val_sids   = sids[n_test:n_test+n_val]
    train_sids = sids[n_test+n_val:]

    def collect(which):
        return [i for sid in which for i in subjects[sid]]

    return collect(train_sids), collect(val_sids), collect(test_sids)


def compute_class_weights(ds, indices, label_key, num_classes):
    labels = [ds[i][label_key] for i in indices]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / (num_classes * counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32), counts


@torch.no_grad()
def evaluate(model, loader, device, label_key):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        b = batch["body"].to(device)
        f = batch["face"].to(device)
        out = model(b, f)
        logits = out["logits_final"]
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        labels.extend(batch[label_key].cpu().tolist())

    acc = (np.array(preds) == np.array(labels)).mean() * 100
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)
    return acc, mf1, preds, labels


def main(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = FusionPoseDataset(args.body_dir, args.face_dir)
    tr, va, te = subject_split(ds, seed=args.seed)

    train_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(ds, va), batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(Subset(ds, te), batch_size=args.batch_size, shuffle=False)

    w, counts = compute_class_weights(ds, tr, args.label, args.num_classes)
    w = w.to(device)
    print(f"Train label counts for {args.label}: {counts.tolist()}  weights: {w.detach().cpu().tolist()}")

    model = TSFFM(num_classes=args.num_classes, pretrained=False, fle_dim=args.fle_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0

        for batch in train_loader:
            b = batch["body"].to(device)
            f = batch["face"].to(device)
            y = batch[args.label].to(device)

            opt.zero_grad()
            loss, _ = model(b, f, labels=y, class_weights=w)
            loss.backward()
            opt.step()
            run_loss += loss.item()

        val_acc, val_f1, _, _ = evaluate(model, val_loader, device, args.label)
        print(f"Epoch {epoch}/{args.epochs}  Loss {run_loss/len(train_loader):.4f}  ValAcc {val_acc:.2f}%  ValF1 {val_f1:.3f}")

        if val_f1 > best:
            best = val_f1
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print("✅ Saved best model!")

    model.load_state_dict(torch.load(args.save_path, map_location=device))
    test_acc, test_f1, preds, labels = evaluate(model, test_loader, device, args.label)

    print(f"\n🎯 TEST  Acc: {test_acc:.2f}%   MacroF1: {test_f1:.3f}")
    names = ["negative", "neutral", "positive"]
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=names, digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser("TSFFM Training (trend labels)")
    ap.add_argument("--body_dir", type=str, default="./rating_trend/processed_body_trends")
    ap.add_argument("--face_dir", type=str, default="./rating_trend/processed_face_trends")

    ap.add_argument("--label", type=str, default="label_n", choices=["label_n", "label_p"])
    ap.add_argument("--num_classes", type=int, default=3)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--fle_dim", type=int, default=256)

    ap.add_argument("--save_path", default="./rating_trend/TSFFM/best_tsffm.pth")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    main(args)
    