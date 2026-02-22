# tsffm_train.py

import torch, argparse, numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import sys
sys.path.append("Res_Res")

from tsffm_model import TSFFM
from resres_datasets import FusionPoseDataset


def seed_all(s=42):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        b = batch["body"].to(device)
        f = batch["face"].to(device)
        out = model(b, f)
        logits = out["logits_final"]
        preds += torch.argmax(logits, 1).cpu().tolist()
        labels += batch["label"].tolist()
    acc = (np.array(preds) == np.array(labels)).mean() * 100
    return acc, preds, labels


def main(args):
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = FusionPoseDataset(args.body_dir, args.face_dir)
    idx = list(range(len(ds)))
    y_all = [ds[i]["label"] for i in idx]

    tr, te = train_test_split(idx, test_size=0.2, stratify=y_all, random_state=42)
    tr, va = train_test_split(tr, test_size=0.1,
                              stratify=[y_all[i] for i in tr], random_state=42)

    train_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, va), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(ds, te), batch_size=args.batch_size, shuffle=False)

    model = TSFFM(num_classes=2, pretrained=args.pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best = -1
    for epoch in range(1, args.epochs+1):
        model.train()
        run_loss = 0
        for batch in train_loader:
            b = batch["body"].to(device)
            f = batch["face"].to(device)
            y = batch["label"].to(device)

            opt.zero_grad()
            loss, _ = model(b, f, y)
            loss.backward()
            opt.step()
            run_loss += loss.item()

        val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  Loss {run_loss/len(train_loader):.4f}  ValAcc {val_acc:.2f}%")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), args.save_path)
            print("✅ Saved best model!")

    model.load_state_dict(torch.load(args.save_path, map_location=device))
    test_acc, preds, labels = evaluate(model, test_loader, device)

    print("\n🎯 TestAcc:", test_acc)
    print(classification_report(labels, preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser("TSFFM Training")
    ap.add_argument("--body_dir", default="processed_body")
    ap.add_argument("--face_dir", default="processed_face")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--save_path", default="./TSFFM/best_tsffm.pth")
    args = ap.parse_args()
    main(args)

# import os
# import argparse
# import torch
# import numpy as np
# from collections import defaultdict, Counter
# from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix

# from tsffm_model import TSFFM
# from resres_datasets import FusionPoseDataset


# # --------------------------------------------------
# # Utils
# # --------------------------------------------------
# def seed_all(seed=42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def get_subject_id(name):
#     """
#     Extract subject ID from filename.
#     Example: 0123_t2_ei.pt -> 0123
#     """
#     return name.split("_")[0]


# # --------------------------------------------------
# # Subject-level split
# # --------------------------------------------------
# def subject_split(dataset, test_size=0.2, val_size=0.1, seed=42):
#     subject_to_indices = defaultdict(list)

#     for idx in range(len(dataset)):
#         sid = get_subject_id(dataset[idx]["name"])
#         subject_to_indices[sid].append(idx)

#     subjects = list(subject_to_indices.keys())
#     labels = []
#     for sid in subjects:
#         labels.append(dataset[subject_to_indices[sid][0]]["label"])

#     # train / test subjects
#     train_s, test_s = train_test_split(
#         subjects, test_size=test_size, stratify=labels, random_state=seed
#     )

#     # train / val subjects
#     train_labels = [
#         dataset[subject_to_indices[sid][0]]["label"] for sid in train_s
#     ]
#     train_s, val_s = train_test_split(
#         train_s, test_size=val_size, stratify=train_labels, random_state=seed
#     )

#     def flatten(sids):
#         return [i for sid in sids for i in subject_to_indices[sid]]

#     train_idx = flatten(train_s)
#     val_idx   = flatten(val_s)
#     test_idx  = flatten(test_s)

#     print("Train:", Counter([dataset[i]["label"] for i in train_idx]))
#     print("Val:  ", Counter([dataset[i]["label"] for i in val_idx]))
#     print("Test: ", Counter([dataset[i]["label"] for i in test_idx]))

#     return train_idx, val_idx, test_idx


# # --------------------------------------------------
# # Evaluation
# # --------------------------------------------------
# @torch.no_grad()
# def evaluate(model, loader, device):
#     model.eval()
#     preds, labels = [], []

#     for batch in loader:
#         b = batch["body"].to(device)
#         f = batch["face"].to(device)
#         out = model(b, f)
#         logits = out["logits_final"]

#         preds.extend(torch.argmax(logits, 1).cpu().tolist())
#         labels.extend(batch["label"].tolist())

#     acc = (np.array(preds) == np.array(labels)).mean() * 100
#     return acc, preds, labels


# # --------------------------------------------------
# # Main
# # --------------------------------------------------
# def main(args):
#     seed_all()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     dataset = FusionPoseDataset(args.body_dir, args.face_dir)

#     train_idx, val_idx, test_idx = subject_split(
#         dataset,
#         test_size=0.2,
#         val_size=0.1,
#         seed=42
#     )

#     train_loader = DataLoader(
#         Subset(dataset, train_idx),
#         batch_size=args.batch_size,
#         shuffle=True
#     )
#     val_loader = DataLoader(
#         Subset(dataset, val_idx),
#         batch_size=args.batch_size,
#         shuffle=False
#     )
#     test_loader = DataLoader(
#         Subset(dataset, test_idx),
#         batch_size=args.batch_size,
#         shuffle=False
#     )

#     model = TSFFM(num_classes=2, pretrained=args.pretrained).to(device)
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=args.lr,
#         weight_decay=1e-4
#     )

#     best = -1
#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         running = 0.0

#         for batch in train_loader:
#             b = batch["body"].to(device)
#             f = batch["face"].to(device)
#             y = batch["label"].to(device)

#             optimizer.zero_grad()
#             loss, _ = model(b, f, y)
#             loss.backward()
#             optimizer.step()

#             running += loss.item()

#         val_acc, _, _ = evaluate(model, val_loader, device)
#         print(f"Epoch {epoch}/{args.epochs}  "
#               f"Loss {running/len(train_loader):.4f}  "
#               f"ValAcc {val_acc:.2f}%")

#         if val_acc > best:
#             best = val_acc
#             torch.save(model.state_dict(), args.save_path)
#             print("✅ Saved best model!")

#     # ----------------- Test -----------------
#     model.load_state_dict(torch.load(args.save_path, map_location=device))
#     test_acc, preds, labels = evaluate(model, test_loader, device)

#     print("\n🎯 Test Accuracy:", test_acc)
#     print("\nClassification Report:")
#     print(classification_report(labels, preds, digits=4))
#     print("Confusion Matrix:")
#     print(confusion_matrix(labels, preds))


# # --------------------------------------------------
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser("TSFFM Training (Subject-Level)")
#     ap.add_argument("--body_dir", default="./processed_body")
#     ap.add_argument("--face_dir", default="./processed_face")
#     ap.add_argument("--epochs", type=int, default=30)
#     ap.add_argument("--batch_size", type=int, default=4)
#     ap.add_argument("--lr", type=float, default=1e-4)
#     ap.add_argument("--pretrained", action="store_true")
#     ap.add_argument("--save_path", default="./TSFFM/best_tsffm_subject.pth")
#     args = ap.parse_args()

#     main(args)
