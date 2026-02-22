# msn_train.py

import os
import torch
import argparse
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from msn_dataset import PoseDataset
from msn_body import MSNBody


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        data = data.permute(0, 1, 2, 3)  # [B, C, T, J]

        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            data = data.permute(0, 1, 2, 3)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    return acc, all_preds, all_labels


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PoseDataset(args.data_dir)
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False)

    model = MSNBody(in_channels=3, num_joints=11, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print("✅ Saved best model!")

    # Final evaluation on test set
    model.load_state_dict(torch.load(args.save_path))
    test_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"\n🎯 Final Test Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Healthy", "Depressed"]))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MSN model on pose data")
    parser.add_argument('--data_dir', type=str, default="./processed_body", help="Path to processed .pt data")
    parser.add_argument('--save_path', type=str, default="./MSN/best_msn_model.pth", help="Path to save best model")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=2)

    args = parser.parse_args()
    main(args)

# import os
# import argparse
# import torch
# import numpy as np
# from collections import defaultdict, Counter
# from torch import nn, optim
# from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix

# from msn_dataset import PoseDataset
# from msn_body import MSNBody


# # --------------------------------------------------
# # Utils
# # --------------------------------------------------
# def seed_all(seed=42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def get_subject_id_from_path(path):
#     """
#     Example filename: 0123_t2_ei.pt  ->  0123
#     """
#     return os.path.basename(path).split("_")[0]


# # --------------------------------------------------
# # Subject-level split
# # --------------------------------------------------
# def subject_split(dataset, test_size=0.2, val_size=0.1, seed=42):
#     subject_to_indices = defaultdict(list)

#     for idx, path in enumerate(dataset.data_files):
#         sid = get_subject_id_from_path(path)
#         subject_to_indices[sid].append(idx)

#     subjects = list(subject_to_indices.keys())
#     labels = [dataset[subject_to_indices[s][0]][1] for s in subjects]

#     # Train / Test split (by subject)
#     train_s, test_s = train_test_split(
#         subjects,
#         test_size=test_size,
#         stratify=labels,
#         random_state=seed
#     )

#     # Train / Val split (by subject)
#     train_labels = [dataset[subject_to_indices[s][0]][1] for s in train_s]
#     train_s, val_s = train_test_split(
#         train_s,
#         test_size=val_size,
#         stratify=train_labels,
#         random_state=seed
#     )

#     def flatten(sids):
#         return [i for s in sids for i in subject_to_indices[s]]

#     train_idx = flatten(train_s)
#     val_idx   = flatten(val_s)
#     test_idx  = flatten(test_s)

#     print("Train:", Counter([dataset[i][1] for i in train_idx]))
#     print("Val:  ", Counter([dataset[i][1] for i in val_idx]))
#     print("Test: ", Counter([dataset[i][1] for i in test_idx]))

#     return train_idx, val_idx, test_idx


# # --------------------------------------------------
# # Train / Eval
# # --------------------------------------------------
# def train_one_epoch(model, loader, criterion, optimizer, device):
#     model.train()
#     total = 0.0

#     for data, labels in loader:
#         data = data.to(device)      # [B,C,T,J]
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         total += loss.item()

#     return total / len(loader)


# @torch.no_grad()
# def evaluate(model, loader, device):
#     model.eval()
#     preds, labels_all = [], []

#     for data, labels in loader:
#         data = data.to(device)
#         labels = labels.to(device)

#         outputs = model(data)
#         p = outputs.argmax(dim=1)

#         preds.extend(p.cpu().tolist())
#         labels_all.extend(labels.cpu().tolist())

#     acc = (np.array(preds) == np.array(labels_all)).mean() * 100
#     return acc, preds, labels_all


# # --------------------------------------------------
# # Main
# # --------------------------------------------------
# def main(args):
#     seed_all()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     dataset = PoseDataset(args.data_dir)

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

#     model = MSNBody(
#         in_channels=3,
#         num_joints=11,
#         num_classes=args.num_classes
#     ).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)

#     best = -1
#     for epoch in range(1, args.epochs + 1):
#         loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
#         val_acc, _, _ = evaluate(model, val_loader, device)

#         print(f"Epoch {epoch}/{args.epochs}  "
#               f"Loss {loss:.4f}  ValAcc {val_acc:.2f}%")

#         if val_acc > best:
#             best = val_acc
#             torch.save(model.state_dict(), args.save_path)
#             print("✅ Saved best model!")

#     # ---------------- Test ----------------
#     model.load_state_dict(torch.load(args.save_path, map_location=device))
#     test_acc, preds, labels = evaluate(model, test_loader, device)

#     print("\n🎯 Test Accuracy:", test_acc)
#     print("\nClassification Report:")
#     print(classification_report(labels, preds, digits=4))
#     print("Confusion Matrix:")
#     print(confusion_matrix(labels, preds))


# # --------------------------------------------------
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser("MSN Training (Subject-Level)")
#     ap.add_argument("--data_dir", default="./processed_body")
#     ap.add_argument("--save_path", default="./MSN/best_msn_subject.pth")
#     ap.add_argument("--batch_size", type=int, default=16)
#     ap.add_argument("--lr", type=float, default=1e-3)
#     ap.add_argument("--epochs", type=int, default=20)
#     ap.add_argument("--num_classes", type=int, default=2)
#     args = ap.parse_args()

#     main(args)
