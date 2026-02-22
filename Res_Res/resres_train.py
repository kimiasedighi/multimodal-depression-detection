# resres_train.py

import os, argparse, torch, numpy as np
from torch import optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from resres_model import ResRes18DecisionFusion
from resres_datasets import PoseAsImageDataset, FusionPoseDataset

def seed_all(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def make_loaders(args):
    
    if args.input_type == "F+B":
        # Use fusion dataset for F+B
        ds = FusionPoseDataset(args.body_dir, args.face_dir)
    elif args.input_type == "F":
        # Use PoseAsImageDataset but point it to the FACE directory
        ds = PoseAsImageDataset(args.face_dir)
    else:
        # Default to Body-only
        ds = PoseAsImageDataset(args.body_dir)

    idx = list(range(len(ds)))
    y_all = [ds[i]["label"] for i in idx]
    tr, te = train_test_split(idx, test_size=0.2, random_state=42, stratify=y_all)
    y_tr = [ds[i]["label"] for i in tr]
    tr, va = train_test_split(tr, test_size=0.1, random_state=42, stratify=y_tr)

    train_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(Subset(ds, va), batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(Subset(ds, te), batch_size=args.batch_size, shuffle=False)
    return ds, train_loader, val_loader, test_loader

def step(model, batch, device, input_type):
    y = batch["label"].to(device)
    
    if input_type == "B":
        b = batch["image"].to(device)
        loss, out = model(body_img=b, labels=y)
        logits = out["logits_body"]
        
    elif input_type == "F":
        f = batch["image"].to(device)
        loss, out = model(face_img=f, labels=y)
        logits = out["logits_face"]
        
    # b = batch.get("body").to(device) if "body" in batch else None
    # f = batch.get("face").to(device) if "face" in batch else None
    else:
        b = batch["body"].to(device)
        f = batch["face"].to(device)
        loss, out = model(body_img=b, face_img=f, labels=y)
        logits = out["logits_fused"]
        
    return loss, logits, y

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
                face_img=batch["face"].to(device)
            )
            logits = out["logits_fused"]

        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        labels.extend(y.tolist())

    acc = (np.array(preds) == np.array(labels)).mean() * 100
    return acc, preds, labels

def main(args):
    seed_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds, train_loader, val_loader, test_loader = make_loaders(args)

    model = ResRes18DecisionFusion(
        num_classes=args.num_classes,
        input_type=args.input_type,
        fusion=args.fusion,
        pretrained=args.pretrained,
        branch_loss_weight=args.branch_loss_w,
        fusion_loss_weight=args.fusion_loss_w,
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = -1
    for epoch in range(1, args.epochs+1):
        model.train()
        run = 0.0
        for batch in train_loader:
            opt.zero_grad()
            loss, _, _ = step(model, batch, device, args.input_type)
            loss.backward(); opt.step()
            run += loss.item()
        val_acc, _, _ = evaluate(model, val_loader, device, args.input_type)
        print(f"Epoch {epoch}/{args.epochs}  Loss {run/len(train_loader):.4f}  ValAcc {val_acc:.2f}%")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), args.save_path)
            print("✅ Saved best model!")

    # Test
    model.load_state_dict(torch.load(args.save_path, map_location=device))
    test_acc, preds, labels = evaluate(model, test_loader, device, args.input_type)
    print(f"\n🎯 Test Acc: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(labels, preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Train Res-Res (ResNet18 x2) with decision fusion")
    ap.add_argument("--input_type", type=str, default="B", choices=["B","F","F+B"])
    ap.add_argument("--body_dir", type=str, default="./processed_body")
    ap.add_argument("--face_dir", type=str, default="./processed_face")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--fusion", type=str, default="avg_logit", choices=["avg_logit","avg_prob"])
    ap.add_argument("--branch_loss_w", type=float, default=1.0)
    ap.add_argument("--fusion_loss_w", type=float, default=1.0)
    ap.add_argument("--save_path", type=str, default="./Res_Res/best_resres18.pth")
    args = ap.parse_args()
    if args.input_type in ("F","F+B") and not args.face_dir:
        ap.error("face_dir is required when input_type is F or F+B")
    main(args)

# import os, argparse, torch, numpy as np
# from collections import defaultdict, Counter
# from torch import optim
# from torch.utils.data import DataLoader, Subset
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix

# from resres_model import ResRes18DecisionFusion
# from resres_datasets import PoseAsImageDataset, FusionPoseDataset

# def seed_all(seed=42):
#     import random
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# def subject_split(ds, test_size=0.2, val_size=0.1, seed=42):
#     subj_to_indices = defaultdict(list)
#     for i in range(len(ds)):
#         subj_to_indices[ds[i]["subject"]].append(i)

#     subjects = list(subj_to_indices.keys())
#     train_subj, test_subj = train_test_split(
#         subjects, test_size=test_size, random_state=seed
#     )
#     train_subj, val_subj = train_test_split(
#         train_subj, test_size=val_size, random_state=seed
#     )

#     def collect(subj_list):
#         idx = []
#         for s in subj_list:
#             idx.extend(subj_to_indices[s])
#         return idx

#     return collect(train_subj), collect(val_subj), collect(test_subj)

# def make_loaders(args):
#     if args.input_type == "F+B":
#         ds = FusionPoseDataset(args.body_dir, args.face_dir)
#     elif args.input_type == "F":
#         ds = PoseAsImageDataset(args.face_dir)
        
#     else:
#         ds = PoseAsImageDataset(args.body_dir)

#     tr, va, te = subject_split(ds)

#     print("Train:", Counter(ds[i]["label"] for i in tr))
#     print("Val:  ", Counter(ds[i]["label"] for i in va))
#     print("Test: ", Counter(ds[i]["label"] for i in te))

#     train_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
#     val_loader   = DataLoader(Subset(ds, va), batch_size=args.batch_size)
#     test_loader  = DataLoader(Subset(ds, te), batch_size=args.batch_size)

#     return ds, train_loader, val_loader, test_loader

# def step(model, batch, device, input_type):
#     b = batch.get("body").to(device) if "body" in batch else None
#     f = batch.get("face").to(device) if "face" in batch else None
#     y = batch["label"].to(device)

#     if input_type == "B":
#         loss, out = model(body_img=b, labels=y)
#         logits = out["logits_body"]
#     elif input_type == "F":
#         loss, out = model(face_img=f, labels=y)
#         logits = out["logits_face"]
#     else:
#         loss, out = model(body_img=b, face_img=f, labels=y)
#         logits = out["logits_fused"]

#     return loss, logits, y

# @torch.no_grad()
# def evaluate(model, loader, device, input_type):
#     model.eval()
#     preds, labels = [], []

#     for batch in loader:
#         if input_type == "B":
#             out = model(body_img=batch["image"].to(device))
#             logits = out["logits_body"]
#         elif input_type == "F":
#             out = model(face_img=batch["image"].to(device))
#             logits = out["logits_face"]
#         else:
#             out = model(
#                 body_img=batch["body"].to(device),
#                 face_img=batch["face"].to(device)
#             )
#             logits = out["logits_fused"]

#         preds.extend(torch.argmax(logits, 1).cpu().tolist())
#         labels.extend(batch["label"].tolist())

#     acc = (np.array(preds) == np.array(labels)).mean() * 100
#     return acc, preds, labels

# def main(args):
#     seed_all()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     ds, train_loader, val_loader, test_loader = make_loaders(args)

#     model = ResRes18DecisionFusion(
#         num_classes=args.num_classes,
#         input_type=args.input_type,
#         fusion=args.fusion,
#         pretrained=args.pretrained,
#         branch_loss_weight=args.branch_loss_w,
#         fusion_loss_weight=args.fusion_loss_w,
#     ).to(device)

#     opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#     best = -1
#     for epoch in range(1, args.epochs+1):
#         model.train()
#         run = 0.0
#         for batch in train_loader:
#             opt.zero_grad()
#             loss, _, _ = step(model, batch, device, args.input_type)
#             loss.backward()
#             opt.step()
#             run += loss.item()

#         val_acc, _, _ = evaluate(model, val_loader, device, args.input_type)
#         print(f"Epoch {epoch}/{args.epochs}  Loss {run/len(train_loader):.4f}  ValAcc {val_acc:.2f}%")

#         if val_acc > best:
#             best = val_acc
#             torch.save(model.state_dict(), args.save_path)
#             print("✅ Saved best model!")

#     model.load_state_dict(torch.load(args.save_path, map_location=device))
#     test_acc, preds, labels = evaluate(model, test_loader, device, args.input_type)

#     print(f"\n🎯 Test Acc: {test_acc:.2f}%")
#     print(classification_report(labels, preds, digits=4))
#     print("Confusion Matrix:")
#     print(confusion_matrix(labels, preds))

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser("Res-Res with SUBJECT-level split")
#     ap.add_argument("--input_type", choices=["B","F","F+B"], default="B")
#     ap.add_argument("--body_dir", default="./processed_body")
#     ap.add_argument("--face_dir", default="./processed_face")
#     ap.add_argument("--epochs", type=int, default=20)
#     ap.add_argument("--batch_size", type=int, default=16)
#     ap.add_argument("--lr", type=float, default=1e-3)
#     ap.add_argument("--weight_decay", type=float, default=1e-4)
#     ap.add_argument("--num_classes", type=int, default=2)
#     ap.add_argument("--fusion", choices=["avg_logit","avg_prob"], default="avg_logit")
#     ap.add_argument("--pretrained", action="store_true")
#     ap.add_argument("--branch_loss_w", type=float, default=1.0)
#     ap.add_argument("--fusion_loss_w", type=float, default=1.0)
#     ap.add_argument("--save_path", default="./Res_Res/best_resres18_subjectsplit.pth")
#     args = ap.parse_args()

#     if args.input_type in ("F","F+B") and not args.face_dir:
#         ap.error("face_dir required for F or F+B")

#     main(args)
