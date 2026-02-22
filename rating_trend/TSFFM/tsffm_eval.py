# tsfffm_eval.py

import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from tsffm_model import TSFFM
from resres_datasets import FusionPoseDataset


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = FusionPoseDataset(args.body_dir, args.face_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = TSFFM(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds, labels = [], []
    for batch in loader:
        b = batch["body"].to(device)
        f = batch["face"].to(device)
        out = model(b, f)
        logits = out["logits_final"]
        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        labels.extend(batch[args.label].cpu().tolist())

    acc = (np.array(preds) == np.array(labels)).mean() * 100
    mf1 = f1_score(labels, preds, average="macro", zero_division=0)

    print(f"Accuracy: {acc:.2f}%  MacroF1: {mf1:.3f}")
    names = ["negative", "neutral", "positive"]
    print(classification_report(labels, preds, target_names=names, digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser("TSFFM Eval (trend labels)")
    ap.add_argument("--body_dir", type=str, default="./rating_trend/processed_body_trends")
    ap.add_argument("--face_dir", type=str, default="./rating_trend/processed_face_trends")
    ap.add_argument("--label", type=str, default="label_n", choices=["label_n", "label_p"])
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--model_path", default="./rating_trend/TSFFM/best_tsffm.pth")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    main(args)
    