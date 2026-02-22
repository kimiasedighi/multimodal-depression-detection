# tsffm_eval.py

import torch, argparse, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from tsffm_model import TSFFM
from resres_datasets import FusionPoseDataset


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = FusionPoseDataset(args.body_dir, args.face_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = TSFFM(num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            b = batch["body"].to(device)
            f = batch["face"].to(device)
            out = model(b, f)
            logits = out["logits_final"]
            preds += torch.argmax(logits, 1).cpu().tolist()
            labels += batch["label"].tolist()

    acc = (np.array(preds) == np.array(labels)).mean() * 100
    print("Accuracy:", acc)
    print(classification_report(labels, preds, digits=4))
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--body_dir", default="processed_body")
    ap.add_argument("--face_dir", default="processed_face")
    ap.add_argument("--model_path", default="./TSFFM/best_tsffm.pth")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    main(args)
