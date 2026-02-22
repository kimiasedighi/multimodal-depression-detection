# resres_eval.py

import argparse, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from resres_model import ResRes18DecisionFusion
from resres_datasets import PoseAsImageDataset, FusionPoseDataset


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.input_type == "F+B":
        ds = FusionPoseDataset(args.body_dir, args.face_dir)
    elif args.input_type == "F":
        ds = PoseAsImageDataset(args.face_dir)
    else:
        ds = PoseAsImageDataset(args.body_dir)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = ResRes18DecisionFusion(
        num_classes=args.num_classes,
        input_type=args.input_type,
        fusion=args.fusion
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds, labels = [], []

    for batch in loader:
        b = batch.get("body").to(device) if "body" in batch else None
        f = batch.get("face").to(device) if "face" in batch else None

        if args.input_type == "B":
            out = model(body_img=b)
            logits = out["logits_body"]
        elif args.input_type == "F":
            out = model(face_img=f)
            logits = out["logits_face"]
        else:
            out = model(body_img=b, face_img=f)
            logits = out["logits_fused"]

        preds.extend(torch.argmax(logits, 1).cpu().tolist())
        labels.extend(batch[args.label].cpu().tolist())

    acc = (np.array(preds) == np.array(labels)).mean() * 100
    print(f"Accuracy: {acc:.2f}%")

    names = ["negative", "neutral", "positive"]
    print(classification_report(labels, preds, target_names=names, digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Eval Res-Res (trend labels)")
    ap.add_argument("--input_type", type=str, default="B", choices=["B", "F", "F+B"])
    ap.add_argument("--body_dir", type=str, default="./rating_trend/processed_body_trends")
    ap.add_argument("--face_dir", type=str, default="./rating_trend/processed_face_trends")
    ap.add_argument("--label", type=str, default="label_n", choices=["label_n", "label_p"])
    ap.add_argument("--num_classes", type=int, default=3)
    ap.add_argument("--fusion", type=str, default="avg_logit", choices=["avg_logit", "avg_prob"])
    ap.add_argument("--model_path", type=str, default="./rating_trend/Res_Res/best_resres18.pth")
    ap.add_argument("--batch_size", type=int, default=32)

    args = ap.parse_args()
    if args.input_type in ("F", "F+B") and not args.face_dir:
        ap.error("face_dir is required when input_type is F or F+B")
    main(args)
