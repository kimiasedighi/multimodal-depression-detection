# resres_eval.py

import os, argparse, torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from resres_model import ResRes18DecisionFusion
from resres_datasets import PoseAsImageDataset, FusionPoseDataset

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = FusionPoseDataset(args.body_dir, args.face_dir) if args.input_type=="F+B" else PoseAsImageDataset(args.body_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = ResRes18DecisionFusion(num_classes=args.num_classes, input_type=args.input_type, fusion=args.fusion)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if args.input_type == "B":
                out = model(body_img=batch["body"].to(device)); logits = out["logits_body"]
            elif args.input_type == "F":
                out = model(face_img=batch["face"].to(device)); logits = out["logits_face"]
            else:
                out = model(body_img=batch["body"].to(device), face_img=batch["face"].to(device)); logits = out["logits_fused"]
            preds += torch.argmax(logits, 1).cpu().tolist()
            labels += batch["label"].tolist()

    acc = (np.array(preds)==np.array(labels)).mean()*100
    print(f"Accuracy: {acc:.2f}%")
    print(classification_report(labels, preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Eval Res-Res")
    ap.add_argument("--input_type", type=str, default="B", choices=["B","F","F+B"])
    ap.add_argument("--body_dir", type=str, default="./processed_body")
    ap.add_argument("--face_dir", type=str, default="./processed_face")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--fusion", type=str, default="avg_logit", choices=["avg_logit","avg_prob"])
    ap.add_argument("--model_path", type=str, default="Res_Res/best_resres18.pth")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    if args.input_type in ("F","F+B") and not args.face_dir:
        ap.error("face_dir is required when input_type is F or F+B")
    main(args)
