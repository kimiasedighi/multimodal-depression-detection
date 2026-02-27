# msn_eval.py 

import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from msn_dataset import PoseDataset
from msn_body import MSNBody


# ---------- SUBJECT SPLIT (STATIC: same defaults as training) ----------
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


@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    pn, pp, yn, yp = [], [], [], []

    for x, y_n, y_p, _ in loader:
        x = x.to(device)
        out_n, out_p = model(x)

        pn.extend(out_n.argmax(1).cpu().tolist())
        pp.extend(out_p.argmax(1).cpu().tolist())
        yn.extend(y_n.cpu().tolist())
        yp.extend(y_p.cpu().tolist())

    f1_n = f1_score(yn, pn, average="macro", zero_division=0)
    f1_p = f1_score(yp, pp, average="macro", zero_division=0)
    return f1_n, f1_p, pn, pp, yn, yp


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("🚀 Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ Using CPU")

    dataset = PoseDataset(args.data_dir)

    # STATIC: same as training (uses defaults seed=42, test=0.2, val=0.1)
    _, _, te_idx = subject_split(dataset)

    test_loader = DataLoader(
        Subset(dataset, te_idx),
        batch_size=args.batch,
        shuffle=False
    )

    model = MSNBody().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    f1n, f1p, pn, pp, yn, yp = eval_model(model, test_loader, device)

    names = ["negative", "neutral", "positive"]

    print("\n🎯 FINAL TEST RESULTS")
    print(f"F1 N: {f1n:.3f} | F1 P: {f1p:.3f}")

    print("\nReport (n_change_type)")
    print(classification_report(yn, pn, target_names=names, zero_division=0))
    print("Confusion Matrix (n_change_type):")
    print(confusion_matrix(yn, pn))

    print("\nReport (p_change_type)")
    print(classification_report(yp, pp, target_names=names, zero_division=0))
    print("Confusion Matrix (p_change_type):")
    print(confusion_matrix(yp, pp))


if __name__ == "__main__":
    ap = argparse.ArgumentParser("Evaluate MSN trend model (static split)")
    ap.add_argument("--data_dir", default="./rating_trend/processed_body_trends")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    main(args)
