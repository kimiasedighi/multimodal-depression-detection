# msn_eval.py

import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from msn_dataset import PoseDataset
from msn_body import MSNBody

NUM_CLASSES = 2


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

    # Load dataset
    dataset = PoseDataset(args.data_dir)

    # Recreate SAME split as training
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False
    )

    # Load model
    model = MSNBody(in_channels=3, num_joints=11, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Evaluate
    test_acc, preds, labels = evaluate(model, test_loader, device)

    print(f"\n🎯 TEST Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Healthy", "Depressed"]))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MSN model")
    parser.add_argument('--data_dir', type=str, default="./symptom_classification/processed_body")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)

    args = parser.parse_args()
    main(args)