import os
import csv
import torch
from torch.utils.data import DataLoader

from data import ChristmasImages
from model import Network


def main():
    val_path = os.path.join("data", "val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ChristmasImages(val_path, training=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    model = Network(num_classes=8)
    model.load_state_dict(torch.load("model.pkl", map_location=device))
    model.to(device)
    model.eval()

    rows = []
    idx = 0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1).cpu().tolist()
            for p in preds:
                rows.append((idx, p))
                idx += 1

    with open("submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "Category"])
        writer.writerows(rows)

    print("Wrote submission.csv")


if __name__ == "__main__":
    main()
