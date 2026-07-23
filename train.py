import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import ChristmasImages
from model import Network


def main():
    train_path = os.path.join("data", "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = ChristmasImages(train_path, training=True)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    model = Network(num_classes=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 15

    model.train()
    for epoch in range(num_epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)

        print(f"epoch {epoch+1}: loss={loss_sum/total:.4f}, acc={correct/total:.4f}")

    model.cpu()
    model.save_model()
    print("Saved model.pkl")


if __name__ == "__main__":
    main()
