"""
Train a small CNN on MNIST and export weights to JSON for the browser demo.

Architecture is intentionally shaped to map onto the 13 visualization steps:
  Conv1 (1->8, 3x3, pad 1) -> ReLU -> MaxPool 2   : 28 -> 28 -> 14
  Conv2 (8->16, 3x3, pad 1) -> ReLU -> MaxPool 2   : 14 -> 14 -> 7
  Flatten (16*7*7 = 784)
  FC1 (784 -> 64) -> ReLU
  FC2 (64 -> 10) -> Softmax
"""
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    tf = transforms.Compose([transforms.ToTensor()])  # [0,1], no normalization (matches browser preprocessing)
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test = datasets.MNIST("./data", train=False, download=True, transform=tf)
    train_dl = DataLoader(train, batch_size=256, shuffle=True)
    test_dl = DataLoader(test, batch_size=512)

    model = SmallCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (model(xb).argmax(1) == yb).sum().item()
        print(f"epoch {epoch+1}: test acc = {correct/len(test):.4f}")

    # Export weights
    sd = model.state_dict()
    out = {k: v.cpu().numpy().flatten().tolist() for k, v in sd.items()}
    shapes = {k: list(v.shape) for k, v in sd.items()}
    with open("model_weights.json", "w") as f:
        json.dump({"weights": out, "shapes": shapes}, f)
    print("saved model_weights.json", {k: shapes[k] for k in shapes})


if __name__ == "__main__":
    main()
