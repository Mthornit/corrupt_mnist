import torch
import torch.nn as nn
from src.corrupt_mnist.data import corrupt_mnist_data
from src.corrupt_mnist.model import MyAwesomeModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import typer

train_data, test_data = corrupt_mnist_data()
batch_size = 64

model = MyAwesomeModel()

def train(lr: float = 1e-3, batch_size: int = 64, epochs: int = 10) -> None:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_losses = []
    train_accuracies = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_losses.append(loss.item())
            accuracy = (logits.argmax(dim=1) == y).float().mean().item()
            train_accuracies.append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_losses)
    axs[0].set_title("Train loss")
    axs[1].plot(train_accuracies)
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")

if __name__ == "__main__":
    typer.run(train)
