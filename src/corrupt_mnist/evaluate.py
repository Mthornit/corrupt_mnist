import torch
from src.corrupt_mnist.data import corrupt_mnist_data
from src.corrupt_mnist.model import MyAwesomeModel
from torch.utils.data import DataLoader
import typer

train_data, test_data = corrupt_mnist_data()
batch_size = 64

model = MyAwesomeModel()

def evaluate(model_checkpoint: str = "models/model.pth"):
    accuracy = 0
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    for x, y in test_loader:
        logits = model(x)
        accuracy += torch.sum(logits.argmax(dim=1) == y)
    accuracy = accuracy.item() / (len(test_loader)*batch_size)
    print(f"Test accuracy: {accuracy}")

if __name__ == "__main__":
    typer.run(evaluate)