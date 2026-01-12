import torch
import torch.nn as nn
from data import corrupt_mnist_data
from model import MyAwesomeModel
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import typer
import os
import wandb
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score


load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key)

train_data, test_data = corrupt_mnist_data()
batch_size = 64

model = MyAwesomeModel()

def train(lr: float = 1e-3, batch_size: int = 64, epochs: int = 10) -> None:
    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mthornit-team",
    # Set the wandb project where this run will be logged.
    project="corrupt_mnist",
    # Track hyperparameters and run metadata.
    config={
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
    }
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_losses = []
    train_accuracies = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #wandb.watch(model, log="all")
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        preds, targets = [], []
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
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(logits.detach().cpu())
            targets.append(y.detach().cpu())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

                # add a plot of the input images
                images = wandb.Image(x[0].detach().unsqueeze(0).cpu(), caption="Input images")
                wandb.log({"images": images})

                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads.cpu())})

        # add a custom matplotlib plot of the ROC curves
        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

        for class_id in range(10):
            one_hot = torch.zeros_like(targets)
            one_hot[targets == class_id] = 1
            _ = RocCurveDisplay.from_predictions(
                one_hot.cpu(),
                preds[:, class_id],
                name=f"ROC curve for {class_id}",
                plot_chance_level=(class_id == 2),
            )

        # alternatively use wandb.log({"roc": wandb.Image(plt)}
        wandb.log({"roc": wandb.Image(plt)})
        plt.close()  # close the plot to avoid memory leaks and overlapping figures

    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    torch.save(model.state_dict(), "model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("model.pth")
    run.log_artifact(artifact)

if __name__ == "__main__":
    typer.run(train)
