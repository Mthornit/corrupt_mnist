import torch
from sklearn.manifold import TSNE
import typer
import matplotlib.pyplot as plt
from src.corrupt_mnist.model import MyAwesomeModel

def visualize(model_checkpoint: str = "models/model.pth", figure_name: str = "embeddings.png") -> None:
    """Visualize model predictions."""
    model: torch.nn.Module = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")
    test_dataset = torch.utils.data.TensorDataset(test_images, test_target)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(test_dataset, batch_size=64):
            images, target = batch
            _ = model(images)
            predictions = model.cnn_features.squeeze()
            embeddings.append(predictions)
            targets.append(target)
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")


if __name__ == "__main__":
    typer.run(visualize)