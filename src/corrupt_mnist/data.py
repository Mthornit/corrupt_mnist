import typer
import torch
from torchvision import transforms

def preprocess_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> None:
    """Process raw data and save it to processed directory."""
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    trainimg_paths = [f"{raw_dir}/train_images_{i}.pt" for i in range(6)]
    traintarget_paths = [f"{raw_dir}/train_target_{i}.pt" for i in range(6)]
    testimg_path = f"{raw_dir}/test_images.pt"
    testtarget_path= f"{raw_dir}/test_target.pt"
    
    train = torch.cat([torch.load(path) for path in trainimg_paths], dim=0).float()
    train_labels = torch.cat([torch.load(path) for path in traintarget_paths], dim=0).long()
    test = torch.load(testimg_path).float()
    test_labels = torch.load(testtarget_path).long()
    
    train = normalize(train)
    test = normalize(test)

    torch.save(train, f"{processed_dir}/train_images.pt")
    torch.save(train_labels, f"{processed_dir}/train_target.pt")
    torch.save(test, f"{processed_dir}/test_images.pt")
    torch.save(test_labels, f"{processed_dir}/test_target.pt")


def corrupt_mnist_data() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
