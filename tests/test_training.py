import wandb
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
wandb.login(key=api_key)

run = wandb.init(
# Set the wandb entity where your project will be logged (generally your team name).
entity="mthornit-team",
# Set the wandb project where this run will be logged.
project="corrupt_mnist",
# Not a new experiment, just reading data
job_type="read"
)

def test_traning():
    # Retrieve the latest version of the artifact
    artifact = wandb.use_artifact('mthornit-team/corrupt_mnist/corrupt_mnist_model:v0', type='model')

    # Access metadata
    accuracy = artifact.metadata['accuracy']
    assert accuracy >= 0.9, "accuracy was too low: " + str(accuracy)