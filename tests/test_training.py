import wandb
import os

def test_traning():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)
    except:
        pass

    run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="mthornit-team",
    # Set the wandb project where this run will be logged.
    project="corrupt_mnist",
    # Not a new experiment, just reading data
    job_type="read",
    settings=wandb.Settings(
        host="localhost"
    )
    )
    
    # Retrieve the latest version of the artifact
    artifact = wandb.use_artifact('mthornit-team/corrupt_mnist/corrupt_mnist_model:v0', type='model')

    # Access metadata
    accuracy = artifact.metadata['accuracy']
    assert accuracy >= 0.9, "accuracy was too low: " + str(accuracy)