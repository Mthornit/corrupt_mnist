from fastapi.testclient import TestClient
from corrupt_mnist.api import app
client = TestClient(app)
