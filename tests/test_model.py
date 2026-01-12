from corrupt_mnist.model import MyAwesomeModel
import torch
import pytest

model = MyAwesomeModel()

def test_model():
    x = torch.randn(1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10), f"model returns wrong y shape. Should be (1,10) but was {y.shape}"

def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Expected input to a 3D tensor: Batchsize, h, w'):
        model(torch.randn(4, 1, 28, 28))