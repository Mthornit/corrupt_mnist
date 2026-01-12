from corrupt_mnist.data import corrupt_mnist_data
import torch
import pytest
import os.path
from tests import _PATH_DATA as file_path

@pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist_data()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (28, 28), "incorrect x shape. Should be 28x28 but was: " + str(x.shape)
            assert y in range(10), "y shape had an impossible label: " + str(y)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all(), "not all numbers represented in train"
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all(), "not all numbers represented in test"
