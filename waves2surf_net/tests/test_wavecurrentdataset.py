import numpy as np
import pytest
import torch

from ocean_velocity.wavecurrentdataset import WaveCurrentDataset


def test_random_patch_is_aligned_and_has_requested_shape():
    dataset = object.__new__(WaveCurrentDataset)
    dataset.patch_size = (4, 5)
    base = np.arange(8 * 9, dtype=np.float32).reshape(8, 9)
    x = np.stack((base, base + 100))
    y = np.stack((base + 200, base + 300))
    valid = (base % 2) == 0

    torch.manual_seed(7)
    cropped_x, cropped_y, cropped_valid = dataset._random_patch(x, y, valid)

    assert cropped_x.shape == (2, 4, 5)
    assert cropped_y.shape == (2, 4, 5)
    assert cropped_valid.shape == (4, 5)
    np.testing.assert_array_equal(cropped_y[0] - cropped_x[0], 200)
    np.testing.assert_array_equal(cropped_x[1] - cropped_x[0], 100)
    np.testing.assert_array_equal(cropped_valid, (cropped_x[0] % 2) == 0)


def test_random_patch_rejects_oversized_window():
    dataset = object.__new__(WaveCurrentDataset)
    dataset.patch_size = (9, 5)
    x = np.zeros((1, 8, 9), dtype=np.float32)

    with pytest.raises(ValueError, match="exceeds sample shape"):
        dataset._random_patch(x, x, np.ones((8, 9), dtype=bool))
