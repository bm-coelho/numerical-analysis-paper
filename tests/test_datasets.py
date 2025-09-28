import numpy as np

from regkit.datasets import create_target_dataset, gen_exploration_dataset


def test_gen_exploration_dataset_shapes_and_seed():
    X1, Y1 = gen_exploration_dataset(32, 3, 2, seed=42)
    X2, Y2 = gen_exploration_dataset(32, 3, 2, seed=42)

    assert X1.shape == (3, 32)
    assert Y1.shape == (2, 32)
    assert np.allclose(X1, X2)
    assert np.allclose(Y1, Y2)


def test_gen_exploration_dataset_noise_changes_samples():
    X_clean, Y_clean = gen_exploration_dataset(16, 2, 2, seed=0, noise=0.0)
    X_noisy, Y_noisy = gen_exploration_dataset(16, 2, 2, seed=0, noise=0.5)

    combined_diff = np.linalg.norm(np.vstack([X_clean - X_noisy, Y_clean - Y_noisy]))
    assert combined_diff > 0.0


def test_create_target_dataset_structure_and_values():
    X, Y = create_target_dataset()

    assert X.shape == (4, 20)
    assert Y.shape == (4, 20)

    # First column corresponds to t=0, q=1 as in the MATLAB prototype.
    np.testing.assert_allclose(X[:, 0], [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(Y[:, 0], [2.0, 1.0, -2.0, 5.0])
