import numpy as np


def skip_clone_repo_check(func):
    """
    When running model zoo functions, the user sometimes needs to type "y"
    before the repo is cloned. When testing in CI, we want to skip this check.

    Add this function as a decorator to any test function that needs to bypass this.

    @skip_clone_repo_check
    def test_fn():
        ...
    """

    def wrapper(monkeypatch, *args, **kwargs):
        monkeypatch.setattr("builtins.input", lambda: "y")
        func(*args, **kwargs)

    return wrapper


def assert_most_same(arr1: np.ndarray, arr2: np.ndarray, tolerance: float) -> None:
    """
    Checks whether most values in the two numpy arrays are the same.

    Particularly for image models, slight differences in the PIL/cv2 envs
    may cause image <-> tensor conversion to be slightly different.

    Instead of using np.assert_allclose, this may be a better way to test image outputs.

    Parameters:
        arr1: First input image array.
        arr2: Second input image array.
        threshold: Float in range [0,1] representing percentage of values
            that can be different while still having the assertion pass.

    Raises:
        AssertionError if input arrays are different size,
            or too many values are different.
    """

    different_values = arr1 != arr2
    assert (
        np.mean(different_values) <= tolerance
    ), f"More than {tolerance * 100}% of values were different."
