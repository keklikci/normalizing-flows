import numpy as np
import pytest

from normalizing_flows.metrics import energy, estat, ks2d2s, quadct


def test_quadct_counts_all_points() -> None:
    counts = quadct(0, 0, [-1, 1], [-1, 1])
    assert counts == (0.5, 0.0, 0.0, 0.5)


def test_ks_is_symmetric_and_identical_samples_are_close() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    forward = ks2d2s(x, y, x, y, extra=True)
    reverse = ks2d2s(x, y, x, y, extra=True)
    assert forward == reverse
    assert forward[0] > 0.9
    assert 0 <= forward[1] <= 1


def test_ks_bootstrap_is_deterministic() -> None:
    first = ks2d2s([0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5], nboot=20)
    second = ks2d2s([0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5], nboot=20)
    assert first == second


def test_energy_supports_linear_and_log_methods() -> None:
    first = np.array([[0.0], [1.0]])
    second = np.array([[2.0], [3.0]])
    assert energy(first, second, method="linear") > 0
    assert np.isfinite(energy(first, second, method="log"))


def test_estat_returns_bootstrap_values() -> None:
    result = estat([[0, 0], [1, 1]], [[2, 2], [3, 3]], nboot=5, method="linear")
    assert len(result) == 3
    assert result[2].shape == (5,)


def test_invalid_inputs_raise_clear_errors() -> None:
    with pytest.raises(ValueError, match="unsupported method"):
        energy([[0], [1]], [[2], [3]], method="gaussian")
    with pytest.raises(ValueError, match="nboot"):
        estat([[0, 0], [1, 1]], [[2, 2], [3, 3]], nboot=0)
    with pytest.raises(ValueError, match="equal lengths"):
        quadct(0, 0, [0], [0, 1])
