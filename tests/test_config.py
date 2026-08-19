"""Configuration validation, broadcasting and reproducibility."""

import numpy as np
import pytest

from npendulum.config import SimulationConfig
from npendulum.simulate import run_simulation


def test_scalar_broadcasting():
    config = SimulationConfig(num_links=4, masses=0.5, lengths=2.0,
                              damping=0.1, initial_angles=1.0)
    assert config.masses == (0.5,) * 4
    assert config.lengths == (2.0,) * 4
    assert config.damping == (0.1,) * 4
    assert config.initial_angles == (1.0,) * 4


def test_sequence_lengths_validated():
    with pytest.raises(ValueError, match="masses"):
        SimulationConfig(num_links=3, masses=(1.0, 2.0))
    with pytest.raises(ValueError, match="initial_angles"):
        SimulationConfig(num_links=2, initial_angles=(0.1, 0.2, 0.3))


@pytest.mark.parametrize("kwargs", [
    dict(num_links=0),
    dict(num_links=2, masses=-1.0),
    dict(num_links=2, lengths=0.0),
    dict(num_links=2, damping=-0.5),
    dict(num_links=2, t_final=-1.0),
    dict(num_links=2, dt_output=0.0),
    dict(num_links=2, integrator="euler"),
    dict(num_links=2, rtol=0.0),
])
def test_invalid_configs_rejected(kwargs):
    with pytest.raises(ValueError):
        SimulationConfig(**kwargs)


def test_output_grid():
    config = SimulationConfig(num_links=1, t_final=1.0, dt_output=0.25)
    assert np.allclose(config.output_times, [0.0, 0.25, 0.5, 0.75, 1.0])


def test_json_round_trip(tmp_path):
    config = SimulationConfig(num_links=3, masses=(1, 2, 3),
                              initial_angles=(0.1, 0.2, 0.3), damping=0.2,
                              integrator="midpoint", dt_internal=0.001)
    path = tmp_path / "config.json"
    config.to_json(path)
    assert SimulationConfig.from_json(path) == config


def test_simulation_is_reproducible():
    config = SimulationConfig(num_links=3, initial_angles=(2.0, -1.0, 0.5),
                              t_final=5.0, dt_output=0.01)
    a = run_simulation(config)
    b = run_simulation(config)
    assert np.array_equal(a.theta, b.theta)
    assert np.array_equal(a.omega, b.omega)
    assert np.array_equal(a.energy, b.energy)


def test_works_for_larger_n():
    """Nothing in the pipeline may be hard-coded to a specific N."""
    config = SimulationConfig(num_links=10, initial_angles=1.0,
                              t_final=1.0, dt_output=0.01)
    result = run_simulation(config)
    assert result.theta.shape == (101, 10)
    assert np.max(np.abs(result.relative_energy_error)) < 1e-7
