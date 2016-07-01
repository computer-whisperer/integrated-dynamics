import pytest


@pytest.fixture(scope="function")
def world():
    import int_dynamics.dynamics
    return int_dynamics.dynamics.WorldBody()
