"""Observation functions for traffic signals."""
from abc import abstractmethod

import numpy as np
from gymnasium import spaces

from .traffic_signal import TrafficSignal


class ObservationFunction:
    """Abstract base class for observation functions."""

    def __init__(self, ts: TrafficSignal):
        """Initialize observation function."""
        self.ts = ts

    @abstractmethod
    def __call__(self):
        """Subclasses must override this method."""
        pass

    @abstractmethod
    def observation_space(self):
        """Subclasses must override this method."""
        pass


class DefaultObservationFunction(ObservationFunction):
    """Default observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize default observation function."""
        super().__init__(ts)

    def __call__(self) -> dict:
        """Return the default observation."""
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        observation = {
            "density": np.array(density, dtype=np.float32),
            "queue": np.array(queue, dtype=np.float32),
        }
        return observation
    def observation_space(self) -> spaces.Dict:
        """Return the observation space."""
        num_lanes = len(self.ts.lanes)

        return spaces.Dict({
            "density": spaces.Box(
                low=np.zeros(num_lanes, dtype=np.float32),
                high=np.ones(num_lanes, dtype=np.float32),
            ),
            "queue": spaces.Box(
                low=np.zeros(num_lanes, dtype=np.float32),
                high=np.ones(num_lanes, dtype=np.float32),
            ),
        })
