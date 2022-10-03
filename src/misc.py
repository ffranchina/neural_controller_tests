import os

import numpy as np
import torch

from diffquantitative import DiffQuantitativeSemantic


def save_models(path, **neuralagent_dict):
    if not os.path.isdir(path):
        os.mkdir(path)

    for name, nn in neuralagent_dict.items():
        file_path = os.path.join(path, f"{name}.pt")
        torch.save(nn.state_dict(), file_path)


def load_models(path, **neuralagent_dict):
    for name, nn in neuralagent_dict.items():
        file_path = os.path.join(path, f"{name}.pt")
        nn.load_state_dict(torch.load(file_path))


class ParametersHyperspace:
    """Class used to sample from the hyper-grid of parameters.
    It also adds some gaussian noise to the sampled point in
    order to encourage the exploration of the space.
    """

    def __init__(self, seed=None, sigma=0.5, **ranges):
        # Force order invariance by sorting the dictionary
        self._ranges = {k: ranges[k] for k in sorted(ranges)}
        self._rng = np.random.default_rng(seed)
        self._sigma = sigma

    def __next__(self):
        return {
            k: self._rng.choice(v) + self._rng.normal(0, self._sigma)
            if isinstance(v, np.ndarray)
            else np.array(v, dtype=np.float64)
            for k, v in self._ranges.items()
        }


class Simulator:
    def __init__(self, world_model, param_generator):
        self.world = world_model
        self._param_generator = param_generator

        self._previous_initial_state = None
        self.recordings = {}
        self.recorded_steps = 0

    def step(self, agent_actions):
        """Updates the physical world with the evolution of
        a single instant of time.
        """
        if self._previous_initial_state is not None:
            self.world.step(agent_actions)
            self.record_step()

    def record_step(self):
        for key, value in self.world.observables.items():
            value = value.reshape(-1)  # To make sure it has a dimension
            if key in self.recordings:
                self.recordings[key].append(value)
            else:
                self.recordings[key] = [value]

        self.recorded_steps += 1

    def reset_to(self, parameters):
        """Sets the world's state as specified"""
        self._previous_initial_state = parameters

        self.world.state = parameters
        self.recordings = {}
        self.recorded_steps = 0
        self.record_step()

    def reset_to_random(self):
        """Sample a random initial state"""
        parameters = next(self._param_generator)
        self.reset_to(parameters)

    def reset(self):
        """Restore the world's state to the last initialization"""
        self.reset_to(self._previous_initial_state)


class RobustnessComputer:
    """Used to compute the robustness value (rho)"""

    def __init__(self, formula):
        self.dqs = DiffQuantitativeSemantic(formula)

    def positive_percentage(self, simulator):
        positive_counter = 0
        for i in range(simulator.recorded_steps):
            recorded_slice = {k: v[i] for k, v in simulator.recordings.items()}
            if self.dqs.compute(**recorded_slice) >= 0:
                positive_counter += 1

        return positive_counter / simulator.recorded_steps

    def compute(self, simulator):
        """Computes rho for the given trace"""
        recordings = {k: torch.cat(v) for k, v in simulator.recordings.items()}

        return self.dqs.compute(**recordings)
