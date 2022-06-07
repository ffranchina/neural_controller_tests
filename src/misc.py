import os
import torch
import numpy as np


def save_models(attacker_model, defender_model, path):
    if not os.path.isdir(path):
        os.mkdir(path)

    atk_path = os.path.join(path, "attacker.pt")
    def_path = os.path.join(path, "defender.pt")

    torch.save(attacker_model.state_dict(), atk_path)
    torch.save(defender_model.state_dict(), def_path)


def load_models(attacker_model, defender_model, path):
    atk_path = os.path.join(path, "attacker.pt")
    def_path = os.path.join(path, "defender.pt")

    attacker_model.load_state_dict(torch.load(atk_path))
    defender_model.load_state_dict(torch.load(def_path))


class ParametersHyperparallelepiped:
    """Class used to sample from the hyper-grid of parameters.
    It also adds some gaussian noise to the sampled point in
    order to encourage the exploration of the space.
    """

    def __init__(self, *ranges, seed=None):
        self._ranges = ranges
        self._rng = np.random.default_rng(seed)

    def sample(self, mu=0, sigma=1):
        while True:
            yield [
                self._rng.choice(r) + self._rng.normal(mu, sigma)
                if isinstance(r, np.ndarray)
                else float(r)
                for r in self._ranges
            ]


class Simulator:
    def __init__(self, physical_model_class, param_generator):
        self.model = physical_model_class()
        self._param_generator = param_generator

        self._previous_initial_state = None

    def step(self, env_input, agent_input, dt):
        """Updates the physical world with the evolution of
        a single instant of time.
        """
        self.model.step(env_input, agent_input, dt)

    def reset_to(self, parameters):
        """Sets the world's state as specified"""
        self._last_initial_state = parameters

        self.model.reinitialize(*self._last_initial_state)

    def reset_to_random(self):
        """Sample a random initial state"""
        parameters = next(self._param_generator)
        self.reset_to(parameters)

    def reset(self):
        """Restore the world's state to the last initialization"""
        self.reset_to(self._last_initial_state)
