import numpy as np
import torch

import abstract_model


class Car:
    """Describes the physical behaviour of the vehicle"""

    def __init__(self, max_acceleration, max_velocity, friction_coefficient):
        self._max_acceleration = max_acceleration
        self._min_acceleration = -self._max_acceleration
        self._max_velocity = max_velocity
        self._min_velocity = -self._max_velocity
        self.gravity = 9.81
        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)
        self.friction_coefficient = friction_coefficient

    def update(self, in_acceleration, angle, dt):
        """Differential equation for updating the state of the car"""
        self.acceleration = torch.clamp(
            in_acceleration, self._min_acceleration, self._max_acceleration
        )
        self.acceleration -= self.gravity * torch.sin(angle)
        if self.velocity != 0:
            self.acceleration -= (
                self.friction_coefficient * self.gravity * torch.cos(angle)
            )
        self.velocity = torch.clamp(
            self.velocity + self.acceleration * dt,
            self._min_velocity,
            self._max_velocity,
        )
        self.position += self.velocity * dt


class Road:
    def __init__(self, road_length, road_bumps, max_angle):
        self.length = road_length
        self.bumps = road_bumps

        self._max_angle = max_angle  # deg
        self._max_angular_coeff = np.tan(np.deg2rad(self._max_angle))
        self._dx = 0.1

        self._fn = self.gaussian_rbf

        self._seed = None
        self._means = None
        self._weights = None
        self._shapes = None
        self.seed = 0  # uses the setter

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self._rng = np.random.default_rng(self._seed)

        self._means = self._rng.random(self.bumps) * self.length
        self._weights = self._rng.random(self.bumps) * 2  # more variance
        self._shapes = self._rng.random(self.bumps) * 0.5  # flatter shapes

    def gaussian_rbf(self, x):
        x = x.reshape(1)
        phi = lambda x: np.exp(-((x * self._shapes) ** 2))
        differences = np.abs(x[:, np.newaxis] - self._means)

        return self._weights.dot(phi(differences).T)

    def get_steepness(self, x):
        """Computes the value of the road's steepness in a given point"""
        x = x.clone().detach().numpy()
        dy = self._fn(x + self._dx) - self._fn(x)
        deriv = dy / torch.tensor(self._dx)
        clamped = torch.clamp(deriv, -self._max_angular_coeff, self._max_angular_coeff)
        return torch.tensor(clamped.item())

    def get_fn(self, x):
        """Computes the value of the road's height in a given point"""
        derivative = np.array([self.get_steepness(i).numpy() for i in x])
        return np.cumsum(derivative) * self._dx


class Environment(abstract_model.Environment):
    def __init__(self, **constants):
        super().__init__(**constants)

        self._road = Road(
            constants["road_length"], constants["road_bumps"], constants["max_angle"]
        )

    @property
    def seed(self):
        return self._road._seed

    @seed.setter
    def seed(self, value):
        self._road._seed = value

    def get_angle(self, position):
        return self._road.get_steepness(position)


class Agent(abstract_model.Agent):
    sensors = 2
    actuators = 1

    def __init__(self, label, nn, target_formula=None, **constants):
        super().__init__(label, nn, target_formula, **constants)

        self._car = Car(
            constants["max_acceleration"],
            constants["max_velocity"],
            constants["friction_coefficient"],
        )

    @property
    def position(self):
        return self._car.position.clone()

    @position.setter
    def position(self, value):
        self._car.position = value

    @property
    def velocity(self):
        return self._car.velocity.clone()

    @velocity.setter
    def velocity(self, value):
        self._car.velocity = value

    @property
    def angle(self):
        return self._environment.get_angle(self.position)

    @property
    def status(self):
        """Representation of the state"""
        return (self.velocity, self.angle)

    def update(self, parameters, dt):
        """Updates the physical state with the parameters generated by the NN."""
        acceleration = parameters
        self._car.update(acceleration, self.angle, dt)


class World(abstract_model.World):
    """The model of the whole world.
    It includes both the attacker and the defender.
    """

    @property
    def state(self):
        values = {}

        values["environment_seed"] = self.environment.seed

        for agent in self.agents.values():
            values[f"{agent.label}_position"] = self.agents[agent.label].position
            values[f"{agent.label}_velocity"] = self.agents[agent.label].velocity

        return values

    @state.setter
    def state(self, values):
        """Sets the world's state as specified"""
        self.environment.seed = values["environment_seed"]

        for agent in self.agents.values():
            self.agents[agent.label].position = torch.tensor(
                values[f"{agent.label}_position"]
            )
            self.agents[agent.label].velocity = torch.tensor(
                values[f"{agent.label}_velocity"]
            )

    @property
    def observables(self):
        return {"vel": self.agents["car"].velocity}
