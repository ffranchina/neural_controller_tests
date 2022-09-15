import torch

import abstract_model


class Car:
    """Describes the physical behaviour of the vehicle"""

    def __init__(self):
        self._max_acceleration = 3.0
        self._min_acceleration = -self._max_acceleration
        self._max_velocity = 20.0
        self._min_velocity = 0.0
        self.gravity = 9.81
        self.position = torch.tensor(0.0)
        self.velocity = torch.tensor(0.0)
        self.acceleration = torch.tensor(0.0)
        self.friction_coefficient = 0.01

    def update(self, in_acceleration, dt):
        """Differential equation for updating the state of the car"""
        self.acceleration = torch.clamp(
            in_acceleration, self._min_acceleration, self._max_acceleration
        )
        if self.velocity > 0:
            self.acceleration -= self.friction_coefficient * self.gravity
        self.velocity = torch.clamp(
            self.velocity + self.acceleration * dt,
            self._min_velocity,
            self._max_velocity,
        )
        self.position += self.velocity * dt


class Environment(abstract_model.Environment):
    @property
    def leader(self):
        return self._agents["leader"]

    @property
    def follower(self):
        return self._agents["follower"]

    @property
    def distance(self):
        return self.leader.position - self.follower.position


class Agent(abstract_model.Agent):
    sensors = 3
    actuators = 1

    def __init__(self, label, nn, target_formula=None):
        super().__init__(label, nn, target_formula)

        self._car = Car()

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
    def status(self):
        """Representation of the state"""
        return (
            self._environment.leader.velocity,
            self._environment.follower.velocity,
            self._environment.distance,
        )

    def update(self, parameters, dt):
        """Updates the physical state with the parameters
        generated by the NN.
        """
        acceleration = parameters
        self._car.update(acceleration, dt)


class Model(abstract_model.Model):
    @property
    def state(self):
        values = {}

        for agent in self.agents.values():
            values[f"{agent.label}_position"] = agent.position
            values[f"{agent.label}_velocity"] = agent.velocity

        return values

    @state.setter
    def state(self, values):
        """Sets the world's state as specified"""
        for agent in self.agents.values():
            agent.position = torch.tensor(values[f"{agent.label}_position"]).reshape(1)
            agent.velocity = torch.tensor(values[f"{agent.label}_velocity"]).reshape(1)

    @property
    def observables(self):
        return {"dist": self.environment.distance}
