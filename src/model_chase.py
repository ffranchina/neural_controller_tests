import torch

import abstract_model
import misc


class Drone:
    """Describes the physical behaviour of the drone"""

    def __init__(self, max_acceleration, max_velocity):
        self._max_acceleration = max_acceleration
        self._min_acceleration = -self._max_acceleration
        self._max_velocity = max_velocity
        self._min_velocity = -self._max_velocity
        self.position = torch.tensor((0.0, 0.0))
        self.velocity = torch.tensor((0.0, 0.0))
        self.acceleration = torch.tensor((0.0, 0.0))

    def update(self, in_acceleration, dt):
        """Differential equation for updating the state of the drone"""
        with torch.no_grad():
            norm = torch.linalg.norm(in_acceleration)

        if norm < self._max_acceleration and norm > self._min_acceleration:
            self.acceleration = in_acceleration
        else:
            self.acceleration = in_acceleration * self._max_acceleration / norm

        self.velocity = torch.clamp(
            self.velocity + self.acceleration * dt,
            self._min_velocity,
            self._max_velocity,
        )
        self.position += self.velocity * dt


class Environment(abstract_model.Environment):
    @property
    def leader(self):
        leaders = [ag for ag in self._agents.values() if isinstance(ag, Leader)]
        return leaders[0]

    @property
    def follower(self):
        followers = [ag for ag in self._agents.values() if not isinstance(ag, Leader)]
        return followers[0]


class Agent(abstract_model.Agent):
    sensors = 3 * 2
    actuators = 1 * 2

    def __init__(self, label, nn, target_formula=None, **constants):
        super().__init__(label, nn, target_formula, **constants)

        self._drone = Drone(constants["max_acceleration"], constants["max_velocity"])

    @property
    def position(self):
        return self._drone.position.clone()

    @position.setter
    def position(self, value):
        self._drone.position = value

    @property
    def velocity(self):
        return self._drone.velocity.clone()

    @velocity.setter
    def velocity(self, value):
        self._drone.velocity = value

    @property
    def distance(self):
        return self._environment.leader.position - self.position

    @property
    def status(self):
        """Representation of the state"""
        return (
            torch.flatten(
                torch.cat(
                    (
                        self._environment.leader.velocity,
                        self.velocity,
                        self.distance,
                    )
                )
            )
            .clone()
            .detach()
            .numpy()
        )

    def update(self, parameters, dt):
        """Updates the physical state with the parameters
        generated by the NN.
        """
        acceleration = parameters
        self._drone.update(acceleration, dt)


class Leader(Agent):
    @property
    def distance(self):
        return self._environment.leader.position - self._environment.follower.position

    @property
    def status(self):
        """Representation of the state"""
        return (
            torch.flatten(
                torch.cat(
                    (
                        self._environment.leader.velocity,
                        self._environment.follower.velocity,
                        self.distance,
                    )
                )
            )
            .clone()
            .detach()
            .numpy()
        )


class World(abstract_model.World):
    """The model of the whole world.
    It includes both the attacker and the defender.
    """

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
            agent.position = torch.tensor(values[f"{agent.label}_position"])
            agent.velocity = torch.tensor(values[f"{agent.label}_velocity"])

    @property
    def observables(self):
        return {"dist": torch.linalg.norm(self.environment.follower.distance)}
