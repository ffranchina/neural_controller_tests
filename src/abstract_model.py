from abc import ABC, abstractmethod

import misc


class Environment:
    def __init__(self, **constants):
        self._agents = None
        self._constants = constants

    def set_agents(self, agents):
        self._agents = agents


class Agent(ABC):
    sensors = 0
    actuators = 0

    def __init__(self, label, nn, target_formula=None, **constants):
        assert self.sensors > 0
        assert self.actuators > 0

        self._label = label
        self._nn = nn

        self._environment = None
        self._constants = constants

        self._robustness_computer = (
            misc.RobustnessComputer(target_formula) if target_formula else None
        )

    def set_environment(self, environment):
        self._environment = environment

    @property
    def label(self):
        return self._label

    @property
    def nn(self):
        return self._nn

    @property
    def robustness_computer(self):
        return self._robustness_computer

    @property
    @abstractmethod
    def status(self):
        """Representation of the state"""
        pass

    @abstractmethod
    def update(self, parameters, dt):
        """Updates the physical state with the parameters
        generated by the NN.
        """
        pass


class World(ABC):
    def __init__(self, environment, *agents, dt=0.05):
        assert environment is not None
        assert agents is not None
        assert len(agents) > 0

        self._dt = dt

        self.environment = environment
        self.agents = {agent.label: agent for agent in agents}

        # Makes sure labels are different
        assert len(agents) == len(self.agents)

        self.environment.set_agents(self.agents)
        for agent in self.agents.values():
            agent.set_environment(self.environment)

    def step(self, agent_actions):
        """Updates the physical world with the evolution of
        a single instant of time.
        """
        for label, agent in self.agents.items():
            agent.update(agent_actions[label], self.dt)

    @property
    def dt(self):
        return self._dt

    @property
    @abstractmethod
    def state(self):
        pass

    @state.setter
    @abstractmethod
    def state(self, values):
        """Sets the world's state as specified"""
        pass

    @property
    @abstractmethod
    def observables(self):
        pass
