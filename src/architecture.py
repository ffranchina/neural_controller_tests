import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.set_default_tensor_type(torch.DoubleTensor)


class NeuralAgent(nn.Module):
    """NN architecture for a generic agent"""

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_hidden_layers,
        layer_size,
        input_noise_size=0,
        n_coeff=3,
    ):
        super().__init__()

        assert n_hidden_layers > 0

        self.optimizer = None

        self.input_noise_size = input_noise_size
        self.n_coeff = n_coeff

        input_layer_size = n_inputs + input_noise_size
        output_layer_size = n_outputs * n_coeff

        layers = []
        layers.append(nn.Linear(input_layer_size, layer_size))
        layers.append(nn.LeakyReLU())

        for i in range(n_hidden_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.LeakyReLU())

        layers.append(nn.Linear(layer_size, output_layer_size))

        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        """Uses the NN's output to compute the coefficients of the policy function"""
        coefficients = self.nn(x)
        coefficients = torch.reshape(coefficients, (-1, self.n_coeff))

        def policy_generator(t):
            """The policy function is defined as polynomial"""
            basis = [t**i for i in range(self.n_coeff)]
            basis = torch.tensor(basis, dtype=torch.get_default_dtype())
            basis = torch.reshape(basis, (self.n_coeff, -1))
            return coefficients.mm(basis).squeeze()

        return policy_generator


class Trainer:
    """The class contains the training logic"""

    def __init__(
        self,
        simulator,
        logging_dir=None,
    ):

        self.simulator = simulator
        self.agents = self.simulator.model.agents

        self.loss_function = lambda x: -x

        for agent in self.agents.values():
            agent.nn.optimizer = optim.Adam(agent.nn.parameters())

        self.logging = True if logging_dir else False

        if self.logging:
            self.log = SummaryWriter(logging_dir)

    def train_agent(self, training_agent, time_horizon):
        policies, actions = {}, {}

        for agent in self.agents.values():
            noise = torch.rand(agent.nn.input_noise_size)
            sensors = torch.tensor(agent.status)

            if agent.label == training_agent.label:
                policies[agent.label] = agent.nn(torch.cat((noise, sensors)))
            else:
                with torch.no_grad():
                    policies[agent.label] = agent.nn(torch.cat((noise, sensors)))

        t = 0
        for i in range(time_horizon):
            actions = {label: policies[label](t) for label in self.agents}

            self.simulator.step(actions)

            t += self.simulator.model.dt

        rho = training_agent.robustness_computer.compute(self.simulator)

        training_agent.nn.optimizer.zero_grad()

        loss = self.loss_function(rho)
        loss.backward()

        training_agent.nn.optimizer.step()

        return float(loss.detach())

    def train(self, agent_replays, time_horizon):
        """Trains all the agents in the same initial senario (different for each)"""
        losses = {}

        for agent in self.agents.values():
            self.simulator.reset_to_random()  # samples a random initial state

            n_replays = agent_replays[agent.label]
            agent_losses = torch.zeros(n_replays)

            for i in range(n_replays):
                agent_losses[i] = self.train_agent(agent, time_horizon)
                self.simulator.reset()  # restores the initial state

            losses[agent.label] = torch.mean(agent_losses)

        return losses

    def run(self, n_steps, agent_replays, time_horizon=100, *, epoch=0):
        """Trains the architecture and provides logging and visual feedback"""
        for i in tqdm(range(n_steps)):
            losses = self.train(agent_replays, time_horizon)

            if self.logging:

                for agent in self.agents.values():
                    self.log.add_scalar(
                        f"{agent.label}_train/loss",
                        losses[agent.label],
                        n_steps * epoch + i,
                    )

                    params = torch.cat(
                        [torch.flatten(param.data) for param in agent.nn.parameters()]
                    )

                    params_var, params_mean = torch.var_mean(params)
                    params_norm = torch.linalg.norm(params)

                    self.log.add_scalar(
                        f"{agent.label}_train/weights variance",
                        params_var,
                        n_steps * epoch + i,
                    )

                    self.log.add_scalar(
                        f"{agent.label}_train/weights mean",
                        params_mean,
                        n_steps * epoch + i,
                    )

                    self.log.add_scalar(
                        f"{agent.label}_train/weights norm",
                        params_norm,
                        n_steps * epoch + i,
                    )


class Tester:
    """The class contains the testing logic"""

    def __init__(
        self,
        simulator,
        logging_dir=None,
    ):

        self.simulator = simulator
        self.agents = self.simulator.model.agents

        self.logging = True if logging_dir else False

        if self.logging:
            self.log = SummaryWriter(logging_dir)

    def test(self, time_horizon):
        """Tests a whole episode"""
        policies, actions = {}, {}

        self.simulator.reset_to_random()

        for t in range(time_horizon):
            for agent in self.agents.values():
                noise = torch.rand(agent.nn.input_noise_size)
                sensors = torch.tensor(agent.status)

                with torch.no_grad():
                    policies[agent.label] = agent.nn(torch.cat((noise, sensors)))

                actions[agent.label] = policies[agent.label](self.simulator.model.dt)

            self.simulator.step(actions)

        positive_percentage = {
            agent.label: agent.robustness_computer.positive_percentage(self.simulator)
            for agent in self.agents.values()
        }

        return positive_percentage

    def run(self, times, time_horizon=1000, epoch=0):
        """Test the architecture and provides logging"""
        rho_list = torch.zeros(times)
        percentages = []
        for i in tqdm(range(times)):
            percentages.append(self.test(time_horizon))

        if self.logging:
            for agent in self.agents.values():
                values = torch.tensor([p[agent.label] for p in percentages])

                self.log.add_scalar(
                    f"{agent.label}_test/mean_robustness", torch.mean(values), epoch
                )

                self.log.add_histogram(f"{agent.label}_test/robustness", values, epoch)
