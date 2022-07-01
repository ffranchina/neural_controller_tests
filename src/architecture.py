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

        self.attacker = self.simulator.model.environment.nn
        self.defender = self.simulator.model.agent.nn

        self.loss_function = lambda x: -x

        atk_optimizer = optim.Adam(self.attacker.parameters())
        def_optimizer = optim.Adam(self.defender.parameters())
        self.attacker_optimizer = atk_optimizer
        self.defender_optimizer = def_optimizer

        self.logging = True if logging_dir else False

        if self.logging:
            self.log = SummaryWriter(logging_dir)

    def train_attacker_step(self, time_horizon, dt, atk_static):
        """Training step for the attacker. The defender's passive."""
        z = torch.rand(self.attacker.input_noise_size)
        oa = torch.tensor(self.simulator.model.agent.status)
        oe = torch.tensor(self.simulator.model.environment.status)

        atk_policy = self.attacker(torch.cat((z, oe)))

        with torch.no_grad():
            def_policy = self.defender(oa)

        t = 0
        for i in range(time_horizon):
            # if the attacker is static (e.g. in the case it does not vary over time)
            # the policy function is always sampled in the same point since the
            # attacker do not vary policy over time
            atk_input = atk_policy(0 if atk_static else t)
            def_input = def_policy(t)

            self.simulator.step(atk_input, def_input, dt)

            t += dt

        rho = self.simulator.model.environment.robustness_computer.compute(
            self.simulator
        )

        self.attacker_optimizer.zero_grad()

        loss = self.loss_function(rho)
        loss.backward()

        self.attacker_optimizer.step()

        return float(loss.detach())

    def train_defender_step(self, time_horizon, dt, atk_static):
        """Training step for the defender. The attacker's passive."""
        z = torch.rand(self.attacker.input_noise_size)
        oa = torch.tensor(self.simulator.model.agent.status)
        oe = torch.tensor(self.simulator.model.environment.status)

        with torch.no_grad():
            atk_policy = self.attacker(torch.cat((z, oe)))

        def_policy = self.defender(oa)

        t = 0
        for i in range(time_horizon):
            # if the attacker is static, see the comments above
            atk_input = atk_policy(0 if atk_static else t)
            def_input = def_policy(t)

            self.simulator.step(atk_input, def_input, dt)

            t += dt

        rho = self.simulator.model.agent.robustness_computer.compute(self.simulator)

        self.defender_optimizer.zero_grad()

        loss = self.loss_function(rho)
        loss.backward()

        self.defender_optimizer.step()

        return float(loss.detach())

    def train(self, atk_steps, def_steps, time_horizon, dt, atk_static):
        """Trains both the attacker and the defender on the same
        initial senario (different for each)
        """
        atk_loss, def_loss = 0, 0

        self.simulator.reset_to_random()  # samples a random initial state
        for i in range(atk_steps):
            atk_loss = self.train_attacker_step(time_horizon, dt, atk_static)
            self.simulator.reset()  # restores the initial state

        self.simulator.reset_to_random()  # samples a random initial state
        for i in range(def_steps):
            def_loss = self.train_defender_step(time_horizon, dt, atk_static)
            self.simulator.reset()  # restores the initial state

        return (atk_loss, def_loss)

    def run(
        self,
        n_steps,
        time_horizon=100,
        dt=0.05,
        *,
        atk_steps=1,
        def_steps=1,
        atk_static=False,
        epoch=0
    ):
        """Trains the architecture and provides logging and visual feedback"""
        for i in tqdm(range(n_steps)):
            atk_loss, def_loss = self.train(
                atk_steps, def_steps, time_horizon, dt, atk_static
            )

            if self.logging:
                atk_params = torch.cat(
                    [torch.flatten(param.data) for param in self.attacker.parameters()]
                )
                atk_params_var, atk_params_mean = torch.var_mean(atk_params)
                atk_norm = torch.linalg.norm(atk_params)

                def_params = torch.cat(
                    [torch.flatten(param.data) for param in self.defender.parameters()]
                )
                def_params_var, def_params_mean = torch.var_mean(def_params)
                def_norm = torch.linalg.norm(def_params)

                self.log.add_scalar(
                    "attacker_train/loss", atk_loss, n_steps * epoch + i
                )
                self.log.add_scalar(
                    "defender_train/loss", def_loss, n_steps * epoch + i
                )

                self.log.add_scalar(
                    "attacker_train/weights variance",
                    atk_params_var,
                    n_steps * epoch + i,
                )
                self.log.add_scalar(
                    "defender_train/weights variance",
                    def_params_var,
                    n_steps * epoch + i,
                )

                self.log.add_scalar(
                    "attacker_train/weights mean", atk_params_mean, n_steps * epoch + i
                )
                self.log.add_scalar(
                    "defender_train/weights mean", def_params_mean, n_steps * epoch + i
                )

                self.log.add_scalar(
                    "attacker_train/weights norm", atk_norm, n_steps * epoch + i
                )
                self.log.add_scalar(
                    "defender_train/weights norm", def_norm, n_steps * epoch + i
                )

        if self.logging:
            self.log.close()


class Tester:
    """The class contains the testing logic"""

    def __init__(
        self,
        simulator,
        logging_dir=None,
    ):

        self.simulator = simulator

        self.attacker = self.simulator.model.environment.nn
        self.defender = self.simulator.model.agent.nn

        self.logging = True if logging_dir else False

        if self.logging:
            self.log = SummaryWriter(logging_dir)

    def test(self, time_horizon, dt):
        """Tests a whole episode"""
        self.simulator.reset_to_random()

        for t in range(time_horizon):
            z = torch.rand(self.attacker.input_noise_size)
            oa = torch.tensor(self.simulator.model.agent.status)
            oe = torch.tensor(self.simulator.model.environment.status)

            with torch.no_grad():
                atk_policy = self.attacker(torch.cat((z, oe)))
                def_policy = self.defender(oa)

            atk_input = atk_policy(dt)
            def_input = def_policy(dt)

            self.simulator.step(atk_input, def_input, dt)

        rho = self.simulator.model.agent.robustness_computer.compute(self.simulator)

        return rho

    def run(self, times, time_horizon=1000, dt=0.05, epoch=0):
        """Test the architecture and provides logging"""
        rho_list = torch.zeros(times)
        for i in tqdm(range(times)):
            def_rho = self.test(time_horizon, dt)
            rho_list[i] = def_rho

            if self.logging:
                self.log.add_scalar(
                    "defender_test/robustness", def_rho, times * epoch + i
                )

        if self.logging:
            self.log.add_scalar(
                "defender_test/avg robustness", torch.mean(rho_list), epoch
            )
            self.log.close()
