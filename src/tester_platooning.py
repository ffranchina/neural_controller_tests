import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

import architecture
import misc
import model_platooning

parser = ArgumentParser()
parser.add_argument("dirname", help="model's directory")
parser.add_argument(
    "-r",
    "--repetitions",
    dest="repetitions",
    type=int,
    default=1,
    help="simulation repetions",
)
args = parser.parse_args()

# Specifies the initial conditions of the setup
follower_position = 0
follower_velocity = np.linspace(0, 20, 40)
leader_position = np.linspace(1, 12, 15)
leader_velocity = np.linspace(0, 20, 40)
initial_conditions_ranges = [
    leader_position,
    leader_velocity,
    follower_position,
    follower_velocity,
]
# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(*initial_conditions_ranges)

# Instantiates the NN architectures
nn_leader = architecture.NeuralAgent(
    model_platooning.Agent.sensors, model_platooning.Agent.actuators, 2, 10, 2
)
nn_follower = architecture.NeuralAgent(
    model_platooning.Agent.sensors, model_platooning.Agent.actuators, 2, 10
)

dt = 0.05

# Build the whole setting for the experiment
env = model_platooning.Environment()
leader = model_platooning.Agent("leader", nn_leader)
follower = model_platooning.Agent("follower", nn_follower)
world_model = model_platooning.World(env, leader, follower, dt=dt)

# Instantiates the world's model
simulator = misc.Simulator(world_model, pg.sample(sigma=0.05))

misc.load_models(args.dirname, leader=nn_leader, follower=nn_follower)

steps = 300


@torch.no_grad()
def run(mode=None):
    simulator.reset_to_random()

    experimental_data = {}
    experimental_data["t"] = np.zeros(steps)
    for agent in simulator.model.agents.values():
        experimental_data[f"{agent.label}_pos"] = np.zeros(steps)
        experimental_data[f"{agent.label}_vel"] = np.zeros(steps)
        experimental_data[f"{agent.label}_acc"] = np.zeros(steps)

    t = 0
    for i in range(steps):
        policies, actions = {}, {}

        for agent in simulator.model.agents.values():
            noise = torch.rand(agent.nn.input_noise_size)
            sensors = torch.tensor(agent.status)

            policies[agent.label] = agent.nn(torch.cat((noise, sensors)))

            if mode == 0:
                policies["leader"] = (
                    lambda x: torch.tensor(2.0)
                    if i > 200 and i < 250
                    else torch.tensor(-2.0)
                )
            elif mode == 1:
                policies["leader"] = (
                    lambda x: torch.tensor(2.0) if i > 150 else torch.tensor(-2.0)
                )
            elif mode == 2:
                policies["leader"] = (
                    lambda x: torch.tensor(2.0) if i < 150 else torch.tensor(-2.0)
                )

            actions[agent.label] = policies[agent.label](simulator.model.dt)

        experimental_data["t"][i] = t
        for agent in simulator.model.agents.values():
            experimental_data[f"{agent.label}_pos"][i] = agent.position.numpy()
            experimental_data[f"{agent.label}_vel"][i] = agent.velocity.numpy()
            experimental_data[f"{agent.label}_acc"][i] = actions[agent.label].numpy()

        simulator.step(actions)

        t += dt

    return {k: v.tolist() for k, v in experimental_data.items()}


records = []
for i in range(args.repetitions):
    sim = {}
    sim["pulse"] = run(0)
    sim["step_up"] = run(1)
    sim["step_down"] = run(2)
    sim["atk"] = run()

    records.append(sim)

with open(os.path.join(args.dirname, "sims.json"), "w") as f:
    json.dump(records, f)
