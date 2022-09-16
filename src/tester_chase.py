import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

import architecture
import misc
import model_chase

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
follower_position = [0.0, 0.0]
follower_velocity = np.array(
    np.meshgrid(np.linspace(0, 5, 10), np.linspace(0, 5, 10))
).T.reshape(-1, 2)
leader_position = np.array(
    np.meshgrid(np.linspace(1, 11, 15), np.linspace(1, 11, 15))
).T.reshape(-1, 2)
leader_velocity = np.array(
    np.meshgrid(np.linspace(0, 5, 10), np.linspace(0, 5, 10))
).T.reshape(-1, 2)
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
    model_chase.Leader.sensors, model_chase.Leader.actuators, 2, 10, 2
)
nn_follower = architecture.NeuralAgent(
    model_chase.Agent.sensors, model_chase.Agent.actuators, 2, 10
)

dt = 0.05  # timestep

# Build the whole setting for the experiment
env = model_chase.Environment()
leader = model_chase.Leader("leader", nn_leader)
follower = model_chase.Agent("follower", nn_follower)
world_model = model_chase.World(env, leader, follower, dt=dt)

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
        experimental_data[f"{agent.label}_pos"] = np.zeros((steps, 2))
        experimental_data[f"{agent.label}_vel"] = np.zeros((steps, 2))
        experimental_data[f"{agent.label}_acc"] = np.zeros((steps, 2))

    t = 0
    for i in range(steps):
        policies, actions = {}, {}

        for agent in simulator.model.agents.values():
            noise = torch.rand(agent.nn.input_noise_size)
            sensors = torch.tensor(agent.status)

            policies[agent.label] = agent.nn(torch.cat((noise, sensors)))
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
    sim["atk"] = run()

    records.append(sim)

with open(os.path.join(args.dirname, "sims.json"), "w") as f:
    json.dump(records, f)
