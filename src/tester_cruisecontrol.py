import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

import architecture
import misc
import model_cruisecontrol

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
env_seed = np.arange(0, 1_000_000)
agent_position = 0
agent_velocity = np.linspace(-12, 12, 25)
initial_conditions_ranges = [env_seed, agent_position, agent_velocity]
# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(*initial_conditions_ranges)

# Instantiates the NN architectures
nn_agent = architecture.NeuralAgent(
    model_cruisecontrol.Agent.sensors, model_cruisecontrol.Agent.actuators, 2, 10
)

dt = 0.05  # timestep

# Build the whole setting for the experiment
env = model_cruisecontrol.Environment()
agent = model_cruisecontrol.Agent("car", nn_agent)
world_model = model_cruisecontrol.World(env, agent, dt=dt)

# Instantiates the world's model
simulator = misc.Simulator(world_model, pg.sample(sigma=0.05))

misc.load_models(args.dirname, car=nn_agent)

steps = 300


@torch.no_grad()
def run(mode=None):
    simulator.reset_to_random()

    experimental_data = {}
    experimental_data["t"] = np.zeros(steps)
    experimental_data["pos"] = np.zeros(steps)
    experimental_data["vel"] = np.zeros(steps)
    experimental_data["acc"] = np.zeros(steps)

    space = torch.linspace(0, simulator.model.environment._road.length, 1000)
    experimental_data["road"] = simulator.model.environment._road.get_fn(space)

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
            experimental_data["pos"][i] = agent.position.numpy()
            experimental_data["vel"][i] = agent.velocity.numpy()
            experimental_data["acc"][i] = actions[agent.label].numpy()

        simulator.step(actions)

        t += dt

    return {k: v.tolist() for k, v in experimental_data.items()}


records = []
for i in range(args.repetitions):
    sim = {}
    sim["up"] = run()

    records.append(sim)

with open(os.path.join(args.dirname, "sims.json"), "w") as f:
    json.dump(records, f)
