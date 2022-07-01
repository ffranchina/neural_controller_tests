import os
import json

import model_chase
import misc
import architecture

import torch
import numpy as np

from argparse import ArgumentParser

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

agent_position = [0.0, 0.0]
agent_velocity = np.array(
    np.meshgrid(np.linspace(0, 5, 10), np.linspace(0, 5, 10))
).T.reshape(-1, 2)
leader_position = np.array(
    np.meshgrid(np.linspace(1, 11, 15), np.linspace(1, 11, 15))
).T.reshape(-1, 2)
leader_velocity = np.array(
    np.meshgrid(np.linspace(0, 5, 10), np.linspace(0, 5, 10))
).T.reshape(-1, 2)
pg = misc.ParametersHyperparallelepiped(
    agent_position, agent_velocity, leader_position, leader_velocity
)

nn_attacker = architecture.NeuralAgent(
    model_chase.Environment.sensors, model_chase.Environment.actuators, 2, 10, 2
)
nn_defender = architecture.NeuralAgent(
    model_chase.Agent.sensors, model_chase.Agent.actuators, 2, 10
)

attacker = model_chase.Environment("attacker", nn_attacker)
defender = model_chase.Agent("defender", nn_defender)

# Passa al Trainer i TrainingAgent
world_model = model_chase.Model(attacker, defender)

# Instantiates the world's model
simulator = misc.Simulator(world_model, pg.sample(sigma=0.05))

misc.load_models(nn_attacker, nn_defender, args.dirname)

dt = 0.05
steps = 300


def run(mode=None):
    simulator.reset_to_random()
    conf_init = {
        "ag_pos": simulator.model.agent.position.numpy().tolist(),
        "ag_vel": simulator.model.agent.velocity.numpy().tolist(),
        "env_pos": simulator.model.environment.l_position.numpy().tolist(),
        "env_vel": simulator.model.environment.l_velocity.numpy().tolist(),
    }

    sim_t = []
    sim_ag_pos = []
    sim_ag_dist = []
    sim_ag_acc = []
    sim_env_pos = []
    sim_env_acc = []

    t = 0
    for i in range(steps):
        with torch.no_grad():
            oa = torch.tensor(simulator.model.agent.status)
            oe = torch.tensor(simulator.model.environment.status)
            z = torch.rand(nn_attacker.input_noise_size)

            atk_policy = nn_attacker(torch.cat((z, oe)))
            def_policy = nn_defender(oa)

        atk_input = atk_policy(dt)
        def_input = def_policy(dt)

        simulator.step(atk_input, def_input, dt)

        sim_ag_acc.append(def_input.numpy())
        sim_env_acc.append(atk_input.numpy())
        sim_t.append(t)
        sim_ag_pos.append(simulator.model.agent.position.numpy())
        sim_env_pos.append(simulator.model.environment.l_position.numpy())
        sim_ag_dist.append(simulator.model.agent.distance.numpy())

        t += dt

    return {
        "init": conf_init,
        "sim_t": np.array(sim_t).tolist(),
        "sim_ag_pos": np.array(sim_ag_pos).tolist(),
        "sim_ag_dist": np.array(sim_ag_dist).tolist(),
        "sim_ag_acc": np.array(sim_ag_acc).tolist(),
        "sim_env_pos": np.array(sim_env_pos).tolist(),
        "sim_env_acc": np.array(sim_env_acc).tolist(),
    }


records = []
for i in range(args.repetitions):
    sim = {}
    sim["atk"] = run()

    records.append(sim)

with open(os.path.join(args.dirname, "sims.json"), "w") as f:
    json.dump(records, f)
