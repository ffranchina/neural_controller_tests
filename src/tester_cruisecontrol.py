import os
import json

import model_cruisecontrol
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

agent_position = 0
agent_velocity = np.linspace(-12, 12, 25)
pg = misc.ParametersHyperparallelepiped(agent_position, agent_velocity)

simulator = misc.Simulator(model_cruisecontrol.Model, pg.sample(sigma=0.05))

attacker = architecture.Attacker(simulator, 1, 10, 5, n_coeff=1)
defender = architecture.Defender(simulator, 2, 10)

misc.load_models(attacker, defender, args.dirname)

dt = 0.05
steps = 300


def run(mode=None):
    simulator.reset_to_random()
    conf_init = {
        "ag_pos": simulator.model.agent.position.numpy().tolist(),
        "ag_vel": simulator.model.agent.velocity.numpy().tolist(),
    }

    sim_t = []
    sim_ag_pos = []
    sim_ag_vel = []
    sim_ag_acc = []

    def rbf(x):
        x = x.reshape(1) if x.dim() == 0 else x
        w = np.array([5]) if mode == 0 else np.array([-5])
        phi = lambda x: np.exp(-((x * 0.2) ** 2))
        d = np.arange(len(w)) + 25
        r = np.abs(x[:, np.newaxis] - d)
        return w.dot(phi(r).T)

    t = 0
    with torch.no_grad():
        z = torch.rand(attacker.noise_size)
        atk_policy = attacker(z)

    if mode is not None:
        simulator.model.environment._fn = rbf

    for i in range(steps):
        oa = torch.tensor(simulator.model.agent.status)

        with torch.no_grad():
            def_policy = defender(oa)

        atk_input = atk_policy(0) if mode is None else None
        def_input = def_policy(dt)

        simulator.step(atk_input, def_input, dt)

        sim_ag_acc.append(def_input.numpy())
        sim_t.append(t)
        sim_ag_pos.append(simulator.model.agent.position.numpy())
        sim_ag_vel.append(simulator.model.agent.velocity.numpy())

        t += dt

    x = np.arange(0, 100, simulator.model.environment._dx)
    y = simulator.model.environment.get_fn(torch.tensor(x))

    return {
        "init": conf_init,
        "space": {"x": x.tolist(), "y": y.tolist()},
        "sim_t": np.array(sim_t).tolist(),
        "sim_ag_pos": np.array(sim_ag_pos).tolist(),
        "sim_ag_vel": np.array(sim_ag_vel).tolist(),
        "sim_ag_acc": np.array(sim_ag_acc).tolist(),
    }


records = []
for i in range(args.repetitions):
    sim = {}
    sim["up"] = run(0)
    sim["down"] = run(1)
    sim["atk"] = run()

    records.append(sim)

with open(os.path.join(args.dirname, "sims.json"), "w") as f:
    json.dump(records, f)
