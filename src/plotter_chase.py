import os
import random
import json

import torch

import misc

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("dirname", help="model's directory")
parser.add_argument(
    "--triplots", default=False, action="store_true", help="Generate triplots"
)
parser.add_argument(
    "--scatter", default=False, action="store_true", help="Generate scatterplot"
)
parser.add_argument(
    "--hist", default=False, action="store_true", help="Generate histograms"
)
parser.add_argument("--dark", default=False, action="store_true", help="Use dark theme")
args = parser.parse_args()

if args.dark:
    plt.style.use("./qb-common_dark.mplstyle")

with open(os.path.join(args.dirname, "sims.json"), "r") as f:
    records = json.load(f)

agents = ("leader", "follower")
fields = [f"{a}_{m}" for a in agents for m in ("pos", "vel", "acc")]

for r in records:
    for var in fields:
        r["atk"][var] = np.array(r["atk"][var])

    for ag in agents:
        d = r["atk"]["leader_pos"] - r["atk"][f"{ag}_pos"]
        r["atk"][f"{ag}_dist"] = np.linalg.norm(d, axis=1)


def hist(time, pulse, step_up, step_down, atk, filename):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharex=True)

    ax[3].plot(time, atk * 100)
    ax[3].fill_between(time, atk * 100, alpha=0.5)
    ax[3].set(xlabel="time (s)", ylabel="% correct")
    ax[3].title.set_text("Against attacker")

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)


def scatter(robustness_array, delta_pos_array, delta_vel_array, filename):
    fig, ax = plt.subplots(figsize=(10, 5))

    customnorm = mcolors.TwoSlopeNorm(0)
    sp = ax.scatter(
        delta_vel_array,
        delta_pos_array,
        c=robustness_array,
        cmap="RdYlGn",
        norm=customnorm,
    )
    ax.set(
        xlabel="$\\Delta$v between leader and follower ($m/s$)", ylabel="Distance ($m$)"
    )

    cb = fig.colorbar(sp)
    cb.ax.set_xlabel("$\\rho$")

    fig.suptitle("Initial conditions vs robustness $\\rho$")
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)


def plot(
    sim_time,
    sim_agent_pos,
    sim_agent_dist,
    sim_agent_acc,
    sim_env_pos,
    sim_env_acc,
    filename,
):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    ax[1].plot(sim_time, sim_agent_dist)
    ax[1].set(xlabel="time (s)", ylabel="distance (m)")
    ax[1].axhline(2, ls="--", color="r")
    ax[1].axhline(10, ls="--", color="r")

    ax[2].plot(sim_time, np.clip(sim_agent_acc, -3, 3), label="follower")
    ax[2].plot(sim_time, np.clip(sim_env_acc, -3, 3), label="leader")
    ax[2].set(xlabel="time (s)", ylabel="acceleration ($m/s^2$)")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)


if args.scatter:
    size = len(records)

    robustness_formula = "G(dist <= 10 & dist >= 2)"
    robustness_computer = misc.RobustnessComputer(robustness_formula)

    robustness_array = np.zeros(size)
    delta_pos_array = np.zeros(size)
    delta_vel_array = np.zeros(size)

    for i in range(size):
        sample_trace = torch.tensor(records[i]["atk"]["follower_dist"][-150:])
        robustness = float(robustness_computer.dqs.compute(dist=sample_trace))
        delta_pos = (
            records[i]["atk"]["leader_pos"][0] - records[i]["atk"]["follower_pos"][0]
        )
        delta_vel = (
            records[i]["atk"]["leader_vel"][0] - records[i]["atk"]["follower_vel"][0]
        )

        robustness_array[i] = robustness
        delta_pos_array[i] = np.linalg.norm(delta_pos, axis=0)
        delta_vel_array[i] = np.linalg.norm(delta_vel, axis=0)

    scatter(robustness_array, delta_pos_array, delta_vel_array, "atk_scatterplot.png")

if args.triplots:
    n = random.randrange(len(records))
    plot(
        records[n]["atk"]["t"],
        records[n]["atk"]["follower_pos"],
        records[n]["atk"]["follower_dist"],
        np.linalg.norm(records[n]["atk"]["follower_acc"], axis=1),
        records[n]["atk"]["leader_pos"],
        np.linalg.norm(records[n]["atk"]["leader_acc"], axis=1),
        "triplot_attacker.png",
    )

if args.hist:
    size = len(records)
    atk_pct = np.zeros(records[0]["atk"]["follower_dist"].shape[0])

    for i in range(size):
        d = records[i]["atk"]["follower_dist"]
        atk_pct = atk_pct + np.logical_and(d > 2, d < 10)

    time = records[0]["atk"]["t"]

    pulse_pct = None
    step_up_pct = None
    step_down_pct = None
    atk_pct = atk_pct / size

    hist(time, pulse_pct, step_up_pct, step_down_pct, atk_pct, "pct_histogram.png")
