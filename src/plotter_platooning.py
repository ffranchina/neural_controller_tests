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

fields = [f"{a}_{m}" for a in ("leader", "follower") for m in ("pos", "vel", "acc")]

for r in records:
    for mode in ["pulse", "step_up", "step_down", "atk"]:
        for var in fields:
            r[mode][var] = np.array(r[mode][var])

        r[mode]["dist"] = r[mode]["leader_pos"] - r[mode]["follower_pos"]


def hist(time, pulse, step_up, step_down, atk, filename):
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharex=True)
    ax[0].plot(time, step_up * 100)
    ax[0].fill_between(time, step_up * 100, alpha=0.5)
    ax[0].set(xlabel="time (s)", ylabel="% correct")
    ax[0].title.set_text("Sudden acceleration")

    ax[1].plot(time, step_down * 100)
    ax[1].fill_between(time, step_down * 100, alpha=0.5)
    ax[1].set(xlabel="time (s)", ylabel="% correct")
    ax[1].title.set_text("Sudden brake")

    ax[2].plot(time, pulse * 100)
    ax[2].fill_between(time, pulse * 100, alpha=0.5)
    ax[2].set(xlabel="time (s)", ylabel="% correct")
    ax[2].title.set_text("Acceleration pulse")

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

    ax[0].plot(sim_time, sim_agent_pos, label="follower")
    ax[0].plot(sim_time, sim_env_pos, label="leader")
    ax[0].set(xlabel="time (s)", ylabel="position (m)")
    ax[0].legend()

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
        sample_trace = torch.tensor(records[i]["atk"]["dist"][-150:])
        robustness = float(robustness_computer.dqs.compute(dist=sample_trace))
        delta_pos = (
            records[i]["atk"]["leader_pos"][0] - records[i]["atk"]["follower_pos"][0]
        )
        delta_vel = (
            records[i]["atk"]["leader_vel"][0] - records[i]["atk"]["follower_vel"][0]
        )

        robustness_array[i] = robustness
        delta_pos_array[i] = delta_pos
        delta_vel_array[i] = delta_vel

    scatter(robustness_array, delta_pos_array, delta_vel_array, "atk_scatterplot.png")

if args.triplots:
    n = random.randrange(len(records))
    plot(
        records[n]["pulse"]["t"],
        records[n]["pulse"]["follower_pos"],
        records[n]["pulse"]["dist"],
        records[n]["pulse"]["follower_acc"],
        records[n]["pulse"]["leader_pos"],
        records[n]["pulse"]["leader_acc"],
        "triplot_pulse.png",
    )

    plot(
        records[n]["step_up"]["t"],
        records[n]["step_up"]["follower_pos"],
        records[n]["step_up"]["dist"],
        records[n]["step_up"]["follower_acc"],
        records[n]["step_up"]["leader_pos"],
        records[n]["step_up"]["leader_acc"],
        "triplot_step_up.png",
    )

    plot(
        records[n]["step_down"]["t"],
        records[n]["step_down"]["follower_pos"],
        records[n]["step_down"]["dist"],
        records[n]["step_down"]["follower_acc"],
        records[n]["step_down"]["leader_pos"],
        records[n]["step_down"]["leader_acc"],
        "triplot_step_down.png",
    )

    plot(
        records[n]["atk"]["t"],
        records[n]["atk"]["follower_pos"],
        records[n]["atk"]["dist"],
        records[n]["atk"]["follower_acc"],
        records[n]["atk"]["leader_pos"],
        records[n]["atk"]["leader_acc"],
        "triplot_attacker.png",
    )

if args.hist:
    size = len(records)
    pulse_pct = np.zeros_like(records[0]["pulse"]["dist"])
    step_up_pct = np.zeros_like(records[0]["step_up"]["dist"])
    step_down_pct = np.zeros_like(records[0]["step_down"]["dist"])
    atk_pct = np.zeros_like(records[0]["atk"]["dist"])

    for i in range(size):
        d = records[i]["pulse"]["dist"]
        pulse_pct = pulse_pct + np.logical_and(d > 2, d < 10)
        d = records[i]["step_up"]["dist"]
        step_up_pct = step_up_pct + np.logical_and(d > 2, d < 10)
        d = records[i]["step_down"]["dist"]
        step_down_pct = step_down_pct + np.logical_and(d > 2, d < 10)
        d = records[i]["atk"]["dist"]
        atk_pct = atk_pct + np.logical_and(d > 2, d < 10)

    time = records[0]["pulse"]["t"]
    pulse_pct = pulse_pct / size
    step_up_pct = step_up_pct / size
    step_down_pct = step_down_pct / size
    atk_pct = atk_pct / size

    hist(time, pulse_pct, step_up_pct, step_down_pct, atk_pct, "pct_histogram.png")
