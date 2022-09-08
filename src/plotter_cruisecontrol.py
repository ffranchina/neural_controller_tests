import json
import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser()
parser.add_argument("dirname", help="model's directory")
parser.add_argument(
    "--triplots", default=False, action="store_true", help="Generate triplots"
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

for r in records:
    for mode in ["up"]:

        for var in ["t", "pos", "vel", "acc", "road"]:
            r[mode][var] = np.array(r[mode][var])


def hist(time, up, down, filename):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3), sharex=True)

    ax[0].plot(time, up * 100)
    ax[0].fill_between(time, up * 100, alpha=0.5)
    ax[0].set(xlabel="time (s)", ylabel="% correct")
    ax[0].title.set_text("Hill")

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)


def plot(road, sim_time, sim_agent_pos, sim_agent_vel, sim_agent_acc, filename):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))

    ax[0].plot(np.linspace(0, 50, 1000), road, zorder=-1)
    ax[0].set(xlabel="space (m)", ylabel="elevation (m)")
    ax[0].set_ylim((np.min(road) - 1, np.max(road) + 1))
    ax[0].set_xlim((-1, sim_agent_pos[-1] + 1))

    ax[1].axhline(4.75, ls="--", color="r")
    ax[1].axhline(5.25, ls="--", color="r")
    ax[1].plot(sim_time, sim_agent_vel)
    ax[1].set(xlabel="time (s)", ylabel="velocity (m/s)")

    ax[2].plot(sim_time, np.clip(sim_agent_acc, -5, 5))
    ax[2].set(xlabel="time (s)", ylabel="acceleration ($m/s^2$)")

    fig.tight_layout()
    fig.savefig(os.path.join(args.dirname, filename), dpi=150)


if args.triplots:
    n = random.randrange(len(records))
    plot(
        records[n]["up"]["road"],
        records[n]["up"]["t"],
        records[n]["up"]["pos"],
        records[n]["up"]["vel"],
        records[n]["up"]["acc"],
        "triplot_up.png",
    )

    # plot(
    #     records[n]["down"]["road"],
    #     records[n]["down"]["t"],
    #     records[n]["down"]["pos"],
    #     records[n]["down"]["vel"],
    #     records[n]["down"]["acc"],
    #     "triplot_down.png",
    # )

if args.hist:
    size = len(records)
    up_pct = np.zeros_like(records[0]["up"]["vel"])
    # down_pct = np.zeros_like(records[0]["down"]["vel"])

    for i in range(size):
        t = records[i]["up"]["vel"]
        up_pct = up_pct + np.logical_and(t > 4.75, t < 5.25)
        # t = records[i]["down"]["vel"]
        # down_pct = down_pct + np.logical_and(t > 4.75, t < 5.25)

    time = records[0]["up"]["t"]
    up_pct = up_pct / size
    down_pct = None

    hist(time, up_pct, down_pct, "pct_histogram.png")
