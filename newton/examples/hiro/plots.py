import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import os

mpl.rc('font', family='serif', serif=['Times New Roman', 'Times', 'DejaVu Serif'])
mpl.rcParams['mathtext.fontset'] = 'stix'
COLORS_dict = {
    "gt": "#4C72B0",
    "b1": "#C44E52",
    "b2": "#55A868",
    "b3": "#8172B3"
}

NAME_dict = {
    "gt": "MuJoCo",
    "b1": "Semi-Implicit \nEuler",
    "b2": "XPBD \nSpheres",
    "b3": "XPBD \nTriangles"
}

def plot_computation_times():
    times_dict = {
        "b2": 1.266184,
        "b3": 1.770638,
        "b1": 0.256699,
        "gt": 2.126597,
        
    }

    methods = list(times_dict.keys())
    times = list(times_dict.values())
    colors = [COLORS_dict[m] for m in methods]

    plt.figure(figsize=(3.5, 2))
    ax = plt.gca()
    plt.bar([NAME_dict[m] for m in methods], times, color=colors)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylabel("Total Computation Time (s)")
    plt.yticks([0, 1, 2.0])

    for i, t in enumerate(times):
        plt.text(i, t + 0.02, f"{t:.3f}s", ha='center')
    plt.tight_layout()
    plt.savefig("newton/examples/hiro/computation_times.png", dpi=300)
    plt.show()


def plot_goal_pos_errors():
    err_L1 = {
        # "gt":    0.000000,
        "b2":    0.037126,
        "b3":    0.005968,
        "b1":    0.131204,
    }

    methods = list(err_L1.keys())
    values = list(err_L1.values())
    colors = [COLORS_dict[m] for m in methods]

    plt.figure(figsize=(3.5, 2))
    ax = plt.gca()
    plt.bar([NAME_dict[m] for m in methods], values, color=colors)

    # simplify frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylabel("Mean L1 Error")
    # plt.title("Goal Position Error Relative to MuJoCo")

    # annotate
    for i, v in enumerate(values):
        plt.text(i, v + 0.002, f"{v:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig("newton/examples/hiro/trajectory_L1_errors.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_computation_times()
    plot_goal_pos_errors()