import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font', family='serif', serif=['Times New Roman', 'Times', 'DejaVu Serif'])
mpl.rcParams['mathtext.fontset'] = 'stix'

def plot_computation_times():
    times_dict = {
        "XPBD spheres": 1.324238,
        "XPBD tet": 1.759468,
        "MuJoCo": 2.097803,
        # "Semi-Implicit Euler": 0.234679,
    }
    methods = list(times_dict.keys())
    times = list(times_dict.values())
    colors = ["#55A868", "#C44E52","#4C72B0"]

    plt.figure(figsize=(3.5, 2))
    ax = plt.gca()
    plt.bar(methods, times, color=colors)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylabel("Computation Time (s)")
    plt.yticks([0, 1, 2.0])

    for i, t in enumerate(times):
        plt.text(i, t + 0.02, f"{t:.3f}s", ha='center')
    plt.tight_layout()
    plt.savefig("newton/examples/hiro/computation_times.png", dpi=300)
    plt.show()


def plot_trajectory_errors():
    err_L1 = {
        "gt":    0.000000,
        "b1":    0.105229,
        "b2":    0.029209,
        "b3":    0.005568,
    }

    methods = list(err_L1.keys())
    values = list(err_L1.values())
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B3"]

    plt.figure(figsize=(3.5, 2))
    ax = plt.gca()
    plt.bar(methods, values, color=colors)

    # simplify frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.ylabel("Mean L1 Error")
    plt.title("Trajectory Error vs Ground Truth")

    # annotate
    for i, v in enumerate(values):
        plt.text(i, v + 0.002, f"{v:.3f}", ha='center')

    plt.tight_layout()
    plt.savefig("newton/examples/hiro/trajectory_L1_errors.png", dpi=300)
    plt.show()

plot_trajectory_errors()