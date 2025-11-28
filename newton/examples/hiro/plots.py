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
