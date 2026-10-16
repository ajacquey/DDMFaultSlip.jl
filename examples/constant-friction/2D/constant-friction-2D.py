import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../../publication.mplstyle')

def plot_figure_time():

    # Read time series
    t, R = np.loadtxt("outputs/constant_friction_2D_T_0.1.csv", delimiter=",", skiprows=1, usecols=[0, 4], unpack=True)
    L = np.sqrt(t)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), constrained_layout=True)

    # Plots
    axes[0].plot(t, L, color="xkcd:blue", label=r"Nominal fluid front")
    axes[0].plot(t, R, color="k", label=r"Rupture front")
    axes[1].plot(t, L, color="xkcd:blue")
    axes[1].plot(t, R, color="k")
       
    # Legend
    axes[0].legend(loc="upper left")

    # Limits
    axes[0].set_xlim(left=0, right=1)
    axes[1].set_xlim(left=1.0e-03, right=1.0e-00)
    axes[0].set_ylim(bottom=0, top=4)
    axes[1].set_ylim(bottom=1.0e-02, top=1.0e+01)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    
    # Labels 
    axes[0].set_xlabel(r"Normalized time, $t / t_{s}$")
    axes[1].set_xlabel(r"Normalized time, $t / t_{s}$")
    axes[0].set_ylabel(r"Rupture front, $R\left(t\right) / L\left(t\right)$")
    axes[1].set_ylabel(r"Rupture front, $R\left(t\right) / L\left(t\right)$")

    # Annotations
    axes[1].text(1.3e-03, 7, r"$T = 0.1$", horizontalalignment="left", verticalalignment="center")

    fig.savefig("constant-friction-2D-time.pdf", format="PDF", bbox_inches="tight")

    return np.nanmean(np.divide(R, L))

def plot_figure_profile(lmbda):
    # Read time series
    t = np.loadtxt("outputs/constant_friction_2D_T_0.1.csv", delimiter=",", skiprows=1, usecols=[0], unpack=True)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), constrained_layout=True)

    # Time stemps
    t_stamps = [0.25, 0.5, 0.75, 1.0]
    for tt in t_stamps:
        idx = np.where(tt <= t)[0][0]
        file = "outputs/constant_friction_2D_T_0.1_values/injection_{}.csv".format(idx)
        x, delta = np.loadtxt(file, delimiter=",", skiprows=1, usecols=[0,3], unpack=True)
        # Plots
        axes[0].plot(x, delta, ls="-", color="{}".format(1.0-tt), label=r"$t = {}\, t_s$".format(tt))
        axes[1].plot(x / (lmbda * np.sqrt(tt)), delta / np.sqrt(tt), ls ="-", color="k")

    # Legend
    axes[0].legend(loc="upper right")

    # Limits
    axes[0].set_xlim(left=-4, right=+4)
    axes[0].set_ylim(bottom=0.0, top=1.0)
    axes[1].set_xlim(left=-1.2, right=+1.2)
    axes[1].set_ylim(bottom=0.0, top=1.0)
    
    # Labels
    axes[0].set_xlabel(r"Distance, $x / L\left(t_{s}\right)$")
    axes[0].set_ylabel(r"Slip, $\delta \, \mu / \left(f \, \Delta p \, h\right)$")
    axes[1].set_xlabel(r"Normalized distance, $x / R\left(t\right)$")
    axes[1].set_ylabel(r"Normalized slip, $\delta \, \mu / \left(f \, \Delta p \, \sqrt{\alpha^{\prime} t}\right)$")

    fig.savefig("constant-friction-2D-profile.pdf", format="PDF", bbox_inches="tight")

if __name__ == '__main__':
    lmbda = plot_figure_time()
    plot_figure_profile(lmbda)