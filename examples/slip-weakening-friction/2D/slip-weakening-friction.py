import numpy as np
import matplotlib.pyplot as plt
plt.style.use('publication.mplstyle')

def plot_figure():

    # Read continuation
    t_cont, delta_cont, ar_cont = np.loadtxt("slip-weakening-friction-cont.csv", delimiter=",", skiprows=1, usecols=[0,1,2], unpack=True)
    t_cont = np.insert(t_cont, 0, 0.0, axis=0)
    delta_cont = np.insert(delta_cont, 0, 0.0, axis=0)
    ar_cont = np.insert(ar_cont, 0, 0.0, axis=0)

    # Read simulation
    t, delta, ar = np.loadtxt("slip-weakening-friction-gs.csv", delimiter=",", skiprows=1, usecols=[0,1,2], unpack=True)
    t = np.insert(t, 0, 0.0, axis=0)
    delta = np.insert(delta, 0, 0.0, axis=0)
    ar = np.insert(ar, 0, 0.0, axis=0)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5), constrained_layout=True)

    # Plots
    axes[0].plot(t_cont, ar_cont, color="k")
    axes[1].plot(t_cont, delta_cont, color="k")
    axes[0].plot(t, ar, color="xkcd:red")
    axes[1].plot(t, delta, color="xkcd:red")

    # Limits
    axes[0].set_xlim(left=0, right=15)
    axes[1].set_xlim(left=0, right=15)
    axes[0].set_ylim(bottom=0, top=15)
    axes[1].set_ylim(bottom=0, top=2.5)
    
    # Labels 
    axes[0].set_xlabel(r"Normalized time, $\sqrt{\alpha^{\prime} t} / a_{w}$")
    axes[1].set_xlabel(r"Normalized time, $\sqrt{\alpha^{\prime} t} / a_{w}$")
    axes[0].set_ylabel(r"Crack length, $a\left(t\right) / a_{w}$")
    axes[1].set_ylabel(r"Peak slip, $\delta\left(x=0\right) / \delta_{w}$")

    # Annotations
    axes[0].text(0.05*15, 0.9*15, r"$f_{r}/f_{p} = 0.6$", horizontalalignment="left", verticalalignment="center")
    axes[0].text(0.05*15, 0.8*15, r"$\Delta p/\sigma^{\prime}_{0} = 0.5$", horizontalalignment="left", verticalalignment="center")
    axes[0].text(0.05*15, 0.7*15, r"$\tau_{0}/\tau_{p} = 0.55$", horizontalalignment="left", verticalalignment="center")
    
    fig.savefig("slip-weakening-friction-2D.pdf", format="PDF", bbox_inches="tight")

if __name__ == '__main__':
    plot_figure()