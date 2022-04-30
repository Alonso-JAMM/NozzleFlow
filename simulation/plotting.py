from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve

import h5py


def plot_timewise_variations(sim_file):
    # Now plot the rho vs iteration for node=15
    node = 15
    rho = get_rho(sim_file)[node, :]
    T = get_T(sim_file)[node, :]
    M = get_M(sim_file)[node, :]
    P = get_p(sim_file)[node, :]

    output_freq = get_output_freq(sim_file)
    time = np.arange(len(rho)) * output_freq

    rho_exact_val = get_exact_rho(sim_file)[node]
    rho_exact = np.zeros_like(rho)
    rho_exact.fill(rho_exact_val)

    T_exact_val = get_exact_T(sim_file)[node]
    T_exact = np.zeros_like(T)
    T_exact.fill(T_exact_val)

    P_exact_val = get_exact_P(sim_file)[node]
    P_exact = np.zeros_like(P)
    P_exact.fill(P_exact_val)

    M_exact_val = get_exact_M(sim_file)[node]
    M_exact = np.zeros_like(M)
    M_exact.fill(M_exact_val)

    # Finally plot the data!
    fig = plt.figure(figsize=(10, 12), dpi=100)
    # suptitle = """ Timewise variations of density, temperature, pressure and
    # \n Mach number at the nozzle throat - Nonconservation form"""
    # fig.suptitle(suptitle, fontsize=20)
    plt.subplot(411)
    plt.plot(time, rho_exact, label="Analytic steady flow")
    plt.plot(time, rho, label="Simulation")
    plt.ylabel(r"$\frac{\rho}{\rho_0}$", rotation=0, labelpad=20, fontsize=20)
    plt.grid(axis="both")
    plt.legend()
    plt.subplot(412)
    plt.plot(time, T_exact, label="Analytic steady flow")
    plt.plot(time, T, label="Simulation")
    plt.ylabel(r"$\frac{T}{T_0}$", rotation=0, labelpad=20, fontsize=20)
    plt.grid(axis="both")
    plt.legend()
    plt.subplot(413)
    plt.plot(time, P_exact, label="Analytic steady flow")
    plt.plot(time, P, label="Simulation")
    plt.ylabel(r"$\frac{P}{P_0}$", rotation=0, labelpad=20, fontsize=20)
    plt.grid(axis="both")
    plt.legend()
    plt.subplot(414)
    plt.plot(time, M_exact, label="Analytic steady flow")
    plt.plot(time, M, label="Simulation")
    plt.ylabel(r"$M$", rotation=0, labelpad=20, fontsize=15)
    plt.grid(axis="both")
    plt.legend()
    plt.xlabel("Number of steps", fontsize=15)
    fig.set_tight_layout(True)
    plt.show()
    # plt.savefig("./plots_31_points/Timewise_variations.png")


def plot_timewise_residuals(sim_file):
    node = 15

    drho = get_drho(sim_file)[node, :]
    drho = np.absolute(drho)
    du = get_du(sim_file)[node, :]
    du = np.absolute(du)
    dT = get_dT(sim_file)[node, :]
    dT = np.absolute(dT)

    output_freq = get_output_freq(sim_file)
    time = np.arange(len(drho)) * output_freq

    fig = plt.figure(figsize=(8, 6), dpi=100)
    # suptitle = """ Timewise variations of the absolute values of the
    # derivatives of density, \n and pressure at the nozzle throat -
    # Nonconservation form"""
    # fig.suptitle(suptitle)

    plt.yscale("log")
    # plt.plot(time, dT, label="temperature residuals")
    plt.plot(time, du, label=r"$|(\frac{\partial V}{\partial t})_{av}|$",
             linestyle="dashed")
    plt.plot(time, drho, label=r"$|(\frac{\partial \rho}{\partial t})_{av}|$")
    plt.legend()
    plt.grid()
    plt.ylabel("Residuals")
    plt.xlabel("Number of iterations")
    fig.set_tight_layout(True)
    plt.show()
    # plt.savefig("./plots_31_points/Timewise_residuals.png")


def plot_mdots(sim_file):
    x = get_x(sim_file)
    mdot = get_m_dot(sim_file)
    # times = ["0", "50", "100", "150", "200", "700"]
    time_indices = [0, 50, 100, 150, 200, 700]
    output_freq = get_output_freq(sim_file)

    fig, ax = plt.subplots(figsize=(6, 8), dpi=100)
    ax.set_ylim(0, 1.9)
    ax.set_xlim(0, 3)
    ax.grid(True)

    for time in time_indices:
        time = int(time / output_freq)
        label = str(time) + r"$\Delta t$"
        ax.plot(x, mdot[:, time], label=label)
        # mdot = []

    # Analytic mdot
    m_exact = get_m_dot_exact(sim_file)
    ax.plot(x, m_exact, label="Analytic steady flow", linestyle="dashed")

    plt.ylabel(r"Nondimensional mass flow $\frac{\rho u A}{\rho_0 a_0 A*}$",
               fontsize=14)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=14)

    fig.set_tight_layout(True)

    plt.legend()
    plt.show()
    # plt.savefig("./plots_31_points/mdots.png")


def plot_steady_mach(sim_file):
    x = get_x(sim_file)
    M_exact = get_exact_M(sim_file)
    M = get_M(sim_file)[:, -1]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(x, M_exact, label="Analytic steady flow")
    ax.plot(x, M, "o", label="Simulation")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3.5)

    plt.ylabel(r"Mach number", fontsize=14)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=14)

    fig.set_tight_layout(True)
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig("./plots_31_points/Mach_numbers.png")


def plot_steady_density(sim_file):
    x = get_x(sim_file)
    rho = get_rho(sim_file)[:, -1]
    rho_exact = get_exact_rho(sim_file)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.plot(x, rho_exact, label="Analytic steady flow")
    ax.plot(x, rho, "o", label="Simulation")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.1)

    plt.ylabel(r"Nondimensional density $\frac{\rho}{\rho_0}$", fontsize=14)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=14)

    fig.set_tight_layout(True)
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig("./plots_31_points/steady_densities.png")


class UpdateDensity:
    def __init__(self, ax, sim_file, offset):
        self.offset = offset
        self.x = get_x(sim_file)
        self.rho = get_rho(sim_file)
        self.exact_line = ax.plot([], [], "b-",
                                  label="Exact (Steady State)")[0]
        self.line = ax.plot([], [], "k-", label="Simulation")[0]
        exact_rho = get_exact_rho(sim_file)
        self.exact_line.set_data(self.x, exact_rho)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_ylim(0, 1.5)
        self.ax.set_xlim(0, 3)
        self.ax.grid(True)

    def __call__(self, i):
        # Try to repeat initial frame at the beginning

        if i < self.offset:
            self.line.set_data(self.x, self.rho[:, 0])
            return self.line,

        i -= self.offset
        self.line.set_data(self.x, self.rho[:, i])
        return self.line,


class UpdateMass:
    def __init__(self, ax, sim_file, offset):
        self.offset = offset
        self.x = get_x(sim_file)
        self.mdot = get_m_dot(sim_file)
        m_exact = [1.2**(-2.5)*sqrt(5/6) for i in self.x]
        self.line = ax.plot([], [], "k--", label="Simulation")[0]
        self.exact_line = ax.plot(self.x, m_exact, "b-",
                                  label="Exact (Steady State)")[0]
        self.ax = ax

        # Set up plot parameters
        self.ax.set_ylim(-1, 2.5)
        self.ax.set_xlim(0, 3)
        self.ax.grid(True)

    def __call__(self, i):
        if i < self.offset:
            self.line.set_data(self.x, self.mdot[:, 0])
            return self.line,

        i -= self.offset

        self.line.set_data(self.x, self.mdot[:, i])
        return self.line,


class UpdatePressure:
    def __init__(self, ax, sim_filem, offset):
        self.offset = offset
        self.line = ax.plot([], [], "k-", label="Simulation", linewidth=2)[0]
        self.x = get_x(sim_file)
        self.p = get_p(sim_file)
        self.exact_line = ax.plot([], [], "b-",
                                  label="Exact (Steady State)")[0]
        exact_p = get_exact_P(sim_file)
        self.exact_line.set_data(self.x, exact_p)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_ylim(0, 1.5)
        self.ax.set_xlim(0, 3)
        self.ax.grid(True)

    def __call__(self, i):
        if i < self.offset:
            self.line.set_data(self.x, self.p[:, 0])
            return self.line,

        i -= self.offset

        self.line.set_data(self.x, self.p[:, i])
        return self.line,


class UpdateVelocity:
    def __init__(self, ax, sim_filem, offset):
        self.offset = offset
        self.line = ax.plot([], [], "k-", label="Simulation", linewidth=2)[0]
        self.x = get_x(sim_file)
        self.u = get_u(sim_file)
        self.exact_line = ax.plot([], [], "b-",
                                  label="Exact (Steady State)")[0]
        exact_u = get_exact_u(sim_file)
        self.exact_line.set_data(self.x, exact_u)
        self.ax = ax

        # Set up plot parameters
        self.ax.set_ylim(-.5, 2.5)
        self.ax.set_xlim(0, 3)
        self.ax.grid(True)

    def __call__(self, i):
        if i < self.offset:
            self.line.set_data(self.x, self.u[:, 0])
            return self.line,

        i -= self.offset

        self.line.set_data(self.x, self.u[:, i])
        return self.line,


class UpdatePlots:
    def __init__(self, ax, sim_filem, offset):
        self.offset = offset
        self.x = get_x(sim_file)
        # density
        self.rho = get_rho(sim_file)
        self.rho_line = ax[0].plot([], [], "k-", label="Simulation")[0]
        exact_rho = get_exact_rho(sim_file)
        self.exact_rho = ax[0].plot([], [], "b-",
                                    labe="Exact (Steady State)")[0]
        self.exact_rho.set_data(self.x, exact_rho)
        # mass flow
        self.mdot = get_m_dot(sim_file)
        m_exact = [1.2**(-2.5)*sqrt(5/6) for i in self.x]
        self.mdot_line = ax[1].plot([], [], "k-", label="Simulation")[0]
        self.exact_mdot = ax[1].plot(self.x, m_exact, "b-",
                                     label="Exact (Steady State)")[0]
        # pressure
        self.p = get_p(sim_file)
        self.p_line = ax[2].plot([], [], "k-", label="Simulation",
                                 linewidth=2)[0]
        self.p_exact = ax[2].plot([], [], "b-",
                                  label="Exact (Steady State)")[0]
        exact_p = get_exact_P(sim_file)
        self.p_exact.set_data(self.x, exact_p)

        self.ax = ax

        # Set up plot parameters
        self.ax[0].set_ylim(0, 1.5)
        self.ax[0].grid(True)
        self.ax[1].set_ylim(-1, 2.5)
        self.ax[1].grid(True)
        self.ax[2].set_ylim(0, 1.5)
        self.ax[2].set_xlim(0, 3)
        self.ax[2].grid(True)

    def __call__(self, i):

        if i < self.offset:
            self.rho_line.set_data(self.x, self.rho[:, 0])
            self.mdot_line.set_data(self.x, self.mdot[:, 0])
            self.p_line.set_data(self.x, self.p[:, 0])
            return self.p_line, self.mdot_line, self.rho_line,

        i -= self.offset

        self.rho_line.set_data(self.x, self.rho[:, i])
        self.mdot_line.set_data(self.x, self.mdot[:, i])
        self.p_line.set_data(self.x, self.p[:, i])
        return self.p_line, self.mdot_line, self.rho_line,


def density_v_x(sim_file):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    # fig.set_dpi(300)
    start_offset = 20
    ud = UpdateDensity(ax, sim_file, start_offset)
    anim = FuncAnimation(fig, ud, frames=range(0, 375, 3), interval=50,
                         blit=True)
    plt.legend()
    plt.ylabel(r"$\frac{\rho}{\rho_0}$", rotation=0, labelpad=20, fontsize=18)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=12)
    fig.suptitle("Nondimensional density along the nozzle over time")
    plt.show()
    # anim.save("./plots/rho_v_x_time.mp4")


def mdot_v_x(sim_file):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    # fig.set_dpi(300)
    start_offset = 50
    ud = UpdateMass(ax, sim_file, start_offset)
    anim = FuncAnimation(fig, ud, frames=range(0, 375, 3), interval=50,
                         blit=True)
    plt.legend()
    label = r"Nondimensional mass flow $\frac{\rho u A}{\rho_0 a_0 A*}$"
    plt.ylabel(label, fontsize=14)
    label = "Nondimensional distance through the nozzle (x)"
    plt.xlabel(label, fontsize=14)
    label = "Nondimensional mass flow along the nozzle over time"
    fig.suptitle(label, fontsize=18)
    plt.show()


def p_v_x(sim_file):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    # fig.set_dpi(300)
    start_offset = 50
    ud = UpdatePressure(ax, sim_file, start_offset)
    anim = FuncAnimation(fig, ud, frames=range(0, 375, 3), interval=50,
                         blit=True)
    plt.legend()
    plt.ylabel(r"$\rho T$", fontsize=14)
    label = "Nondimensional distance through the nozzle (x)"
    plt.xlabel(label, fontsize=12)
    label = "Nondimensional pressure along the nozzle over time"
    fig.suptitle(label)
    plt.show()


def u_v_x(sim_file):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    # fig.set_dpi(300)
    start_offset = 50
    ud = UpdateVelocity(ax, sim_file, start_offset)
    anim = FuncAnimation(fig, ud, frames=range(0, 375, 3), interval=50,
                         blit=True)
    plt.legend()
    plt.ylabel(r"Velocity, $u$", fontsize=14)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=12)
    fig.suptitle("Nondimensional velocity along the nozzle over time")
    plt.show()


def all_plots(sim_file):
    fig, ax = plt.subplots(3, 1, sharex="all")
    fig.set_size_inches(10, 6)
    fig.set_dpi(300)
    start_offset = 50
    ud = UpdatePlots(ax, sim_file, start_offset)
    anim = FuncAnimation(fig, ud, frames=range(0, 375, 3), interval=50,
                         blit=True)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels)
    ax[0].set_ylabel(r"Density, $\frac{\rho}{\rho_0}$", fontsize=14)
    ax[1].set_ylabel(r"Mass flow, $\frac{\rho u A}{\rho_0 a_0 A*}$",
                     fontsize=14)
    ax[2].set_ylabel(r"Pressure, $\rho T$", fontsize=14)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=14)
    fig.suptitle("Nondimensional variables over time", fontsize=18)
    # plt.show()
    anim.save("./plots/vars_time.mp4")


def get_x(sim_file):
    """ Gets the array containing the x-positions of all the nodes """
    sim_data = h5py.File(sim_file, "r")
    x = sim_data["ConstData/x"]
    return x[...]


def get_A(sim_file):
    """ Gets the array containing the nondimensional area at each node """
    sim_data = h5py.File(sim_file, "r")
    A = sim_data["ConstData/A"]
    return A[...]


def get_T(sim_file):
    """ Gets the array containing the nondimensional temperature on each node
    over the entire simulation """
    sim_data = h5py.File(sim_file, "r")
    T = sim_data["SimOut/T"]
    return T[...]


def get_rho(sim_file):
    """ Gets the array containing the nondimensional density on each node over
    the entire simulation """
    sim_data = h5py.File(sim_file, "r")
    rho = sim_data["SimOut/rho"]
    return rho[...]


def get_u(sim_file):
    """ Gets the array containing the nondimensional velocity on each node over
    the entire simulation """
    sim_data = h5py.File(sim_file, "r")
    u = sim_data["SimOut/u"]
    return u[...]


def get_m_dot(sim_file):
    """ Gets the array containing the nondimensional mass flow on each node
    over the entire simulation """
    sim_data = h5py.File(sim_file, "r")
    m_dot = sim_data["PostProcess/m_dot"]
    return m_dot[...]


def get_p(sim_file):
    """ Gets the array containing the nondimensional pressure on each node
    over the entire simulation """
    sim_data = h5py.File(sim_file, "r")
    p = sim_data["PostProcess/p"]
    return p[...]


def get_M(sim_file):
    """ Gets the array containing the nondimensional Mach number on each node
    over the entire simulation """
    sim_data = h5py.File(sim_file, "r")
    M = sim_data["PostProcess/M"]
    return M[...]


def get_time(sim_file):
    """ Gets the array containing the nondimensional time on each step of the
    simulation """
    sim_data = h5py.File(sim_file, "r")
    time = sim_data["SimOut/time"]
    return time[...]


def get_output_freq(sim_file):
    """ Gets the output frequency of the simulation """
    sim_data = h5py.File(sim_file, "r")
    sim_out = sim_data["SimOut"]
    output_freq = sim_out.attrs["OutputFreq"]
    return output_freq


def get_drho(sim_file):
    """ Gets the time derivative of the density at each node"""
    sim_data = h5py.File(sim_file, "r")
    drho = sim_data["SimOut/Residuals/drho"]
    return drho[...]


def get_dT(sim_file):
    """ Gets the time derivative of the temperature at each node"""
    sim_data = h5py.File(sim_file, "r")
    dT = sim_data["SimOut/Residuals/dT"]
    return dT[...]


def get_du(sim_file):
    """ Gets the time derivative of the density at each node"""
    sim_data = h5py.File(sim_file, "r")
    du = sim_data["SimOut/Residuals/du"]
    return du[...]


def get_exact_M(sim_file):
    """ Gets the array containing the exact Mach number on each node"""
    sim_data = h5py.File(sim_file, "r")
    M_exact = sim_data["ExactSol/M"]
    return M_exact[...]


def get_exact_rho(sim_file):
    """ Gets the array containing the exact density at each node """
    sim_data = h5py.File(sim_file, "r")
    rho_exact = sim_data["ExactSol/rho"]
    return rho_exact[...]


def get_exact_P(sim_file):
    """ Gets the array containing the exact pressure at each node """
    sim_data = h5py.File(sim_file, "r")
    P_exact = sim_data["ExactSol/P"]
    return P_exact[...]


def get_exact_u(sim_file):
    """ Gets the array containing the exact velocity at each node """
    sim_data = h5py.File(sim_file, "r")
    u_exact = sim_data["ExactSol/u"]
    return u_exact[...]


def get_exact_T(sim_file):
    """ Gets the array containing the exact temperature at each node """
    sim_data = h5py.File(sim_file, "r")
    T_exact = sim_data["ExactSol/T"]
    return T_exact[...]


def get_m_dot_exact(sim_file):
    """ Gets the array containing the exact mass flow at each node """
    sim_data = h5py.File(sim_file, "r")
    m_dot_exact = sim_data["ExactSol/m_dot_exact"]
    return m_dot_exact[...]


def save_exact_values(sim_file):
    """ Calculates and saves the exact Mach number for each node """
    M = []
    A = get_A(sim_file)
    x = get_x(sim_file)
    u = get_u(sim_file)[:, -1]
    T = get_T(sim_file)[:, -1]

    def M_func(M):
        return (1/M**2)*((5/6)+(1/6)*M**2)**6 - A**2

    M_o = u / np.sqrt(T)
    M = fsolve(M_func, M_o)
    rho = (1 + 0.2*M**2)**(-2.5)
    P = (1 + 0.2*M**2)**(-3.5)
    T = (1 + 0.2*M**2)**(-1)
    u = M*np.sqrt(T)
    m_dot_exact = np.zeros_like(x)
    m_dot_exact.fill(1.2**(-2.5)*sqrt(5/6))

    sim_data = h5py.File(sim_file, "r+")
    exact_sol_grp = sim_data["ExactSol"]
    create_or_update_ds(exact_sol_grp, "M", M)
    create_or_update_ds(exact_sol_grp, "rho", rho)
    create_or_update_ds(exact_sol_grp, "P", P)
    create_or_update_ds(exact_sol_grp, "T", T)
    create_or_update_ds(exact_sol_grp, "u", u)
    create_or_update_ds(exact_sol_grp, "m_dot_exact", m_dot_exact)


def save_m_dot(sim_file):
    """ Calculates and saves the mass flow at each node """
    rho = get_rho(sim_file)
    u = get_u(sim_file)
    A = get_A(sim_file)[:, np.newaxis]
    m_dot = rho*u*A
    sim_data = h5py.File(sim_file, "r+")
    post_grp = sim_data["PostProcess"]
    create_or_update_ds(post_grp, "m_dot", m_dot)


def save_M(sim_file):
    """ Calculates and saves the Mach number at each node from the simulation
    data """
    u = get_u(sim_file)
    T = get_T(sim_file)

    M = u / np.sqrt(T)

    sim_data = h5py.File(sim_file, "r+")
    post_grp = sim_data["PostProcess"]
    create_or_update_ds(post_grp, "M", M)


def save_p(sim_file):
    """ Calculates and saves the pressure at each node """
    rho = get_rho(sim_file)
    T = get_T(sim_file)
    p = rho * T
    sim_data = h5py.File(sim_file, "r+")
    post_grp = sim_data["PostProcess"]
    create_or_update_ds(post_grp, "p", p)


def save(sim_file):
    """ Saves different calculated variables """
    save_exact_values(sim_file)
    save_m_dot(sim_file)
    save_p(sim_file)
    save_M(sim_file)


def create_or_update_ds(group, ds_name, data):
    """ Creates a new dataset in group and adds data to it, if the dataset
    already exists, then it updates its values. """
    if ds_name not in group:
        group.create_dataset(ds_name, data=data)
    else:
        group[ds_name][...] = data


if __name__ == "__main__":
    # file containing the simulation data
    sim_file = "./output/simulation.h5"

    save(sim_file)
    # plot_timewise_residuals(sim_file)
    # plot_timewise_variations(sim_file)
    # plot_steady_density(sim_file)
    plot_mdots(sim_file)
