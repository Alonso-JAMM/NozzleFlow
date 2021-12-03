import json
import os
import math

from math import sqrt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import fsolve


def plot_timewise_variations(sim_steps):
    # Now plot the rho vs iteration for node=15
    rho = []
    u = []
    T = []
    time = []
    for step in sim_steps:
        time.append(float(step))
        rho.append(sim_steps[step]["nodes"][15]["rho"])
        u.append(sim_steps[step]["nodes"][15]["u"])
        T.append(sim_steps[step]["nodes"][15]["T"])

    # sort the lists in ascending order according to time
    # https://stackoverflow.com/questions/9764298/how-to-sort-two-lists-which-reference-each-other-in-the-exact-same-way
    time, rho, u, T = zip(*sorted(zip(time, rho, u, T)))

    time = np.array(time)
    rho = np.array(rho)
    u = np.array(u)
    T = np.array(T)

    # calculated variables
    M = u / np.sqrt(T)
    P = rho*T

    # exact values
    rho_exact = np.zeros_like(rho)
    rho_exact.fill(1.2**(-2.5))
    T_exact = np.zeros_like(T)
    T_exact.fill(1.2**(-1))
    P_exact = np.zeros_like(P)
    P_exact.fill(1.2**(-3.5))
    M_exact = np.ones_like(M)

    print(rho_exact)
    print(T_exact)

    # Finally plot the data!
    fig = plt.figure(figsize=(10,12), dpi=400)
    #fig.suptitle("Timewise variations of density, temperature, pressure and \n Mach number at the nozzle throat - Nonconservation form", fontsize=20)
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
    #plt.show()
    #plt.savefig("./plots_31_points/Timewise_variations.png")


def plot_first_step_vals(sim_steps):
    first_step = sim_steps["0"]
    for node in first_step["nodes"]:
        print(f"{node}")

def plot_timewise_residuals(sim_steps):
    drho = []
    du = []
    dT = []
    time = []
    for step in sim_steps:
        time.append(float(step))
        drho.append(sim_steps[step]["residuals"][15]["drho"])
        du.append(sim_steps[step]["residuals"][15]["du"])
        dT.append(sim_steps[step]["residuals"][15]["dT"])

    # sort
    fig = plt.figure(figsize=(8,6), dpi=400)
    #fig.suptitle("Timewise variations of the absolute values of the derivatives of density, \n and pressure at the nozzle throat - Nonconservation form")
    time, drho, du, dT = zip(*sorted(zip(time, drho, du, dT)))
    drho = np.abs(np.array(drho))
    du = np.abs(np.array(du))
    dT = np.abs(np.array(dT))
    plt.yscale("log")
    #plt.plot(time, dT, label="temperature residuals")
    plt.plot(time, du, label=r"$|(\frac{\partial V}{\partial t})_{av}|$", linestyle="dashed")
    plt.plot(time, drho, label=r"$|(\frac{\partial \rho}{\partial t})_{av}|$")
    plt.legend()
    plt.grid()
    plt.ylabel("Residuals")
    plt.xlabel("Number of iterations")
    #plt.show()
    fig.set_tight_layout(True)
    plt.savefig("./plots_31_points/Timewise_residuals.png")


def plot_mdots(sim_steps):
    x = []
    mdot = []
    times = ["0", "50", "100", "150", "200", "700"]

    for node in sim_steps["0"]["nodes"]:
        x.append(float(node["x"]))

    #fig = plt.figure(figsize=(8,6))
    fig, ax = plt.subplots(figsize=(6,8), dpi=400)
    ax.set_ylim(0, 1.9)
    ax.set_xlim(0, 3)
    ax.grid(True)

    for time in times:
        for node in sim_steps[time]["nodes"]:
            rho = float(node["rho"])
            u = float(node["u"])
            A = float(node["A"])
            mdot.append(rho*u*A)
        label = time + r"$\Delta t$"
        ax.plot(x, mdot, label=label)
        mdot = []

    # Analytic mdot
    m_exact = [1.2**(-2.5)*sqrt(5/6) for i in x]
    ax.plot(x, m_exact, label="Analytic steady flow", linestyle="dashed")


    plt.ylabel(r"Nondimensional mass flow $\frac{\rho u A}{\rho_0 a_0 A*}$", fontsize=14)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=14)

    fig.set_tight_layout(True)

    plt.legend()
    #plt.show()
    plt.savefig("./plots_31_points/mdots.png")


def plot_steady_mach(sim_steps):
    keys = sorted(sim_steps.keys(), key=int)
    last_step = keys[-1]

    x = []
    M_exact = exact_M(sim_steps)
    M = []

    for node in sim_steps[last_step]["nodes"]:
        x.append(float(node["x"]))
        u = float(node["u"])
        T = float(node["T"])
        M_i = u / sqrt(T)
        M.append(M_i)


    fig, ax = plt.subplots(figsize=(8,6), dpi=100)
    ax.plot(x, M_exact, label="Analytic steady flow")
    ax.plot(x, M, "o", label="Simulation")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3.5)

    plt.ylabel(r"Mach number", fontsize=14)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=14)

    fig.set_tight_layout(True)
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig("./plots_31_points/Mach_numbers.png")


def plot_steady_density(sim_steps):
    keys = sorted(sim_steps.keys(), key=int)
    last_step = keys[-1]

    x = []
    rho = []
    M_exact = exact_M(sim_steps)
    rho_exact = exact_density(M_exact)

    for node in sim_steps[last_step]["nodes"]:
        x.append(float(node["x"]))
        rho.append(float(node["rho"]))

    fig, ax = plt.subplots(figsize=(8,6), dpi=400)
    ax.plot(x, rho_exact, label="Analytic steady flow")
    ax.plot(x, rho, "o", label="Simulation")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.1)

    plt.ylabel(r"Nondimensional density $\frac{\rho}{\rho_0}$", fontsize=14)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=14)

    fig.set_tight_layout(True)
    plt.legend()
    plt.grid()
    #plt.show()
    plt.savefig("./plots_31_points/steady_densities.png")


def exact_M(steps):
    M = []
    A = 1.0
    keys = sorted(sim_steps.keys(), key=int)
    last_step = keys[-1]

    def M_func(M):
        return (1/M**2)*((5/6)+(1/6)*M**2)**6 - A**2

    for node in steps[last_step]["nodes"]:
        A = node["A"]
        u = node["u"]
        T = node["T"]
        M_o = u / sqrt(T)
        M_i = fsolve(M_func, M_o)[0]
        M.append(M_i)

    return M


def exact_density(M):
    M = np.array(M)
    rho = (1 + 0.2*M**2)**(-2.5)
    return rho


def exact_pressure(M):
    M = np.array(M)
    p = (1 + 0.2*M**2)**(-3.5)
    return p


class UpdateDensity:
    def __init__(self, ax, sim_steps):
        x = []
        for node in sim_steps["0"]["nodes"]:
            x.append(float(node["x"]))
        x = np.array(x)
        self.x = x
        M_exact = exact_M(sim_steps)
        self.exact_line = ax.plot([], [], "b-", label="Exact (Steady State)")[0]
        self.line = ax.plot([], [], "k--", label="Simulation")[0]
        exact_rho = exact_density(M_exact)
        self.exact_line.set_data(self.x, exact_rho)
        self.ax = ax
        self.steps = sim_steps

        # Set up plot parameters
        self.ax.set_ylim(0, 1.5)
        self.ax.set_xlim(0, 3)
        self.ax.grid(True)

    def __call__(self, i):
        i = str(i)
        rho = []
        lines = []
        if i in self.steps:
            for node in sim_steps[i]["nodes"]:
                rho.append(float(node["rho"]))
            self.line.set_data(self.x, rho)
        return self.line,


class UpdateMass:
    def __init__(self, ax, sim_steps):
        x = []
        for node in sim_steps["0"]["nodes"]:
            x.append(float(node["x"]))
        x = np.array(x)
        self.x = x
        m_exact = [1.2**(-2.5)*sqrt(5/6) for i in x]
        self.line = ax.plot([], [], "k--", label="Simulation")[0]
        self.exact_line = ax.plot(x, m_exact, "b-", label="Exact (Steady State)")[0]
        self.ax = ax
        self.steps = sim_steps

        # Set up plot parameters
        self.ax.set_ylim(-1, 2.5)
        self.ax.set_xlim(0, 3)
        self.ax.grid(True)

    def __call__(self, i):
        i = str(i)
        mdot = []
        lines = []
        if i in self.steps:
            for node in sim_steps[i]["nodes"]:
                rho = float(node["rho"])
                u = float(node["u"])
                A = float(node["A"])
                mdot.append(rho*u*A)
            self.line.set_data(self.x, mdot)
        return self.line,


class UpdatePressure:
    def __init__(self, ax, sim_steps):
        self.line = ax.plot([], [], "k-", label="Simulation", linewidth=2)[0]
        x = []
        for node in sim_steps["0"]["nodes"]:
            x.append(float(node["x"]))
        x = np.array(x)
        self.x = x
        M_exact = exact_M(sim_steps)
        self.exact_line = ax.plot([], [], "b-", label="Exact (Steady State)")[0]
        exact_p = exact_pressure(M_exact)
        self.exact_line.set_data(self.x, exact_p)
        self.ax = ax
        self.steps = sim_steps

        # Set up plot parameters
        self.ax.set_ylim(0, 2.5)
        self.ax.set_xlim(0, 3)
        self.ax.grid(True)

    def __call__(self, i):
        i = str(i)
        mdot = []
        if i in self.steps:
            for node in sim_steps[i]["nodes"]:
                rho = float(node["rho"])
                T = float(node["T"])
                mdot.append(rho*T)
            self.line.set_data(self.x, mdot)
        return self.line,


def density_v_x(sim_steps):
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    #fig.set_dpi(300)
    ud = UpdateDensity(ax, sim_steps)
    anim = FuncAnimation(fig, ud, frames=range(0, 550, 2), interval=100, blit=True)
    plt.legend()
    plt.ylabel(r"$\frac{\rho}{\rho_0}$", rotation=0, labelpad=20, fontsize=18)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=12)
    fig.suptitle("Nondimensional density along the nozzle over time")
    plt.show()
    #anim.save("./plots_31_points/rho_v_x_time.mp4")


def mdot_v_x(sim_steps):
    fig, ax = plt.subplots()
    fig.set_size_inches(8,6)
    #fig.set_dpi(300)
    ud = UpdateMass(ax, sim_steps)
    anim = FuncAnimation(fig, ud, frames=range(0, 700, 2), interval=100, blit=True)
    plt.legend()
    plt.ylabel(r"Nondimensional mass flow $\frac{\rho u A}{\rho_0 a_0 A*}$", fontsize=14)
    plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=14)
    fig.suptitle("Nondimensional mass flow along the nozzle over time", fontsize=18)
    plt.show()


def p_v_x(sim_steps):
    fig, ax = plt.subplots()
    ud = UpdatePressure(ax, sim_steps)
    anim = FuncAnimation(fig, ud, frames=range(0, 200, 2), interval=200, blit=True)
    plt.legend()
    #plt.ylabel(r"$\frac{\rho u A}{\rho_0 a_0 A*}$", fontsize=18)
    #plt.xlabel("Nondimensional distance through the nozzle (x)", fontsize=12)
    #fig.suptitle("Nondimensional mass flow along the nozzle over time")
    plt.show()


if __name__ == "__main__":
    # directories containing data for the saved simulation steps
    sim_steps_dirs = [f for f in os.scandir("./output/") if f.is_dir()]

    sim_steps = {}
    nodes = []
    residuals = []

    # CAUTION: On simulations with large amounts of data this may not be the best
    # option. A generator or something that doesn't load all the data should be
    # a better approach
    for step_dir in sim_steps_dirs:
        sim_steps[step_dir.name] = {}
        # Load the node data
        with open(step_dir.path+"/node") as f:
            for line in f:
                nodes.append(json.loads(line))
            sim_steps[step_dir.name]["nodes"] = nodes
            nodes = []

        # Load the time data
        with open(step_dir.path+"/time") as f:
            # the time file should contain only one line
            sim_steps[step_dir.name]["time"] = float(f.read())

        # Load the residuals data
        with open(step_dir.path+"/residual") as f:
            for line in f:
                residuals.append(json.loads(line))
            sim_steps[step_dir.name]["residuals"] = residuals
            residuals = []


    # At this point all the data should be loaded so we can plot it
    density_v_x(sim_steps)


