"""
Molecular Dynamics of the Lennard-Jones fluid - Week 2
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from Part1 import init_positions, init_velocities, lj_force_potential, velocity_verlet_step


# Adds the kinetic energy of every atom in the lattice.
def kinetic_energy(velocities):
    mass = 1.0
    return 0.5 * mass * np.sum(velocities ** 2)


"""
T = 2K / (3N), where K = 0.5 Σ m v² is the total kinetic energy.
"""
def temperature(velocities):
    number_of_atoms = len(velocities)
    return 2.0 * kinetic_energy(velocities) / (3.0 * number_of_atoms)


"""
So, the temperature constantly changes while the code runs.
So in this new calculation, it readjusts the value of the velocity. so we maintain the same temperature. 
The temperature was meant to be 0.722 for argon. 
"""
def rescale_velocities(velocities, target_temperature):
    scale_factor = np.sqrt(target_temperature / temperature(velocities))
    return velocities * scale_factor


"""
Counts how many pairs of atoms have a separation that falls in each bin.
This is because the formula in the paper states r and change in r. 
The change in r is given as 0.2 in the paper.
The histogram counts the number of atoms between a particular range for every change in r.
"""
def gr_pair_counts(positions, box_length, bin_edges):
    number_of_atoms = len(positions)

    separation_vectors = positions[:, None, :] - positions[None, :, :]
    separation_vectors -= box_length * np.round(separation_vectors / box_length)
    distances = np.sqrt(np.sum(separation_vectors ** 2, axis=2))

    i_index, j_index = np.triu_indices(number_of_atoms, k=1)   # all pairs with i < j
    pair_counts, _ = np.histogram(distances[i_index, j_index], bins=bin_edges)
    return pair_counts


"""
There are some ranges where there are significantly more atoms.
So, this helps normalise that.

Here, we are coding out the first part of the formula given.
"""
def normalize_gr(total_pair_counts, number_of_samples, number_of_atoms, box_length, bin_edges):
    density = number_of_atoms / box_length ** 3
    shell_volumes = (4.0 / 3.0) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    average_pair_counts = total_pair_counts / number_of_samples
    g_of_r = 2.0 * average_pair_counts / (number_of_atoms * density * shell_volumes)
    return bin_centers, g_of_r


"""
Start from the same place every time. Take a fresh copy of the equilibrated system, so each dt gets a fair, identical start.
Work out the number of steps needed to be taken in the algorithm. 
Run it and write down the energy after every jump. That's the inner loop. Each pass moves the atoms one step, then stores kinetic + potential, divided by N. 
You end up with a list of 2000 numbers.
Summarise that list in two numbers.
np.std . Here it's only the fifth decimal place, so the answer is about 0.00002.
np.polyfit()[0] asks whether the list is trending up or down overall, by fitting a straight line and reading off its slope. Close to zero means no trend.
Print one row of the table with those two numbers.
"""
def energy_drift_test(positions, velocities, box_length, time_steps, total_time):
    number_of_atoms = len(positions)
    results = []


    for dt in time_steps:
        number_of_steps = int(round(total_time / dt))
        test_positions = positions.copy()
        test_velocities = velocities.copy()
        forces, potential_energy = lj_force_potential(test_positions, box_length)

        times = np.zeros(number_of_steps)
        total_energies = np.zeros(number_of_steps)
        for step in range(number_of_steps):
            test_positions, test_velocities, forces, potential_energy = velocity_verlet_step(
                test_positions, test_velocities, forces, box_length, dt)
            times[step] = (step + 1) * dt
            total_energies[step] = (kinetic_energy(test_velocities) + potential_energy) / number_of_atoms

        fluctuation = np.std(total_energies)
        drift = np.polyfit(times, total_energies, 1)[0]
        results.append((dt, times, total_energies))

        print(dt, number_of_steps, fluctuation, drift)

    return results


"""
Puts everything together.
1. Place the atoms on the lattice and give them Maxwell-Boltzmann velocities at T = 0.722.
2. Equilibrate: run velocity verlet and rescale the velocities every few steps so the temperature stays at 0.722.
3. Energy drift test: check which dt keeps the total energy constant.
4. Production: plain NVE (no more rescaling). Collect g(r) histograms, energies and the total momentum.
5. Normalise g(r), print the sanity checks and save the plots.
"""

def main():
    np.random.seed(0)

    number_of_atoms = 512
    target_temperature = 0.722
    dt = 0.002

    equilibration_steps = 1000
    rescale_every = 10
    production_steps = 2000
    sample_every = 10

    drift_time_steps = [0.001, 0.002, 0.004]
    drift_total_time = 0.4

    dr = 0.02

    positions, box_length = init_positions(number_of_atoms)
    velocities = init_velocities(number_of_atoms, target_temperature)
    velocities = rescale_velocities(velocities, target_temperature)
    forces, potential_energy = lj_force_potential(positions, box_length)

    print("N =", number_of_atoms, " L =", box_length, " rho =", number_of_atoms / box_length ** 3)
    print("Initial T =", temperature(velocities), " PE/N =", potential_energy / number_of_atoms)

    start_time = time.time()
    for step in range(1, equilibration_steps + 1):
        positions, velocities, forces, potential_energy = velocity_verlet_step(
            positions, velocities, forces, box_length, dt)
        if step % rescale_every == 0:
            velocities = rescale_velocities(velocities, target_temperature)
        if step % 100 == 0:
            print("equilibration step", step, " T =", temperature(velocities),
                  " PE/N =", potential_energy / number_of_atoms)
    print("Equilibration took", time.time() - start_time, "s")

    print("\ndt  steps  fluctuation  drift")
    drift_results = energy_drift_test(positions, velocities, box_length, drift_time_steps, drift_total_time)

    plt.figure()
    for test_dt, times, total_energies in drift_results:
        plt.plot(times, total_energies, label="dt = " + str(test_dt))
    plt.xlabel("t*")
    plt.ylabel("E / N")
    plt.legend()
    plt.title("Energy conservation")
    plt.savefig("energy_drift.png", dpi=150)
    plt.close()

    bin_edges = np.arange(0.0, box_length / 2 + dr, dr)
    total_pair_counts = np.zeros(len(bin_edges) - 1)
    number_of_samples = 0

    times = np.zeros(production_steps)
    kinetic_energies = np.zeros(production_steps)
    potential_energies = np.zeros(production_steps)
    temperatures = np.zeros(production_steps)
    momenta = np.zeros(production_steps)

    start_time = time.time()
    for step in range(production_steps):
        positions, velocities, forces, potential_energy = velocity_verlet_step(
            positions, velocities, forces, box_length, dt)

        times[step] = (step + 1) * dt
        kinetic_energies[step] = kinetic_energy(velocities)
        potential_energies[step] = potential_energy
        temperatures[step] = temperature(velocities)
        momenta[step] = np.linalg.norm(np.sum(velocities, axis=0))

        if (step + 1) % sample_every == 0:
            total_pair_counts += gr_pair_counts(positions, box_length, bin_edges)
            number_of_samples += 1

        if (step + 1) % 100 == 0:
            print("production step", step + 1, " T =", temperatures[step],
                  " E/N =", (kinetic_energies[step] + potential_energies[step]) / number_of_atoms)
    print("Production took", time.time() - start_time, "s")

    bin_centers, g_of_r = normalize_gr(total_pair_counts, number_of_samples, number_of_atoms, box_length, bin_edges)
    total_energies = kinetic_energies + potential_energies

    print("\nKE/N =", np.mean(kinetic_energies) / number_of_atoms)
    print("PE/N =", np.mean(potential_energies) / number_of_atoms)
    print("E/N  =", np.mean(total_energies) / number_of_atoms, "+/-", np.std(total_energies) / number_of_atoms)
    print("T    =", np.mean(temperatures))
    print("Equipartition: K =", np.mean(kinetic_energies), " 3/2 N <T> =", 1.5 * number_of_atoms * np.mean(temperatures))
    print("Max |total momentum| =", np.max(momenta))
    peak = np.argmax(g_of_r)
    print("First g(r) peak at r =", bin_centers[peak], " height =", g_of_r[peak])
    print("g(r) average over the last 10 bins =", np.mean(g_of_r[-10:]))

    plt.figure()
    plt.plot(bin_centers, g_of_r)
    plt.axhline(1.0, color="gray", linestyle="--")
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.title("Radial distribution function")
    plt.savefig("gr.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(times, kinetic_energies / number_of_atoms, label="KE / N")
    plt.plot(times, potential_energies / number_of_atoms, label="PE / N")
    plt.plot(times, total_energies / number_of_atoms, label="E / N")
    plt.xlabel("t")
    plt.ylabel("energy per atom")
    plt.legend()
    plt.title("Production energies (NVE)")
    plt.savefig("production_energies.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(times, temperatures)
    plt.axhline(target_temperature, color="gray", linestyle="--")
    plt.xlabel("t")
    plt.ylabel("T")
    plt.title("Temperature (NVE)")
    plt.savefig("temperature.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
