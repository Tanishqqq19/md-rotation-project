"""
Molecular Dynamics of the Lennard-Jones 
"""

import numpy as np


"""
This function will  take the number of atoms and place them in a lattice. The atoms are evenly spaced in there. 

This function will return the box_length and the xyz coordinates of each atom
"""
def init_positions(number_of_atoms):
    RHO = 0.8442 # this is given in the pdf

    BOX_LENGTH = (number_of_atoms / RHO) ** (1 / 3) # This is to find the length of the cube(space) where all the atoms would fit.

    n_side = int(np.ceil(number_of_atoms ** (1.0 / 3.0))) # this tells you the amount of atoms in each side of the cube.
    spacing = BOX_LENGTH / n_side # this tells me how spaced apart each atom should be from each other

    positions = []
    for x in range(n_side):
        for y in range(n_side):
            for z in range(n_side):
                if len(positions) < number_of_atoms:
                    pos = np.array([x, y, z]) * spacing - BOX_LENGTH / 2 + spacing / 2
                    positions.append(pos)

    return np.array(positions[:number_of_atoms]), BOX_LENGTH


def init_velocities(n, T):
    velocities = np.random.normal(0.0, np.sqrt(T), size=(n, 3))
    velocities -= velocities.mean(axis=0)
    return velocities


"""
Periodic boundary conditions
"""
def periodic_boundary_conditions(atom_x_coordinates, atom_y_coordinates, box_length):
    normal_length = abs(atom_y_coordinates - atom_x_coordinates)
    periodic_length = box_length - abs(atom_y_coordinates - atom_x_coordinates)
    return min(normal_length, periodic_length)

"""
Wraps particle positions back into the simulation box
Since the length of the box is from -L/2 to L/2
this function places that particular atom coordinate into this adjusted box
"""
def wrap_positions(positions, box_length):
    return positions - box_length * np.round(positions / box_length)


"""
Computes Lennard-Jones forces and potential energy for every pair of
atoms, using a cutoff at rc.
"""

def lj_force_potential(positions, box_length):
    cutoff_radius = 2.5
    epsilon = 1.0
    sigma = 1.0

    number_of_atoms = len(positions)
    forces = np.zeros_like(positions)
    total_potential_energy = 0.0

    sigma_over_cutoff_6 = (sigma / cutoff_radius) ** 6
    sigma_over_cutoff_12 = sigma_over_cutoff_6 ** 2
    potential_energy_at_cutoff = 4.0 * epsilon * (sigma_over_cutoff_12 - sigma_over_cutoff_6)
    force_magnitude_at_cutoff = 24.0 * epsilon / cutoff_radius * (2.0 * sigma_over_cutoff_12 - sigma_over_cutoff_6)

    for i in range(number_of_atoms - 1):
        for j in range(i + 1, number_of_atoms):
            raw_separation_vector = positions[i] - positions[j]
            separation_vector = raw_separation_vector - box_length * np.round(raw_separation_vector / box_length)
            distance = np.sqrt(np.dot(separation_vector, separation_vector))

            if distance < cutoff_radius:
                sigma_over_distance_6 = (sigma / distance) ** 6
                sigma_over_distance_12 = sigma_over_distance_6 ** 2

                potential_energy = 4.0 * epsilon * (sigma_over_distance_12 - sigma_over_distance_6)
                force_magnitude = 24.0 * epsilon / distance * (2.0 * sigma_over_distance_12 - sigma_over_distance_6)

                shifted_potential_energy = (potential_energy - potential_energy_at_cutoff) + (distance - cutoff_radius) * force_magnitude_at_cutoff
                shifted_force_magnitude = force_magnitude - force_magnitude_at_cutoff

                force_vector = shifted_force_magnitude * (separation_vector / distance)
                forces[i] += force_vector
                forces[j] -= force_vector

                total_potential_energy += shifted_potential_energy

    return forces, total_potential_energy


def velocity_verlet_step(positions, velocities, forces_old, box_length, dt):
    mass = 1.0

    accel_old = forces_old / mass

    velocities_half = velocities + 0.5 * dt * accel_old  # eqn 3.11a
    positions_new = positions + dt * velocities_half      # eqn 3.11b

    positions_new = wrap_positions(positions_new, box_length)

    forces_new, potential_energy = lj_force_potential(positions_new, box_length)  
    accel_new = forces_new / mass

    velocities_new = velocities_half + 0.5 * dt * accel_new  # eqn 3.11c

    return positions_new, velocities_new, forces_new, potential_energy