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
    spacing = number_of_atoms / n_side # this tells me how spaced apart each atom should be from each other

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


