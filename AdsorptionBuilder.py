#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-03-10 15:05:09

@author: Alfonso Cabezon
@email: alfonso.cabezon@nextmol.com
"""


desc = """
This code builds the system to run adsorption simulations at CG level.
"""
usage = """
python3.12 AdsorptoinBuilder.py -s_gro surface.gro -s_top surface.itp -p_gro polymer.gro -np 8 \
-p_top polymer.itp -w_gro water.gro -gmx_bin PATH/gmx
"""

# =============================================================================
# Imports
# =============================================================================

import MDAnalysis as mda
import subprocess
import argparse
import numpy as np

# =============================================================================
# Variables
# =============================================================================

z_mix = 195
z_sol = 24
w_mass = 76
polymer_fraction = 0.09

# =============================================================================
# Functions
# =============================================================================

def run_gmx(cmd : list) -> str:
    """Run a GROMACS command using the shell with error handling

    Parameters
    ----------
    cmd : list
        list where each item is a part of the command

    Returns
    -------
    str
        The output of the command

    Raises
    ------
    RuntimeError
        raises if the command retuens a code different from 0 (failed command)
    """    
    print("Running: ", " ".join(cmd))
    result = subprocess.run(
        cmd, capture_output = True, text = True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"GROMACS command failed: {' '.join(cmd)}")
    
    return result.stdout

def build_system(surface, polymer_gro, polymer_mass, polymer_charge, x, y, water_gro, gmx_bin):
    # Create the mixture of solvent + polymer
    cmd = [
        gmx_bin, "solvate",
        "-cs", water_gro,
        "-box", str(x), str(y), str(int(z_mix/10)) ,
        "-o", "tmp_1.gro"
    ]
    solvate = run_gmx(cmd)
    u = mda.Universe("tmp_1.gro")
    water_tot_mass = len(u.atoms) * w_mass
    polymer_number = int(np.ceil(water_tot_mass * polymer_fraction / polymer_mass))
    polymer_tot_charge = polymer_number * polymer_charge
    cmd = [ 
        gmx_bin, "insert-molecules",
        "-f", "tmp_1.gro",
        "-ci", polymer_gro,
        "-nmol", polymer_number,
        "-radius", "0.21",
        "-o", "tmp_2.gro"
    ]
    mixture = run_gmx(cmd)
    u_mix = mda.Universe("tmp_2.gro")
    # Create the solvent only buffer
    cmd = [
        gmx_bin, "solvate",
        "-cs", water_gro,
        "-box", str(x), str(y), str(int(z_sol/10)) ,
        "-o", "tmp_3.gro"
    ]
    buffer = run_gmx(cmd)
    u_buffer = mda.Universe("tmp_3.gro")

    # Combine buffer and mixture and add counter ions
    buffer_max_z = np.max(u_buffer.atoms.positions[:, -1])
    u_mix.atoms.positions = np.array([0.0, 0.0, buffer_max_z]) # displace mixture
    buffer_mix = mda.Merge(u_buffer.atoms, u_mix.atoms) # Merge system
    
    surface_charge = np.sum(surface.atoms.charges)
    system_charge = surface_charge + polymer_charge

    if system_charge > 0:
        ion_resname = "ION"
        ion_name = "NA"
        charged = True
    elif system_charge < 0:
        ion_resname = "ION"
        ion_name = "CL"
        charged = True
    else:
        charged = False
    
    if charged:
        waters = buffer_mix.select_atoms("resname W")
        replace_index = np.random.choice(waters.indices, size = abs(system_charge), replace = False)
        for idx in replace_index:
            atom = buffer_mix.atoms[idx]
            atom.resname = ion_resname
            atom.name = ion_name
    
    z_dim = np.max(buffer_mix.atoms.positions[:, -1])
    # Separate systems and merge to reorder and put polymer, then water, and the ions
    polymers = buffer_mix.select_atoms("not resname W ION")
    waters = buffer_mix.select_atoms("resname W") 
    ions = buffer_mix.select_atoms("resname ION")
    ordered_u = mda.Merge(polymers, waters, ions)
    ordered_u.dimensions = np.array([x, y, z_dim, 90.0, 90.0, 90.0])


    # Mix three components
    z_surface = surface.dimensions[2]
    ordered_u.atoms.positions += np.array([0.0, 0.0, z_surface]) # Displace mix

    system = mda.Merge(surface.atoms, ordered_u.atoms) # Merge system
    system.dimensions = np.array([x, y, z_dim + z_surface, 90.0, 90.0, 90.0])
    system.atoms.write("final_system")



# =============================================================================
# Main function
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description = desc, usage = usage)
    parser.add_argument("-s_gro", "--surface_gro", dest = "surface_gro", required = True,
                        action = "store", type = str, 
                        metavar = f"{"<str>":<10}{".GRO/.PDB":>15}",
                        help = "3D coordinate file of the surface to simulate.")
    parser.add_argument("-s_top", "--surface_top", dest = "surface_top", required = True,
                        action = "store", type = str, 
                        metavar = f"{"<str>":<10}{".ITP/.TOP":>15}",
                        help = "The topology file of the surface.")
    parser.add_argument("-p_gro", "--polymer_gro", dest = "polymer_gro", required = True,
                        action = "store", type = str,
                        metavar = f"{"<str>":<10}{".GRO/.PDB":>15}",
                        help = "Coordinate file of the polymer.")
    parser.add_argument("-p_top", "--polymer_top", dest = "polymer_top", required = True,
                        action = "store", type = str, 
                        metavar = f"{"<str>":<10}{".ITP/.TOP":>15}",
                        help = "Polymer topology file.")
    parser.add_argument("-w_gro", "--water_gro", dest = "water_gro", required = True,
                        action = "store", type = str,
                        metavar = f"{"<str>":<10}{".GRO/.PDB":>15}",
                        help = "Coordinate file of preequilibrated water box.")
    parser.add_argument("-gmx_bin", "--gmx_bin", dest = "gmx_bin", required = True,
                        action = "store", type = str, 
                        metavar = f"{"<str>":<10}{"PATH":>15}",
                        help = "PATH to the gmx executable")
    args = parser.parse_args()

    # Assign inputs to variables
    surface_gro = args.surface_gro
    surface_top = args.surface_top
    polymer_gro = args.polymer_gro
    polymer_top = args.polymer_top
    water_gro = args.water_gro
    gmx_bin = args.gmx_bin

    # Read surface and get simensions
    surface = mda.Universe(surface_top, surface_gro, topology_format = "ITP")
    dimensions = surface.dimensions
    x, y = dimensions[0], dimensions[1]

    # Read polymer and get mass
    polymer = mda.Universe(polymer_top, polymer_gro, topology_format = "ITP")
    polymer_mass = np.sum(polymer.atoms.masses)
    polymer_charge = np.sum(polymer.atoms.charges)

    # Call build
    build_system(surface, polymer_gro, polymer_mass, polymer_charge, x, y, water_gro, gmx_bin)

    
# =============================================================================
# Execute
# =============================================================================
if __name__ == "__main__":
    main()