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
import os
import sys

# =============================================================================
# Variables
# =============================================================================

z_mix = 195
z_sol = 24
w_mass = 72
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

def write_dummy_mdp(file_name="dummy.mdp"):
    """Writes a dummy .mdp file to generate .tprs

    Parameters
    ----------
    file_name : str, optional
        _description_, by default "dummy.mdp"
    """    
    mdp_content = """integrator               = steep
nsteps                   = 100

nstxout                  = 0
nstvout                  = 0
nstfout                  = 0

cutoff-scheme            = Verlet
nstlist                  = 20
nsttcouple               = 20
nstpcouple               = 20
rlist                    = 1.35
verlet-buffer-tolerance  = -1
ns_type                  = grid
pbc                      = xyz

coulombtype              = reaction-field
rcoulomb                 = 1.1
epsilon_r                = 15\t; 2.5 (with polarizable water)
epsilon_rf               = 0
vdw_type                 = cutoff
vdw-modifier             = Potential-shift-verlet
rvdw                     = 1.1

constraints              = none
constraint_algorithm     = Lincs
lincs_order              = 8
lincs_warnangle          = 90
lincs_iter               = 2
"""
    with open(file_name, "w") as f:
        f.write(mdp_content)
    
def write_surface_top(surface_itp : str, file_name : str = "surface.top"):
    """Writes a .top file for the surface 

    Parameters
    ----------
    surface_itp : str
        _description_
    file_name : str, optional
        _description_, by default "surface.top"
    """    
    top_content=f"""#include "martini_v3.0.0.itp"
#include "{surface_itp}"

[ system ]
; name
CG surface

[ molecules ]
; name         number
  CG_surface   1
"""
    with open(file_name, "w+") as top:
        top.write(top_content)

def write_system_top(
        surface_itp : str, polymer_itp : str, n_polymer : int,
        n_water : int, name_ions : str, n_ions : int,
        file_name : str = "system.top"
):
    """Writes a .top file for the surface 

    Parameters
    ----------
    surface_itp : str
        _description_
    file_name : str, optional
        _description_, by default "surface.top"
    """    
    top_content=f"""#include "martini_v3.0.0.itp"
#include "{surface_itp}"
#include "{polymer_itp}"
#include "martini_v3.0.0_solvents_v1.itp"
#include "martini_v3.0.0_ions_v1.itp"


[ system ]
; name
CG Adsorption

[ molecules ]
; name         number
  CG_surface   1
  CG_POL       {n_polymer}
  W            {n_water}
  {name_ions}           {n_ions}
"""
    with open(file_name, "w+") as top:
        top.write(top_content)

def build_system(surface, polymer_gro, polymer_mass, polymer_charge, x, y, water_gro, gmx_bin):
    # Create the mixture of solvent + polymer
    cmd = [
        gmx_bin, "solvate",
        "-cs", water_gro,
        "-box", str(x/10), str(y/10), str(z_mix/10) ,
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
        "-nmol", str(polymer_number),
        "-radius", "0.21",
        "-replace", "W",
        "-o", "tmp_2.gro"
    ]
    mixture = run_gmx(cmd)
    u_mix = mda.Universe("tmp_2.gro")
    # Create the solvent only buffer
    cmd = [
        gmx_bin, "solvate",
        "-cs", water_gro,
        "-box", str(x/10), str(y/10), str(z_sol/10) ,
        "-o", "tmp_3.gro"
    ]
    buffer = run_gmx(cmd)
    u_buffer = mda.Universe("tmp_3.gro")

    # Combine buffer and mixture and add counter ions
    buffer_max_z = np.max(u_buffer.atoms.positions[:, -1])
    u_mix.atoms.positions += np.array([0.0, 0.0, buffer_max_z + 2]) # displace mixture with a safety buffer
    z_dim = buffer_max_z + u_mix.dimensions[2] + 2# Box dim in Z for merged universe
    buffer_mix = mda.Merge(u_buffer.atoms, u_mix.atoms) # Merge system
    buffer_mix.dimensions = np.array([x, y, z_dim, 90.0, 90.0, 90.0])
    buffer_mix.atoms.write("mix_no_ions.gro")
    
    surface_charge = np.sum(surface.atoms.charges)
    system_charge = surface_charge + polymer_tot_charge
    system_charge = int(system_charge)

    if system_charge < 0:
        ion_resname = "ION"
        ion_name = "NA"
        charged = True
    elif system_charge > 0:
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
            atom.residue.resname = ion_resname
            atom.name = ion_name
    
    z_dim = buffer_mix.dimensions[2]
    # Separate systems and merge to reorder and put polymer, then water, and then ions
    polymers = buffer_mix.select_atoms("not resname W ION")
    waters = buffer_mix.select_atoms("resname W") 
    ions = buffer_mix.select_atoms("resname ION")
    ordered_u = mda.Merge(polymers, waters, ions)
    ordered_u.dimensions = np.array([x, y, z_dim, 90.0, 90.0, 90.0])
    ordered_u.atoms.write("tmp_4.gro")


    # Mix three components
    z_surface = surface.dimensions[2]
    ordered_u.atoms.positions += np.array([0.0, 0.0, z_surface + 2]) # Displace mix and add safety buffer

    system = mda.Merge(surface.atoms, ordered_u.atoms) # Merge system
    system.dimensions = np.array([x, y, z_dim + z_surface + 2, 90.0, 90.0, 90.0])
    system.atoms.write("final_system.gro")
    os.system("rm \#* tmp* mix*")
    return system, polymer_number



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
    parser.add_argument("-r", "--restart", dest = "restart", required = False,
                        action = "store_true",
                        help = "ONLY IF surface.tpr EXIST")
    args = parser.parse_args()

    # Set environment variables
    os.environ["GMXLIB"] = "/home/acabezon-nextmol/forcefields/martini_v300"
    # Assign inputs to variables
    surface_gro = args.surface_gro
    surface_top = args.surface_top
    polymer_gro = args.polymer_gro
    polymer_top = args.polymer_top
    water_gro = args.water_gro
    gmx_bin = args.gmx_bin

    if not args.restart:
        write_dummy_mdp()
        write_surface_top(surface_itp = surface_top)
        cmd = [
            gmx_bin, "grompp",
            "-p", "surface.top",
            "-c", surface_gro,
            "-f", "dummy.mdp",
            "-o", "surface"
        ]
        run_gmx(cmd)
    # Read surface and get simensions
    surface = mda.Universe("surface.tpr", surface_gro)
    # surface = mda.Universe(surface_top, surface_gro, topology_format = "ITP")
    dimensions = surface.dimensions
    x, y = dimensions[0], dimensions[1]

    # Read polymer and get mass
    polymer = mda.Universe(polymer_top, polymer_gro, topology_format = "ITP")
    polymer_mass = np.sum(polymer.atoms.masses)
    polymer_charge = np.sum(polymer.atoms.charges)

    # Call build
    system, n_polymers = build_system(surface, polymer_gro, polymer_mass, polymer_charge, x, y, water_gro, gmx_bin)
    n_water = len(system.select_atoms("resname W"))
    ions = system.select_atoms("resname ION")
    n_ions = len(ions)
    name_ions = ions.atoms.names[0]
    write_system_top(
        surface_itp = surface_top,
        polymer_itp = polymer_top,
        n_polymer = n_polymers,
        n_water = n_water,
        name_ions = name_ions,
        n_ions = n_ions
    )

    
# =============================================================================
# Execute
# =============================================================================
if __name__ == "__main__":
    main()