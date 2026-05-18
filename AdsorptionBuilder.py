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
python3.12 AdsorptionBuilder.py -s_gro surface_CG_75x25.gro -s_top surface_CG_75x25.itp \
	-p_gro CG_CHIT.gro -p_top CG_CHIT.itp -w_gro water_bead.gro -r
"""

# =============================================================================
# Imports
# =============================================================================

import MDAnalysis as mda
import subprocess
import argparse
import numpy as np
import os
import glob
import sys
import datetime
from scipy.constants import Avogadro
from typing import Tuple, List, Dict
import logging

# Initialize logger
logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
	handlers=[
		logging.FileHandler("system_builder.log"),
		logging.StreamHandler(sys.stdout)
	]
)

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
		raises if the command returns a code different from 0 (failed command)
	""" 
	# author = Alfonso Cabezón <alfonso.cabezon@nextmol.com>
	# Created on (DD/MM/YYYY): 10/03/2026
	cmd_str = " ".join(cmd)
	logging.info(f"Executing: {cmd_str}")
	
	# Capture start time as a datetime object
	start_time = datetime.datetime.now()
	
	# Execute the process
	result = subprocess.run(cmd, capture_output=True, text=True)
	
	# Capture end time and calculate timedelta
	end_time = datetime.datetime.now()
	elapsed_time = end_time - start_time
	
	if result.returncode != 0:
		logging.error(f"Command failed:\n{result.stderr}")
		raise RuntimeError(f"GROMACS command failed: {cmd_str}")
	
	# timedelta automatically formats as HH:MM:SS.mmmmmm
	logging.info(f"Completed in {elapsed_time}.")
	
	return result.stdout

def determine_system_composition(lx : float, ly : float, lz : float, polymer_mass : int,
				  water_mass : int = 72, density : float = 0.5,
					polymer_fraction : float = 0.09) -> Tuple[int, int]:
	"""Computes the number of water and polymer chians to be added to fill the box and get the 
	desired density.

	follows formula: N = (density V Na) / water_mass

	Parameters
	----------
	lx : float
		x dimension of simulation box in nm
	ly : float
		y dimension of simulation box
	lz : float
		z dimension of simulation box
	water_mass : int, optional
		mass of a water bead, by default 72
	density : float, optional
		density of martini 3 water, by default 0.5 g / cm3

	Returns
	-------
	int, int
		Number of water beads to fill the box and number of polymer chains
	"""    
	# author = Alfonso Cabezón <alfonso.cabezon@nextmol.com>
	# Created on (DD/MM/YYYY): 13/03/2026
	V = lx * ly * lz
	density = density * 1e-24 # Convert to angstrom3
	target_mass = density * V * Avogadro
	polymer_target_mass = target_mass * polymer_fraction
	P = int(polymer_target_mass / polymer_mass) # number of polymer chains to add
	W = int(target_mass / water_mass) # Water beads to add
	return W, P

def calculate_water_beads(lx : float, ly : float, lz :float, target_density : float = 0.9843,
						  water_mass : int = 72) -> int:
	"""Calculates the number of water beads to insert to a box of specific dimensions
	(in Angstrom) in order to match the target density.

	Parameters
	----------
	lx : float
		Box dimension in X in Angstrom
	ly : float
		Box dimension in Y in Angstrom
	lz : float
		Box dimension in Z in Angstrom
	target_density : float, optional
		A density in g/cm3 to target, by default 0.9843
	water_mass : int, optional
		Mass of water bead, by default 72

	Returns
	-------
	int
		number of water beads for target density
	"""	
	V = lx * ly * lz
	target_density = target_density * 1e-24
	n_water_beads = int(target_density * V * Avogadro / water_mass)
	return n_water_beads


def write_dummy_mdp(file_name="dummy.mdp"):
	"""Writes a dummy .mdp file to generate .tprs

	Parameters
	----------
	file_name : str, optional
		a .mdp file to create .tpr with grompp, by default "dummy.mdp"
	"""
	# author = Alfonso Cabezón <alfonso.cabezon@nextmol.com>
	# Created on (DD/MM/YYYY): 10/03/2026 
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
		.ITP file of the surface
	file_name : str, optional
		A .TOP file including the martini 3 FF, by default "surface.top"
	"""
	# author = Alfonso Cabezón <alfonso.cabezon@nextmol.com>
	# Created on (DD/MM/YYYY): 10/03/2026
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

def create_walls_gro(lx : float, ly : float, lz : float, 
					 file_name : str = "walls.gro", grid_spacing : float = 0.3):
	"""This function creates a .GRO file with fake walls on top and bottom.
	The walls are atoms separated by grid_spacing. This is used to avoid the 
	insertion of polymers that leave the box in Z

	Parameters
	----------
	lx : float
		X dimensions of the box
	ly : float
		Y dimension of the box
	lz : float
		Z dimension of the box
	file_name : str, optional
		Name of the output .GRO, by default "walls.gro"
	grid_spacing : float, optional
		Separation between atoms in the wall, by default 0.3

	Returns
	-------
	mda.core.universe.Universe
		The MDAnalysis universe with the walls
	"""
	# author = Alfonso Cabezón <alfonso.cabezon@nextmol.com>
	# Created on (DD/MM/YYYY): 11/03/2026
	z_bottom = 0.2 # Small buffer to avoid pbcs
	z_top = lz - 0.2
	positions = []

	# Modified 06/05/2026. Aim: Speed up coordinate generation using numpy
	# Vectorized coordinate generation
	x_coords = np.arange(0, lx, grid_spacing)
	y_coords = np.arange(0, ly, grid_spacing)
	xx, yy = np.meshgrid(x_coords, y_coords)
	
	x_flat = xx.flatten()
	y_flat = yy.flatten()
	
	# Generate Z coordinates
	z_bottom_arr = np.full_like(x_flat, z_bottom) # Array with same dimensions as x_flat with z_bottom value
	z_top_arr = np.full_like(x_flat, z_top)
	
	# Combine and stack bottom and top arrays
	bottom_wall = np.column_stack((x_flat, y_flat, z_bottom_arr)) # Put X,Y, and Z together
	top_wall = np.column_stack((x_flat, y_flat, z_top_arr))
	positions = np.vstack((bottom_wall, top_wall))

	n_atoms = len(positions)
	u_walls = mda.Universe.empty(n_atoms,
								 n_residues = n_atoms,
								 atom_resindex = np.arange(n_atoms),
								 trajectory = True)
	u_walls.add_TopologyAttr("name", ["C"] * n_atoms)
	u_walls.add_TopologyAttr("resname", ["WALL"] * n_atoms)
	u_walls.add_TopologyAttr("resid", np.arange(1, n_atoms + 1))
	u_walls.atoms.positions = np.array(positions)
	u_walls.dimensions = np.array([lx, ly, lz, 90.0, 90.0, 90.0])
	u_walls.atoms.write(file_name)
	return u_walls

def write_system_top(
		surface_itp : str, polymer_itp : str, topology_entries : List[Dict[str, str]],
		file_name : str = "system.top"
):
	"""Writes the system.top file including the martini 3 FF and .ITP of the
	different molecules

	Parameters
	----------
	surface_itp : str
		.ITP file of the surface
	polymer_itp : str
		.ITP file of the polymer
	topology_entries : List[Dict[str, str]]
		List with name : count pairs for the molecules directive
	file_name : str, optional
		name for the topology .TOP file, by default "system.top"
	"""	
	# author = Alfonso Cabezón <alfonso.cabezon@nextmol.com>
	# Created on (DD/MM/YYYY): 10/03/2026
	lines = ""
	for entry in topology_entries:
		line = f"  {entry["name"]:<14}{entry["count"]}\n"
		lines += line
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

"""
	with open(file_name, "w+") as top:
		top.write(top_content)
		top.writelines(lines)
		top.close()

def modify_system_top(
		system_top : str, polymer_itp : str, topology_entries : list,
):
	"""This function modifies an existing system topology file. It first parses the current
	file to store the relevant information. Then adds new info from topology_entries
	and writes a nre system topology.

	Parameters
	----------
	system_top : str
		The name of the current topology file
	polymer_itp : str
		The .itp file of the polymer
	topology_entries : list
		new topology entries for the file.
	"""	
	# author = Alfonso Cabezón <alfonso.cabezon@nextmol.com>
	# Created on (DD/MM/YYYY): 28/04/2026
	# =============================================================================
	# Parse current topology
	# =============================================================================
	includes: List[str] = [ ]
	molecules: List[Dict[str, str]] = [ ]
	in_molecules: bool = False
	with open(system_top, "r") as top:
		# Loop over file lines
		for line in top:
			# Clean line
			clean_line = line.strip()
			# Skip empty lines
			if not clean_line or clean_line.startswith(";"):
				continue
			# Store includes
			if clean_line.startswith("#include"):
				includes.append(line)
				continue
			# Detect section headers
			if clean_line.startswith("[") and clean_line.endswith("]"):
				# Normalize header line
				header = clean_line.replace(" ", "")
				if header == "[molecules]":
					in_molecules = True
				else:
					in_molecules = False
				continue
			if in_molecules:
				# Split name and number
				info = clean_line.split()
				if len(info) >= 2:
					molecules.append({
						"name" : info[0],
						"count" : info[1]
					})
		top.close()
	
	# =============================================================================
	# Modify current file
	# =============================================================================
	# update molecules
	molecules += topology_entries
	top_content=f"""#include "{polymer_itp}"
#include "martini_v3.0.0_ions_v1.itp"


[ system ]
; name
CG Adsorption simulation system

[ molecules ]
; name         number
"""
	molecules_lines = [ ]
	# print(molecules)
	for molecule in molecules:
		# print(molecule)
		line = f"  {molecule["name"]:<14}{molecule["count"]}\n"
		molecules_lines.append(line)
	
	with open(system_top, "w+") as top:
		top.writelines(includes)
		top.write(top_content)
		top.writelines(molecules_lines)
		top.close()



def build_system(surface : mda.core.universe.Universe, polymer_gro : str, polymer_mass : int,
				  polymer_charge : int, x : float, y : float, water_gro : str,
					gmx_bin : str, W : int, P : int) -> Tuple[mda.core.universe.Universe, list]:
	"""This function builds the system for simulation. The system contains the 
	hair surface, polymer, water, and ions. The steps followed are listed below:

		1. Calculate the composition of the system if not specified as input.
		2. Add the polymer chains to the simulation box. A gird of atoms is added
		on top and bottom to avoid polymers wrapping through the Z coordinate
		3. Solvate polymer chains.
		4. Create the water only buffer.
		5. Combine buffer and solvated polymer. Add ions to neutralize the charge.
		This step keeps the amount of ions needed to neutralize the surface and the polymer
		which can lead to a mixture of NA and CL
		6. Sort the different components so they follow the order: Polymer, Water, and Ions.
		This is important for the topology generation.
		7. Put everything together in the same .gro.


	Parameters
	----------
	surface : mda.core.universe.Universe
		Universe of the hair surface
	polymer_gro : str
		Polymer .gro file
	polymer_mass : int
		Mass of one unit of polymer chain in uma
	polymer_charge : int
		Charge of one unit of polymer chain
	x : float
		X dimension of the surface box
	y : float
		Y dimension of the surface box
	water_gro : str
		.gro file of the water
	gmx_bin : str
		path og gmx executable
	W : int
		Number of water beads of the system
	P : int
		Number of polymer chains of the system

	Returns
	-------
	mda.core.universe.Universe, list
		Universe of the final system generated, list for the molecules directive of the topology
	"""	
	# author = Alfonso Cabezón <alfonso.cabezon@nextmol.com>
	# Created on (DD/MM/YYYY): 12/03/2026
	# Adapted on (DD/MM/YYYY): 16/03/2026 by Alfonso Cabezón <alfonso.cabezon@nextmol.com>
	# TODO: Document and clean
	# Step 1: Calculate the composition of the system if not specified
	if W is None or P is None:
		W, P = determine_system_composition(x, y, z_mix, polymer_mass)
	# Step 2: Add the polymer chains to the box
	# Modified 06/05/2026. Aim: Reduce workload. Grid spacing was too small generating a very dense grid
	# This made the code slow in consecutive steps. 1 nm spacing is enough with -rot z
	u_walls = create_walls_gro(x, y, z_mix, grid_spacing = 10.0) # Create walls to prevent polymer leakage in Z.
	cmd = [ # Write gmx command
		gmx_bin, "insert-molecules",
		"-f", "walls.gro",
		"-ci", polymer_gro,
		"-nmol", str(P), 
		"-rot", "z",
		"-o", "tmp_2.gro",
		"-try", "20000"
	]
	mixture = run_gmx(cmd) # run gmx command
	pol_box = mda.Universe("tmp_2.gro") # Read generated .gro
	pol_no_walls = pol_box.select_atoms("not resname WALL") # Eliminate Walls
	pol_no_walls.atoms.write("polymers.gro") # Write polymer only .gro

	# Step 3: Solvate polymer chains
	cmd = [
		gmx_bin, "solvate",
		"-cp", "polymers.gro",
		"-cs", water_gro,
		"-maxsol", str(W),
		"-radius", "0.21",
		"-o", "tmp_3.gro"
	]

	run_gmx(cmd)

	# Step 4: Create the solvent only buffer
	n_waters = calculate_water_beads(x, y, z_sol)
	cmd = [
		gmx_bin, "solvate",
		"-cs", water_gro,
		"-box", str(x/10), str(y/10), str(z_sol/10) ,
		"-radius", "0.21",
		"-o", "tmp_4.gro"
	]
	buffer = run_gmx(cmd)
	u_buffer = mda.Universe("tmp_4.gro")
	# Remove W outside in Z
	dimensions = u_buffer.dimensions[:3]
	u_buffer = u_buffer.select_atoms(f"(prop z > 0) and (prop z < {dimensions[-1]})")
	# Step 5: Combine buffer and mixture and add counter ions
	solvated_polymer = mda.Universe("tmp_3.gro") # Load solvated polymer
	# Remove W beads outside the box in Z
	dimensions = solvated_polymer.dimensions[:3]
	solvated_polymer = solvated_polymer.select_atoms(f"(prop z > 0) and (prop z < {dimensions[-1]})")
	# Merge buffer and polymer slab
	buffer_max_z = np.max(u_buffer.atoms.positions[:, -1]) # Get top Z pisition
	solvated_polymer.atoms.positions += np.array([0.0, 0.0, buffer_max_z + 0.2]) # displace mixture with a safety buffer
	z_dim = buffer_max_z + solvated_polymer.dimensions[2] + 0.2 # Box dim in Z for merged universe
	buffer_mix = mda.Merge(u_buffer.atoms, solvated_polymer.atoms) # Merge system
	buffer_mix.dimensions = np.array([x, y, z_dim, 90.0, 90.0, 90.0])
	
	surface_charge = int(np.sum(surface.atoms.charges)) # Calculate surface charge
	# Assign ions depending on charge
	ion_resname = "ION"
	if surface_charge < 0:
		ion_name_surface = "NA"
	elif surface_charge > 0:
		ion_name_surface = "CL"
	# Repeat for polymer
	if polymer_charge < 0:
		ion_name_polymer = "NA"
		charged_polymer = True
	elif polymer_charge > 0:
		ion_name_polymer = "CL"
		charged_polymer = True
	else:
		charged_polymer = False

	# Define overall charge
	if charged_polymer:
		polymer_tot_charge = P * polymer_charge
		system_charge = surface_charge + polymer_tot_charge
		number_of_ions = abs(surface_charge) + abs(polymer_tot_charge)
		system_charge = int(system_charge)
	else:
		system_charge = surface_charge
		system_charge = int(system_charge)
		number_of_ions = abs(system_charge)

	if number_of_ions > 0:
		charged = True
	else:
		charged = False

	# Replace waters by ions
	if charged:
		waters = buffer_mix.select_atoms("resname W")
		replace_index = np.random.choice(waters.indices, size = int(number_of_ions), replace = False)
		shuffled_index = np.random.permutation(replace_index) # Shuffle to add randomness to the split
		replace_surface = shuffled_index[:abs(surface_charge)]
		ions_surface = buffer_mix.atoms[replace_surface]
		replace_polymer = shuffled_index[abs(surface_charge):]
		ions_polymer = buffer_mix.atoms[replace_polymer]
		for idx in replace_surface:
			atom = buffer_mix.atoms[idx]
			atom.residue.resname = ion_resname
			atom.name = ion_name_surface
		
		for idx in replace_polymer:
			atom = buffer_mix.atoms[idx]
			atom.residue.resname = ion_resname
			atom.name = ion_name_polymer
	
	z_dim = buffer_mix.dimensions[2]
	# Step 6: Separate systems and merge to reorder and put polymer, then water, and then ions
	polymers = buffer_mix.select_atoms("not resname W ION")
	waters = buffer_mix.select_atoms("resname W")
	if charged:
		if charged_polymer:
			ordered_u = mda.Merge(polymers, waters, ions_surface, ions_polymer)
		else:
			ordered_u = mda.Merge(polymers, waters, ions_surface)
	else:
		ordered_u = mda.Merge(polymers, waters)
	ordered_u.dimensions = np.array([x, y, z_dim, 90.0, 90.0, 90.0])
	ordered_u.atoms.write("tmp_5.gro")


	# Step 7: Mix all the compinents in one Universe
	z_surface = surface.dimensions[2]
	ordered_u.atoms.positions += np.array([0.0, 0.0, z_surface + 0.2]) # Displace mix and add safety buffer
	surface_waters = len(surface.select_atoms("resname W"))

	system = mda.Merge(surface.atoms, ordered_u.atoms) # Merge system
	system.dimensions = np.array([x, y, z_dim + z_surface + 0.2, 90.0, 90.0, 90.0])

	system.atoms.write("final_system.gro")
	topology_entries = [
		{"name" : "CG_surface", "count" : 1},
		{"name" : "CG_POL", "count" : int(P)},
		{"name" : "W", "count" : len(waters)},
	]
	if charged:
		topology_entries.append({"name" : ion_name_surface, "count" : len(ions_surface)})
		if charged_polymer:
			topology_entries.append({ "name" : ion_name_polymer, "count" : len(ions_polymer)})

	return system, topology_entries



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
	parser.add_argument("-s_tpr", "--surface_tpr", dest = "surface_tpr",
						action = "store", type = str,
						metavar = f"{"<str>":<10}{".TPR":>15}",
						help = ".TPR file of the surface")
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
	parser.add_argument("-top", "--top", dest = "topology",
						action = "store", type = str, default = "system.top",
						metavar = f"{"<str>":<10}{".TOP file":>15}",
						help = "Current topology file for the surface")
	parser.add_argument("-gmx_bin", "--gmx_bin", dest = "gmx_bin", required = False,
						action = "store", type = str, 
						metavar = f"{"<str>":<10}{"PATH":>15}",
						default = "/cvmfs/software.eessi.io/versions/2023.06/software/linux/x86_64/intel/haswell/software/GROMACS/2024.1-foss-2023b/bin/gmx",
						help = "PATH to the gmx executable")
	parser.add_argument("-r", "--restart", dest = "restart", required = False,
						action = "store_true",
						help = "ONLY IF surface.tpr EXIST")
	parser.add_argument("-aa_p", "--aa_polymer", dest = "aa_polymer", required = False,
						action = "store", type = int, default = None,
						metavar = f"{"<int>":<10}{"25":>15}",
						help = "Number of polymer chains in  the atomistic reference")
	parser.add_argument("-aa_w", "--aa_water", dest = "aa_water", required = False,
						action = "store", type = int, default = None,
						metavar = f"{"<int>":<10}{"600000":>15}",
						help = "Number of water molecules in atomistic reference")
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
	system_top = args.topology

	if args.aa_polymer is not None and args.aa_water is not None:
		aa_polymer_chains = args.aa_polymer
		aa_water_molecules = args.aa_water
		water_beads = aa_water_molecules // 4
	elif args.aa_polymer is not None and args.aa_water is None:
		raise ValueError(f"Number of water molecules from AA reference \
				   was not specified for {args.aa_polymer} polymer chains")
	elif args.aa_water is not None and args.aa_polymer is None:
		raise ValueError(f"Number of polymer chains from AA reference \
				   was not specified for {args.aa_water} water molecules")
	else:
		aa_polymer_chains = None
		water_beads = None

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
	else:
		surface = mda.Universe(args.surface_tpr, surface_gro)
	# surface = mda.Universe(surface_top, surface_gro, topology_format = "ITP")
	dimensions = surface.dimensions
	x, y = dimensions[0], dimensions[1]

	# Read polymer and get mass
	polymer = mda.Universe(polymer_top, polymer_gro, topology_format = "ITP")
	polymer_mass = np.sum(polymer.atoms.masses)
	polymer_charge = np.sum(polymer.atoms.charges)

	# Call build
	system, topology_entries = build_system(surface, polymer_gro, polymer_mass,
									polymer_charge, x, y, water_gro, gmx_bin,
									  water_beads, aa_polymer_chains)
	print(topology_entries)
	print("\n")
	
	if system_top is None:
		write_system_top(
			surface_itp = surface_top,
			polymer_itp = polymer_top,
			topology_entries = topology_entries
		)
	else:
		modify_system_top(
			system_top,
			polymer_top,
			topology_entries[1:]
		)

	
# =============================================================================
# Execute
# =============================================================================
if __name__ == "__main__":
	main()
	for f in glob.glob("tmp*gro"):
		os.remove(f)
	os.remove("walls.gro")
	os.remove("polymers.gro")
