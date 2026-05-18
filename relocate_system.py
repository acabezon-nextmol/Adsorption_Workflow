#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-05-07 11:24:32

@author: Alfonso Cabezon
@email: alfonso.cabezon@nextmol.com
"""


desc = """
This code unwraps the hair surface to locate the graphene part at the bottom
of the box prior to add walls
"""
usage = """
python3.12 relocate_system.py -i selections.yaml
"""

import argparse
import yaml
import MDAnalysis as mda
from typing import Dict, Any

def create_combined_group(
    universe: mda.Universe, 
    selections: Dict[str, str]
) -> mda.core.groups.AtomGroup:
    """
    Combines multiple selection strings into a single optimized MDAnalysis AtomGroup.
    
    Args:
        universe (mda.Universe): The MDAnalysis Universe object.
        selections (Dict[str, str]): Named selections (e.g., {'GRA4': 'resname GRA4'}).
        
    Returns:
        mda.core.groups.AtomGroup: Atoms matching the combined selections.
    """
    if not selections:
        raise ValueError("Provided selection dictionary is empty.")
        
    combined_sel_string = " or ".join(f"({sel})" for sel in selections.values())
    return universe.select_atoms(combined_sel_string)

def process_system(config_path: str) -> None:
    """
    Reads a YAML configuration, builds dynamic selections, aligns the unbonded 
    layers of the system, and writes the output coordinate file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    """
    # 1. Load and validate the YAML configuration
    with open(config_path, "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    try:
        tpr = config["FILES"]["tpr"]
        gro = config["FILES"]["gro"]
        out = config["FILES"]["out"]
    except KeyError as e:
        raise KeyError(f"Missing required file path in configuration: {e}")

    box_increment = config.get("SETTINGS", {}).get("box_increment", 30)
    z_buffer = box_increment / 2

    # 2. Initialize Universe
    u = mda.Universe(tpr, gro)
    dimensions = u.dimensions
    z_height = dimensions[2]
    all_atoms = u.atoms

    # 3. Build AtomGroups dynamically
    graphene_ag = create_combined_group(u, config.get("GRAPHENE", {}))
    solvent_ag = create_combined_group(u, config.get("SOLVENT", {}))

    if len(graphene_ag) == 0 or len(solvent_ag) == 0:
        raise ValueError("One of the atom groups has 0 atoms.")

    # 4. Apply Geometric Transformations
    # Shift by Z/2 to assemble split components
    all_atoms.translate([0.0, 0.0, z_height / 2.0])

    # Wrap to centralize
    graphene_ag.wrap()
    solvent_ag.wrap(compound="fragments", center = "cog")

    # Shift system to target Z buffer
    shift_z = graphene_ag.positions[:, 2].min() - 1.5 # Ensure grahene is 1.5 angstrom above 0
    all_atoms.translate([0.0, 0.0, -shift_z])

    # Final fragment wrap for solvent safety
    solvent_ag.wrap(compound="fragments", center = "cog")

    # Expand box dimensions for vacuum
    u.dimensions[2] += box_increment
    # relocate system
    all_atoms.translate([0, 0, z_buffer])

    # 5. Write output
    u.atoms.write(out)
        
    print(f"Successfully processed and saved to {out}")

def main():
    parser = argparse.ArgumentParser(description = desc, usage = usage)
    parser.add_argument("-i", "--input", dest = "input", required = True,
                        action = "store", type = str, default = "config.yaml",
                        metavar = f"{"<str>":<10}{".YAML":>15}",
                        help = ".YAML file with the input parameters")
    args = parser.parse_args()

    process_system(args.input)

if __name__ == "__main__":
    main()