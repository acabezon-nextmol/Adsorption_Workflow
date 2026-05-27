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
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")

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
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    with open(config_path, "r") as file:
        config: Dict[str, Any] = yaml.safe_load(file)

    try:
        tpr = config["FILES"]["tpr"]
        gro = config["FILES"]["gro"]
        out = config["FILES"]["out"]
    except KeyError as e:
        raise KeyError(f"Missing required file path in configuration: {e}")

    box_increment = config.get("SETTINGS", {}).get("box_increment", 30)
    z_buffer = box_increment / 2.0
    target_z_bottom = 1.5

    # 2. Initialize Universe
    logging.info(f"Loading Universe from {tpr} and {gro}.")
    u = mda.Universe(tpr, gro)
    dimensions = u.dimensions
    z_height = dimensions[2]
    all_atoms = u.atoms

    # 3. Build AtomGroups dynamically
    logging.info("Building AtomgGroups from selections.")
    graphene_ag = create_combined_group(u, config.get("GRAPHENE", {}))
    solv_dict = config.get("SOLVENT", {})
    polymer_sel = solv_dict.get("polymer", "")
    # Isolate W and ION
    water_ion_dict = { k : v for k,v in solv_dict.items() if k != "polymer"}
    polymer_ag = u.select_atoms(polymer_sel)
    solvent_ag = create_combined_group(u, water_ion_dict)
    # Sanity check
    if len(graphene_ag) == 0 or len(solvent_ag) == 0 or len(polymer_ag) == 0:
        raise ValueError("One of the atom groups has 0 atoms.")
    
    # 4. Apply Geometric Transformations
    # unwrap fragments in polymer selection
    logging.info("Unwrapping broken polymer chains.")
    polymer_ag.unwrap(compound = "fragments", reference = "cog")
    # NOTE: Water and IONS are single beads that do not need unwrapping
    
    # Shift by Z/2 to assemble split components
    all_atoms.translate([0.0, 0.0, z_height / 2.0])

    # Wrap to centralize graphene
    graphene_ag.wrap()
    # Position graphene at the bottom
    current_graphene_min = graphene_ag.positions[:, 2].min()
    z_shift_to_bottom = target_z_bottom - current_graphene_min

    logging.info("Shifting system to locate graphene at the bottom")
    all_atoms.translate([0.0, 0.0, z_shift_to_bottom])

    # 5. Deterministic solvent wrapping
    wi_positions = solvent_ag.positions
    # Mask beads below target Z
    below_graphene_mask = wi_positions[:, 2] < target_z_bottom
    # Add a full box height to the masked beads
    wi_positions[below_graphene_mask, 2] += z_height
    solvent_ag.positions = wi_positions
    logging.info(f"Relocated {below_graphene_mask.sum()} water/ion beads.")

    # 6. Relocate polymer chains
    logging.info("Relocating polymer chains.")
    for frag in polymer_ag.fragments:
        if frag.center_of_geometry()[2] < target_z_bottom:
            frag.translate([0.0, 0.0, z_height])

    # 7. Expand box dimensions and centralize
    logging.info("Adding buffer in Z to use walls.")
    u.dimensions[2] += box_increment
    # relocate system
    all_atoms.translate([0, 0, z_buffer - target_z_bottom])
    max_Z_coordinate = all_atoms.positions[:, 2].max()
    u.dimensions[2] = max_Z_coordinate + (box_increment / 2)

    # 5. Write output
    logging.info(f"Writing output to {out}")
    u.atoms.write(out)

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
