#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2026-02-23 14:11:48

@author: Alfonso Cabezon
@email: alfonso.cabezon@nextmol.com
"""


desc = """
Generates an index file from a .yaml defining name-selection pairs
"""
usage = """
python3.12 GMX_ndx.py -i selections.yaml -o index.ndx
"""

# =============================================================================
# Imports
# =============================================================================
import MDAnalysis as mda
import yaml
import argparse

# =============================================================================
# Main function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description = desc, usage = usage)
    parser.add_argument("-i", "--infile", dest = "infile", required = True,
                        action = "store", type = str, metavar = f"  {"(selection.yaml)":>30}",
                        help = ".YAML file containing name-selection pairs")
    parser.add_argument("-o", "--outfile", dest = "outfile", required = False,
                        action = "store", type = str, default = "index.ndx",
                        metavar = f"  {"(index.ndx)":>30}",
                        help = "Name for the gromacs .NDX file")
    parser.add_argument("-tpr", "--tpr", dest = "tpr", required = True,
                        action = "store", type = str, metavar = f"  {"(sim.tpr)":>30}",
                        help = ".TPR file of the system")
    parser.add_argument("-gro", "--gro", dest = "gro", required = True,
                        action = "store", type = str, metavar = f"  {"(sim.gro)":>30}",
                        help = ".GRO file of the system")
    args = parser.parse_args()
    # Assign inputs to vars
    infile = args.infile
    outfile = args.outfile
    u = mda.Universe(args.tpr, args.gro)

    with open("selections.yaml") as f:
        selections = yaml.safe_load(f)

    with mda.selections.gromacs.SelectionWriter(outfile, mode = "w") as ndx:
        for names, sels in selections.items():
            mda_sel = u.select_atoms(sels)
            ndx.write(mda_sel, name = names)
        ndx.close()

if __name__ == "__main__":
    main()

