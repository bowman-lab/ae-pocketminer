import argparse
import warnings

import numpy as np
from Bio.PDB import PDBParser, PDBIO

# warnings.filterwarnings("ignore")

"""
This script annotates a protein structure with predicted per-residue pocket
probabilities by assigning the prediction values to the B-factor field of
each atom in a PDB file. The resulting PDB file can be visualized using
PyMOL or Chimera to highlight regions of interest, such as predicted cryptic
pockets.
"""

# Mapping 3-letter residue codes to 1-letter codes
residue_types = {
        "ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" ,
        "CYM" : "C" , "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" ,
        "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" ,
        "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" ,
        "VAL" : "V"}


def make_bfactor_pdb(path_to_pdb, pred_file_path, output_fname):
    """Assigns per-residue predicted probabilities as B-factors and writes
    out a new PDB file.

    path_to_pdb : path to the input PDB structure
    pred_file_path : path to the {pdb_name}-preds.npy file from
        xtal_predict.py, shape (1, num_residues)
    output_fname : path to write the B-factor-annotated PDB to
    """
    p = PDBParser(QUIET=True)
    structure = p.get_structure("structure", path_to_pdb)

    # get only the first chain as that's the one where we have predictions
    for chain in structure.get_chains():
        break

    # Load predicted probabilities
    predictions = np.load(pred_file_path)

    # Assign predicted values as B-factors to atoms
    num_res = 0
    for res in chain.get_residues():
        if res.get_resname() in residue_types:
            for atom in res.get_atoms():
                atom.set_bfactor(predictions[0, num_res])
            num_res = num_res + 1

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_fname)
    print(f'Wrote B-factor-annotated PDB to {output_fname}', flush=True)


if __name__ == '__main__':
    # These defaults reproduce the original hardcoded values; override any
    # of them from the command line instead of editing this file, e.g.:
    #   python src/make_pdb.py --pdb-name 9m1h-chainA-rmHOH-rmPGE2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', default='aepocketminer',
                         help='subfolder of results/ that predictions were written to '
                              '(must match the output_directory used in xtal_predict.py)')
    parser.add_argument('--pdb-name', default='XXXX',
                         help='basename (no extension) of the PDB in inputs/ and of the '
                              '*-preds.npy file in results/{model_name}/')
    parser.add_argument('--input-dir', default='inputs',
                         help='directory containing the source PDB file')
    parser.add_argument('--results-dir', default=None,
                         help='directory containing {pdb_name}-preds.npy; '
                              'defaults to results/{model_name}/')
    parser.add_argument('--output-pdb', default=None,
                         help='path to write the output PDB to; defaults to '
                              'results/{model_name}/{pdb_name}_B-factor.pdb')
    args = parser.parse_args()

    results_dir = args.results_dir or f'results/{args.model_name}'

    path_to_pdb = f'{args.input_dir}/{args.pdb_name}.pdb'
    pred_file_path = f'{results_dir}/{args.pdb_name}-preds.npy'
    output_fname = args.output_pdb or f'{results_dir}/{args.pdb_name}-Bfactor.pdb'

    make_bfactor_pdb(path_to_pdb, pred_file_path, output_fname)
