"""
find_key_attention_residues.py

Given a saved attention_weights.npy from xtal_predict.py (see models.py's
`save_attention_weights` option) and a set of known/apo pocket residue
numbers, this script:

  1. For each pocket residue (as query), averages how much every other
     residue (as key) feeds into its representation -- the "rest of the
     protein -> pocket" direction, i.e. which residues may influence or
     induce opening/closing of the pocket.
  2. Excludes the pocket residues themselves and their immediate sequence
     neighbors (adjacent residues trivially get high attention due to local
     structure), replacing their score with a distinct sentinel value.
  3. Prints the top-N remaining residues by attention score.
  4. Writes out a PDB with these per-residue scores set as the B-factor, so
     the result can be visualized directly in PyMOL/Chimera.

Usage:
    python src/find_key_attention_residues.py \\
        --pdb inputs/9m1h-chainA-rmHOH-rmPGE2.pdb \\
        --attention-weights results/aepocketminer/9m1h-chainA-rmHOH-rmPGE2-attention_weights.npy \\
        --pocket-resnums 10 14 15 \\
        --top-n 40 \\
        --output-pdb results/aepocketminer/9m1h-chainA-rmHOH-rmPGE2-attn_Bfactor.pdb
"""

import argparse

import numpy as np
import mdtraj as md
from Bio.PDB import PDBParser, PDBIO

# Mapping 3-letter residue codes to 1-letter codes 
residue_types = {
        "ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" ,
        "CYM" : "C" , "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" ,
        "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" ,
        "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" ,
        "VAL" : "V"}


def get_backbone_resnums(pdb_path):
    """Reproduces the residue ordering used by process_strucs() in
    validate_performance_on_xtals.py / models.py, so attention-matrix
    indices line up with residue numbers (resSeq) correctly.
    """
    struc = md.load(pdb_path)
    prot_iis = struc.top.select("protein and (name N or name CA or name C or name O)")
    prot_bb = struc.atom_slice(prot_iis)
    res_ids = np.array([r.resSeq for r in prot_bb.top.residues])
    return res_ids


def load_attention_weights(attention_weights_path, n_residues):
    """Loads a saved attention_weights.npy and returns a single (N, N)
    matrix, averaging over heads/batch dim as needed.

    MultiHeadAttention(..., return_attention_scores=True) returns shape
    (batch, num_heads, N, N); xtal_predict.py saves this directly. Handles
    that shape as well as an already-squeezed (heads, N, N) or (N, N) array,
    in case you preprocessed the file yourself before calling this script.
    """
    attn = np.load(attention_weights_path)

    if attn.ndim == 4:
        # (batch, heads, N, N) -- assume batch size 1 (single structure)
        assert attn.shape[0] == 1, (
            f"Expected batch size 1 in attention weights, got shape {attn.shape}. "
            "xtal_predict.py processes one PDB at a time, so this shouldn't happen "
            "unless the file was saved differently.")
        attn = attn[0].mean(axis=0)  # average over heads -> (N, N)
    elif attn.ndim == 3:
        # (heads, N, N)
        attn = attn.mean(axis=0)
    elif attn.ndim != 2:
        raise ValueError(f"Unexpected attention_weights shape {attn.shape}; "
                          "expected 2, 3, or 4 dimensions.")

    if attn.shape[0] != n_residues or attn.shape[1] != n_residues:
        raise ValueError(
            f"Attention matrix shape {attn.shape} does not match the number of "
            f"backbone residues found in the PDB ({n_residues}). Double check "
            "that --pdb and --attention-weights come from the same structure/run.")

    return attn


def adjacent_residue_inds(active_site_residue_inds, sequence_length):
    adjacent = set()
    for res in active_site_residue_inds:
        adjacent.add(res)
        if res - 1 >= 0:
            adjacent.add(res - 1)
        if res + 1 < sequence_length:
            adjacent.add(res + 1)
    return np.array(sorted(adjacent))


def compute_key_residue_scores(attn, res_ids, pocket_resnums):
    """Returns (scores, excluded_inds) where `scores` has one entry per
    residue (average attention paid to it by the chosen pocket residues),
    and `excluded_inds` are the array indices of the pocket residues
    themselves plus their immediate neighbors.
    """
    n_residues = len(res_ids)

    selected_inds = []
    for resnum in pocket_resnums:
        matches = np.where(res_ids == resnum)[0]
        if len(matches) == 0:
            raise ValueError(f"Residue number {resnum} not found among backbone "
                              f"residues in this structure (res_ids range "
                              f"{res_ids.min()}-{res_ids.max()}).")
        selected_inds.append(matches[0])

    rows = [attn[i] for i in selected_inds]
    scores = np.mean(rows, axis=0)

    excluded_inds = adjacent_residue_inds(selected_inds, n_residues)
    return scores, excluded_inds


def print_top_residues(scores, excluded_inds, res_ids, top_n):
    mask = np.ones_like(scores, dtype=bool)
    mask[excluded_inds] = False

    valid_inds = np.arange(len(scores))[mask]
    order = np.argsort(scores[valid_inds])[::-1][:top_n]
    top_inds = valid_inds[order]

    print(f"\nExcluded residues (pocket residues + immediate neighbors): "
          f"{sorted(res_ids[excluded_inds].tolist())}")
    print(f"\nTop {top_n} residues by attention score:")
    for resnum, score in zip(res_ids[top_inds], scores[top_inds]):
        print(f"  residue {resnum:>5d}   score {score:.6f}")

    return res_ids[top_inds], scores[top_inds]


def write_bfactor_pdb(pdb_path, res_ids, scores, excluded_inds, exclude_value, output_pdb):
    """Writes a copy of the PDB with per-residue attention scores set as
    the B-factor. Excluded residues (pocket residues + neighbors) get
    `exclude_value` instead, so they stand out distinctly in PyMOL/Chimera.
    """
    b_factor_by_resnum = dict(zip(res_ids.tolist(), scores.tolist()))
    excluded_resnums = set(res_ids[excluded_inds].tolist())
    for resnum in excluded_resnums:
        b_factor_by_resnum[resnum] = exclude_value

    p = PDBParser(QUIET=True)
    structure = p.get_structure("structure", pdb_path)

    # get only the first chain, matching make_pdb.py's convention
    for chain in structure.get_chains():
        break

    n_missing = 0
    for res in chain.get_residues():
        if res.get_resname() not in residue_types:
            continue
        resnum = res.get_id()[1]
        value = b_factor_by_resnum.get(resnum)
        if value is None:
            # residue present in the PDB but not in our backbone-derived
            # res_ids (e.g. missing backbone atoms mdtraj excluded)
            value = exclude_value
            n_missing += 1
        for atom in res.get_atoms():
            atom.set_bfactor(value)

    if n_missing:
        print(f"\nWarning: {n_missing} residue(s) in the PDB had no attention "
              f"score (missing backbone atoms?) and were set to {exclude_value}.")

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb)
    print(f"\nWrote B-factor-annotated PDB to {output_pdb}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pdb', required=True,
                         help='path to the PDB file the attention weights were computed on')
    parser.add_argument('--attention-weights', required=True,
                         help='path to the saved *-attention_weights.npy from xtal_predict.py')
    parser.add_argument('--pocket-resnums', required=True, type=int, nargs='+',
                         help='residue numbers (resSeq) defining the known/apo pocket, '
                              'e.g. --pocket-resnums 10 14 15')
    parser.add_argument('--top-n', type=int, default=40,
                         help='number of top-attended residues to print (default: 40)')
    parser.add_argument('--exclude-value', type=float, default=-100.0,
                         help='B-factor value assigned to pocket residues and their '
                              'immediate neighbors, so they stand out distinctly when '
                              'visualized (default: -100.0)')
    parser.add_argument('--output-pdb', default=None,
                         help='where to write the B-factor-annotated PDB; defaults to '
                              '{pdb}_key_residues.pdb next to the input PDB')
    args = parser.parse_args()

    output_pdb = args.output_pdb or args.pdb.rsplit('.', 1)[0] + '_key_residues.pdb'

    res_ids = get_backbone_resnums(args.pdb)
    attn = load_attention_weights(args.attention_weights, n_residues=len(res_ids))

    scores, excluded_inds = compute_key_residue_scores(attn, res_ids, args.pocket_resnums)
    print_top_residues(scores, excluded_inds, res_ids, args.top_n)
    write_bfactor_pdb(args.pdb, res_ids, scores, excluded_inds, args.exclude_value, output_pdb)
