import sys
import os
import shutil
from glob import glob

import tensorflow as tf
import numpy as np
import mdtraj as md
import yaml

from models import MQAModel
#from validate_performance_on_xtals import process_strucs
from util import load_checkpoint


# # NEED TO CHANGE HERE or GIVE YAML CONFIG FILE
# DEFAULT_CONFIG = {
#     'nn_path': '/mnt/pure/bowmanlab/sizhang/repos/ae-pocketminer/models/aepocketminer',
#     # Absolute path to the trained model checkpoint.
#     # After downloading the ae-pocketminer repo, update this to:
#     # /path/to/ae-pocketminer/models/aepocketminer 
#     'input_pdb_directory': 'inputs',
#     'output_directory': 'results/aepocketminer',
#     'use_attention': True,
#     # 'save_attention_weights': True, # can be modified if needed
#     # 'attention_weights_filename': 'attention_weights.npy', # can be modified if needed
#     'debug': False,
# }

abbrev = {"ALA" : "A" , "ARG" : "R" , "ASN" : "N" , "ASP" : "D" , "CYS" : "C" , "CYM" : "C", "GLU" : "E" , "GLN" : "Q" , "GLY" : "G" , "HIS" : "H" , "ILE" : "I" , "LEU" : "L" , "LYS" : "K" , "MET" : "M" , "PHE" : "F" , "PRO" : "P" , "SER" : "S" , "THR" : "T" , "TRP" : "W" , "TYR" : "Y" , "VAL" : "V"}
lookup = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9, 'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8, 'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 'N': 2, 'Y': 18, 'M': 12}

def process_strucs(strucs):
    """Takes a list of single frame md.Trajectory objects
    """

    pdbs = []
    for s in strucs:
        prot_iis = s.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = s.atom_slice(prot_iis)
        pdbs.append(prot_bb)

    B = len(strucs)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])
    X = np.zeros([B, L_max, 4, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)

    for i, prot_bb in enumerate(pdbs):
        l = prot_bb.top.n_residues
        xyz = prot_bb.xyz.reshape(l, 4, 3)

        seq = [r.name for r in prot_bb.top.residues]
        S[i, :l] = np.asarray([lookup[abbrev[a]] for a in seq], dtype=np.int32)
        X[i] = np.pad(xyz, [[0,L_max-l], [0,0], [0,0]],
                      'constant', constant_values=(np.nan, ))

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.
    X = np.nan_to_num(X)

    return X, S, mask


def predict_on_xtals(model, X, S, mask):
    prediction = model(X, S, mask, train=False, res_level=True)
    return prediction


def make_predictions(pdb_paths, model, debug=False, output_basename=None):
    '''
        pdb_paths : list of pdb paths
        model : MQAModel corresponding to network in nn_path (already loaded)
    '''
    strucs = [md.load(s) for s in pdb_paths]
    X, S, mask = process_strucs(strucs)
    if debug:
        np.save(f'{output_basename}_X.npy', X)
        np.save(f'{output_basename}_S.npy', S)
        np.save(f'{output_basename}_mask.npy', mask)
    predictions = predict_on_xtals(model, X, S, mask)
    return predictions


if __name__ == '__main__':
    # Optional yaml config; falls back to DEFAULT_CONFIG above if not given,
    # so `python src/xtal_predict.py` with no arguments still works.
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as stream:
            config = yaml.safe_load(stream)
    else:
        config = DEFAULT_CONFIG

    #--- INPUT/OUTPUT DIRECTORIES ---
    INPUT_PDB_DIRECTORY = config.get('input_pdb_directory', 'inputs')
    OUTPUT_DIRECTORY = config.get('output_directory', 'results')
    debug = config.get('debug', False)

    # --- MODEL HYPERPARAMETERS ---
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 4
    HIDDEN_DIM = 100
    NN_PATH = config['nn_path'] # path to the trained model checkpoint

    # --- ATTENTION ---
    USE_ATTENTION = config.get('use_attention', True)
    NUM_HEADS = 2 if USE_ATTENTION else None # will not be used if USE_ATTENTION is False
    SAVE_ATTENTION_WEIGHTS = config.get('save_attention_weights', USE_ATTENTION) and USE_ATTENTION
    # If only one PDB is provided, the attention weights filename could be
    # specified in the config. For simplicity, we use the same filename for
    # all PDBs here, and move/rename it after each prediction.
    ATTENTION_WEIGHTS_FILENAME = config.get('attention_weights_filename', 'attention_weights.npy')\
          if USE_ATTENTION else None
    
    # Check for potential mismatch between the model checkpoint and the attention setting
    if 'pocketminer' in NN_PATH.lower() and 'aepocketminer' not in NN_PATH.lower() and USE_ATTENTION:
        print("\nWARNING: nn_path appears to point to a PocketMiner checkpoint, but "
              "use_attention=True (only valid for AE-PocketMiner). PocketMiner was "
              "trained without attention layers; so any attention_weights.npy "
              "produced would be meaningless.", flush=True)

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    model = MQAModel(
        node_features=(8, 50),
        edge_features=(1, 32),
        hidden_dim=(16, HIDDEN_DIM),
        attention_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE,
        use_attention=USE_ATTENTION,
        save_attention_weights=SAVE_ATTENTION_WEIGHTS,
        attention_weights_filename=ATTENTION_WEIGHTS_FILENAME,
    )

    # Load model checkpoint ONCE, outside the per-file loop
    print("\nLOADING TRAINED CHECKPOINT...", flush=True)
    opt = tf.keras.optimizers.Adam()
    load_checkpoint(model, opt, NN_PATH)

    print(f'Processing the PDB dataset in {INPUT_PDB_DIRECTORY} ...', flush=True)
    for filename in os.listdir(INPUT_PDB_DIRECTORY):
        print(f'Processing {filename} .. ', flush=True)
        strucs = [f'{INPUT_PDB_DIRECTORY}/{filename}']
        output_name = f'{filename.split(".")[0]}'

        try:
            if debug:
                output_basename = f'{OUTPUT_DIRECTORY}/{output_name}'
                predictions = make_predictions(strucs, model, debug=True, output_basename=output_basename)
            else:
                predictions = make_predictions(strucs, model)
        except Exception as e:
            print(f'ERROR {filename}:\n{e}', flush=True)
            continue

        np.save(f'{OUTPUT_DIRECTORY}/{output_name}-preds.npy', predictions)

        # Check if attention weights were saved by the model and move/rename
        # them alongside this file's predictions
        if SAVE_ATTENTION_WEIGHTS and os.path.exists(ATTENTION_WEIGHTS_FILENAME):
            dest_path = f'{OUTPUT_DIRECTORY}/{output_name}-attention_weights.npy'
            shutil.move(ATTENTION_WEIGHTS_FILENAME, dest_path)
            print(f'Moved attention weights to {dest_path}', flush=True)
