import sys
import os
import shutil
from glob import glob

import tensorflow as tf
import numpy as np
import mdtraj as md
import yaml

from models import MQAModel
from validate_performance_on_xtals import process_strucs
from util import load_checkpoint


DEFAULT_CONFIG = {
    'nn_path': '../models/ligsite_bigdataset_attention/'
               'aepocketminer',
    'input_pdb_directory': 'inputs',
    'output_directory': 'results/aepocketminer',
    'use_attention': True,
    'num_heads': 2,
    'dropout_rate': 0.1,
    'num_layers': 4,
    'hidden_dim': 100,
    'save_attention_weights': True,
    'attention_weights_filename': 'attention_weights.npy',
    'debug': False,
}


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

    nn_path = config['nn_path']
    INPUT_PDB_DIRECTORY = config.get('input_pdb_directory', 'inputs')
    OUTPUT_DIRECTORY = config.get('output_directory', 'results')
    debug = config.get('debug', False)

    NUM_HEADS = config.get('num_heads', 2)
    DROPOUT_RATE = config.get('dropout_rate', 0.1)
    NUM_LAYERS = config.get('num_layers', 4)
    HIDDEN_DIM = config.get('hidden_dim', 100)

    # --- ATTENTION ---
    USE_ATTENTION = config.get('use_attention', True)
    SAVE_ATTENTION_WEIGHTS = config.get('save_attention_weights', USE_ATTENTION)
    ATTENTION_WEIGHTS_FILENAME = config.get('attention_weights_filename', 'attention_weights.npy')

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
    print("Loading trained checkpoint...", flush=True)
    opt = tf.keras.optimizers.Adam()
    load_checkpoint(model, opt, nn_path)

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
