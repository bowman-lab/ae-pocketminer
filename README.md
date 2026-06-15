---
title: AE-PocketMiner
emoji: 🧬
colorFrom: teal
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# AE-PocketMiner — Cryptic Pocket & Allosteric Coupling Prediction

An AI method for simultaneously predicting cryptic binding pockets and their 
allosteric coupling to the rest of the protein from a single input structure,
using an attention-enabled Geometric Vector Perceptron graph neural network.

<p align="center">
  <img src="docs/imgs/model_archt.png" width="700" alt="AE-PocketMiner model architecture"/>
  <br/>
  <em>Figure 1. AE-PocketMiner model architecture.</em>
</p>

---

## Table of contents

- [Web interface](#web-interface)
- [Local prediction](#local-prediction)
- [Installation](#installation)
- [Training](#training)
- [Training data](#training-data)
- [Attribution](#attribution--original-work)
- [Citation](#citation)
- [License](#license)

---

## Web interface

The easiest way to run AE-PocketMiner is through the Hugging Face web app —
no installation required.

Upload any PDB file (crystal structure, AlphaFold model, or simulation
snapshot) and get back:

| Output | Description |
|---|---|
| **Interactive 3D viewer** | Structure coloured blue→white→red by per-residue pocket probability, rotatable in the browser |
| **Per-residue probability chart** | Hover any bar to highlight that residue in the 3D view |
| `output.pdb` | Input PDB with B-factors replaced by cryptic pocket probability × 100 (ready for PyMOL / VMD) |
| `preds.npy` | Per-residue probability array, shape `(1, n_residues)` |
| `attention.npy` | Attention weight matrix encoding allosteric coupling, shape `(n_residues, n_residues)` |

---

## Local prediction

If you prefer to run predictions on your own machine — for example on a batch
of structures or on a compute cluster — use `xtal_predict.py` directly.

```bash
# Place your PDB file(s) in the inputs/ folder
mkdir -p inputs results

cp your_protein.pdb inputs/

# Run prediction
python src/xtal_predict.py
```

Output files are written to `results/`:
- `results/your_protein-preds.npy` — per-residue pocket probabilities
- `results/your_protein-attention_weights.npy` — attention weight matrix

To write a PDB with B-factors set to pocket probability (for PyMOL visualisation):

```bash
python src/write_bfactor_pdb.py \
    --input  inputs/your_protein.pdb \
    --preds  results/your_protein-preds.npy \
    --output results/your_protein_scored.pdb
```

---

## Installation

> *Full instructions coming soon.*

---

## Training

> *Full training guide coming soon.*

---

## Training data

> *Dataset download instructions coming soon.*

---

## Attribution & original work

This work builds on the infrastructure provided by **PocketMiner**, developed 
by the [Bowman Lab](https://bowmanlab.seas.upenn.edu), while introducing a 
newly developed predictive model with expanded training data, a modified 
network architecture with attention, and additional functionality for 
allosteric residue prediction.

> Meller, A., Ward, M., Borowsky, J. *et al.* Predicting locations of cryptic 
> pockets from single protein structures using the PocketMiner graph neural 
> network. *Nature Communications*, 14, 1177 (2023).
> https://doi.org/10.1038/s41467-023-36699-3

**Original repository:** 
👉 https://github.com/Mickdub/gvp/tree/pocket_pred

The GVP model source files in `src/` are copied from that repository.
For retraining, benchmarking, or understanding the full methodology,
please refer to the original repo.

---

## Citation

If you use AE-PocketMiner in your research, please cite:
https://doi.org/10.64898/2026.05.21.726899

---

## License

Released under the **MIT License** — see [LICENSE](LICENSE).
