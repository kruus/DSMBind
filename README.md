# DSMBind

This is the official Github repo for the following papers:
* Jin et al., DSMBind: SE(3) denoising score matching for unsupervised binding energy prediction and nanobody design, Biorxiv 2023
* Jin et al., Unsupervised protein-ligand binding energy prediction with Neural Euler's Rotation Equations, NeurIPS 2023

# Installation

Please make sure you install the following packages
* pytorch (tested on v1.13)
* biotite (https://www.biotite-python.org/install.html)
* SRU++ (https://github.com/asappresearch/sru)
* ESM-2 (https://github.com/facebookresearch/esm)

Once you finished these installation, please run `pip install -e .`. To install SRU++, run 
```
git clone https://github.com/asappresearch/sru
cd sru
git checkout 3.0.0-dev
pip install .
```
