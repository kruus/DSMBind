<!--
vim: ts=4 sw=4 et
-->
# mytut.py
- converted from (my-)tutorial.ipynb
- conda packages (add to hug-envs.sh) ...
    - seaborn
- pip packages:
    - pip install 'git+https://github.com/NVIDIA/dllogger.git'
    - pip install 'sru==3.0.0.dev6' 

- ISSUES:
    - pip install 'biopython==1.79'
    - FAILING:       Bio/Align/_aligners.c: In function 'set_alphabet':
      Bio/Align/_aligners.c:1786:18: error: 'PyUnicode_WCHAR_KIND' undeclared
      (first use in this function); did you mean 'PyUnicode_4BYTE_KIND'?
      (The fault may lie in package 'pyobjc')
    
    - pip install git+https://github.com/jonathanking/sidechainnet/ 
    - seems to require python 3.11 (?)

changed hug-envs.sh (on js01) to use python 3.11
hug-envs.sh has several other mods, like openmm<2 (1.7.1 is latest acceptable version).

# DSMBind data:
Install [DSMBind data fromzenodo](https://zenodo.org/records/10402853)
or from local copy `snake10:/datasets/data/data-DSMBind-zenodo-10402853.zip`.

Unzipped in DSMBind/, these seem to have various data and scripts.

# Code notes
- bind and tgt are represented as foo_X, foo_S, foo_A, foo_V[opt.] from 'featurize'
    - S ~ indices
    - X ~ coords
    - A ~ atom types
    - V ~ dihedrals
- dataset -> self.data is a list of dictionary entry, with keys for
    - binder\_foo,
    - target\_foo, and
    - receptor\_seqseq (may be same as target\_seq)
- dataset may be read from pickle(DrugDataset, PeptideDataset) or json(AntibodyDataset)
- like
  receptor\_seq, target\_atypes, target\_coords, target\_seq, dihedrals
- `tgt_S <-- embedding of 'receptor_seq'`
- `bind_S <-- embedding of 'binder_seq'`
