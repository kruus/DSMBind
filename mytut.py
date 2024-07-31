#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import numpy as np
import scipy
# import random
import glob
# import csv
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve
import torch
# from bindenergy import *
from bindenergy.data.drug import DrugDataset
from bindenergy.data.antibody import AntibodyDataset
from bindenergy.data.protein import ProteinDataset
from bindenergy.models.energy import AllAtomEnergyModel
from bindenergy.models.drug import DrugAllAtomEnergyModel
from bindenergy.utils import load_esm_embedding
import tqdm


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


if is_notebook():
    get_ipython().run_line_magic('env', 'TORCH_EXTENSIONS_DIR=.')
else:
    os.environ['TORCH_EXTENSIONS_DIR'] = '.'

sns.set(rc={"figure.dpi": 300, 'savefig.dpi': 300})

# ### Protein-Ligand Binding

# Load model (DrugAllAtomEnergyModel considers the full-atom structure
# of a protein-ligand complex, including side-chains).

# In[3]:

fn = "ckpts/model.drug.allatom"
model_ckpt, opt_ckpt, model_args = torch.load(fn)
model = DrugAllAtomEnergyModel(model_args).cuda()
model.load_state_dict(model_ckpt)
model.eval()


# In our NeurIPS paper, we used the PDBBind core (CASF 2016) as the test set.
# We use the docking test set from EquiBind (Starks et al., ICML 2022)as our validation set.
#
# In our new biorxiv paper, we further evaluate DSMBind on Merck free energy
# perturbation (FEP) benchmark.

# In[4]:


test_casf16 = DrugDataset("data/drug/test_casf16.pkl", 50)
test_equibind = DrugDataset("data/drug/test_equibind.pkl", 50)
test_fep = DrugDataset("data/drug/test_fep.pkl", 50)


# In[5]:


embedding = load_esm_embedding(
    test_equibind.data + test_casf16.data + test_fep.data, ['target_seq']
)


# Inference script (note: DSMBind predicted score is the higher the better,
# while binding affinity is the lower the better)

# In[6]:


def pdbbind_evaluate(model, data, embedding, args):
    model.eval()
    score = []
    label = []
    with torch.no_grad():
        for entry in tqdm(data):
            binder, target = DrugDataset.make_bind_batch([entry], embedding, args)
            pred = model.predict(binder, target)
            score.append(-1.0 * pred.item())
            label.append(entry['affinity'])
    return scipy.stats.spearmanr(score, label)[0], score, label


# Make predictions on CASF-2016/Equibind test sets

# In[7]:


casf16_corr, casf16_score, casf16_label = pdbbind_evaluate(
    model, test_casf16, embedding, model_args
)
equibind_corr, equibind_score, equibind_label = pdbbind_evaluate(
    model, test_equibind, embedding, model_args
)


# In[8]:


sns.regplot(x=casf16_score, y=casf16_label)
plt.xlabel('Predicted binding energy')
plt.ylabel(r'Experimental binding affinity ($\log_{10}$)')
plt.title(f'CASF 2016 Spearman R = {casf16_corr:.4f}')


# In[9]:


sns.regplot(x=equibind_score, y=equibind_label)
plt.xlabel('Predicted binding energy')
plt.ylabel(r'Experimental binding affinity ($\log_{10}$)')
plt.title(f'Equibind test Spearman R = {equibind_corr:.4f}')


# ##### Merck FEP Benchmark Evaluation

# In[10]:


score = defaultdict(list)
label = defaultdict(list)
with torch.no_grad():
    for entry in tqdm(test_fep):
        pdb = entry['pdb']
        binder, target = DrugDataset.make_bind_batch([entry], embedding, model_args)
        pred = model.predict(binder, target)
        score[pdb].append(pred.item())
        label[pdb].append(-1.0 * entry['affinity'])


# In[11]:


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
fig.text(0.5, 0.04, 'Predicted binding energy', ha='center')
fig.text(0.08, 0.5, 'Experimental binding affinity', va='center', rotation='vertical')
for i, pdb in enumerate(score.keys()):
    i, j = i // 4, i % 4
    sns.regplot(x=score[pdb], y=label[pdb], ax=axes[i, j])
    corr = scipy.stats.spearmanr(score[pdb], label[pdb])[0]
    axes[i, j].set_title(f'{pdb}: Spearman R={corr:.3f}')


# ### Antibody-Antigen Binding

# In[12]:


def sabdab_evaluate(model, data, embedding, args):
    model.eval()
    pred, label = [], []
    for ab in tqdm(data):
        binder, target = AntibodyDataset.make_local_batch([ab], embedding, args)
        score = model.predict(binder, target)
        pred.append(-1.0 * score.item())
        label.append(ab['affinity'])
    return scipy.stats.spearmanr(pred, label)[0], pred, label


# In our NeurIPS and biorxiv paper, we evaluate on two test sets:
# * The first test set is from SAbDab. It has 566 antibody-antigen complexes
#   with binding affinity labels
# * The second test set is from . It has 424 HER2-trastuzumab variants (CDR3 mutation)
#   with binding affinity labels

# In[13]:


test_sabdab = AntibodyDataset("data/antibody/test_sabdab.jsonl", cdr_type='123456', epitope_size=50)
test_HER2 = AntibodyDataset("data/antibody/test_HER2.jsonl", cdr_type='123456', epitope_size=50)


# In[14]:


embedding = load_esm_embedding(test_sabdab.data + test_HER2.data, ['antibody_seq', 'antigen_seq'])


# In[15]:


model_ckpt, opt_ckpt, model_args = torch.load("ckpts/model.antibody.allatom")
model = AllAtomEnergyModel(model_args).cuda()
model.load_state_dict(model_ckpt)
model.eval()


# In[16]:


test_corr, pred, label = sabdab_evaluate(model, test_sabdab, embedding, model_args)


# In[17]:


sns.regplot(x=pred, y=label)
plt.xlabel('Predicted binding energy')
plt.ylabel('Experiment binding affinity')
plt.title(f'SAbDab test correlation: Spearman R = {test_corr:.4f}')


# In[18]:


pred, label = [], []
for ab in tqdm(test_HER2):
    binder, target = AntibodyDataset.make_local_batch([ab], embedding, model_args)
    score = model.predict(binder, target)
    pred.append(score.item())
    label.append(int(ab['affinity'] < -8.7))  # better than wildtype trastuzumab binding affinity

fpr, tpr, _ = roc_curve(label, pred)
plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc_score(label, pred),
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("HER2 AUROC")
plt.legend(loc="lower right")
plt.show()


# #### SKEMPI Results

# In[19]:


test_data = ProteinDataset("data/skempi/skempi_all.pkl", 50)
embedding = load_esm_embedding(test_data.data, ['binder_full', 'target_full'])


# In[20]:


model_ckpt, _, model_args = torch.load("ckpts/model.skempi.allatom")
model = AllAtomEnergyModel(model_args).cuda()
model.load_state_dict(model_ckpt)
model.eval()


# In[21]:


with torch.no_grad():
    wt_map = {}
    pred, label = [], []
    for entry in test_data.data:
        pdb, mutation, ddg = entry['pdb']
        if len(mutation) == 0:
            binder, target = ProteinDataset.make_local_batch([entry], embedding, model_args,
                                                             'binder', 'target')
            wt_map[pdb] = model.predict(binder, target) + model.predict(target, binder)

    for entry in test_data.data:
        pdb, mutation, ddg = entry['pdb']
        if len(mutation) > 0:
            binder, target = ProteinDataset.make_local_batch([entry], embedding, model_args,
                                                             'binder', 'target')
            score = model.predict(binder, target) + model.predict(target, binder)
            score = score - wt_map[pdb]
            pred.append(-1.0 * score.item())
            label.append(ddg)


# In[22]:


test_corr = scipy.stats.spearmanr(pred, label)[0]
sns.regplot(x=pred, y=label)
plt.xlabel('Predicted DDG')
plt.ylabel('Experimental DDG')
plt.title(f'SKEMPI test correlation: Spearman R = {test_corr:.4f}')


# ### Ligand virtual screening

# In[4]:


torch.cuda.set_device(0)
fn = 'ckpts/model.recA'
model_ckpt, opt_ckpt, model_args = torch.load(fn)
model = DrugAllAtomEnergyModel(model_args).cuda()
model.load_state_dict(model_ckpt)
model.eval()


# In[6]:


sdf_list = sorted(glob.glob('data/recA/*/rank1.sdf'))
result, _ = model.virtual_screen('data/recA/1xmv.pdb', sdf_list, batch_size=1)
pd.DataFrame(result)

# end of file
