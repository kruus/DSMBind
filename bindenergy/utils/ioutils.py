# import biotite.structure as struc
# from biotite.structure import AtomArray, Atom
import gc

import esm
import torch
from biotite.structure import Atom
from tqdm import tqdm  # , trange

# from biotite.structure.io import save_structure
# from bindenergy.data.constants import *
# from bindenergy.data.constants import RESTYPE_1to3, ALPHABET, ATOM_TYPES
# from bindenergy.data.constants import RES_ATOM14, AA_WEIGHTS, ATOM_WEIGHTS
from bindenergy.data.constants import ALPHABET, RES_ATOM14, RESTYPE_1to3


def print_pdb(coord, seq, chain, indices=None):
    array = []
    for i in range(coord.shape[0]):
        idx = indices[i] + 1 if indices else i + 1
        aaname = seq[i]
        aid = ALPHABET.index(aaname)
        aaname = RESTYPE_1to3[aaname]
        for j, atom in enumerate(RES_ATOM14[aid]):
            if atom != "" and (coord[i, j] ** 2).sum() > 1e-4:
                atom = Atom(
                    coord[i, j],
                    chain_id=chain,
                    res_id=idx,
                    atom_name=atom,
                    res_name=aaname,
                    element=atom[0],
                )
                array.append(atom)
    return array


def print_ca_pdb(coord, seq, chain, indices=None):
    array = []
    for i in range(coord.shape[0]):
        idx = indices[i] + 1 if indices else i + 1
        aaname = seq[i]
        aaname = RESTYPE_1to3[aaname]
        atom = Atom(
            coord[i, 1],
            chain_id=chain,
            res_id=idx,
            atom_name="CA",
            res_name=aaname,
            element="C",
        )
        array.append(atom)
    return array


# esm will use 16-24 Gb GPU mem
#  https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb
#  so I can convert the model the 'half'
#  but first try
#    import torch
#    print(torch.cuda.is_bf16_supported()
#
# memory usage for attention during training will scale as O(batch_size * num_layers * seq_len^2)
#
def load_esm_embedding(data, fields, truncation_seq_length: int = None):
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()

    # Note: 3B float32 suggests 16G to 24G GPU mem
    # We can switch the esm "stem" to bfloat16 (training used bf16 anyway)
    #   (If your GPU supports it)
    if torch.cuda.is_bf16_supported():
        model.esm = model.esm.half()

    batch_converter = alphabet.get_batch_converter(
        # truncation_seq_length=200   # just for kicks [default=None, "fixes" CUDA out of mem]
        # This api change allows "load", but fails later from unexpected lengths
        # (perhaps fully ignore such items?)
        truncation_seq_length=truncation_seq_length
    )
    model = model.cuda()
    model.eval()
    embedding = {}
    with torch.no_grad():
        for f in fields:
            seqs = [d[f] for d in data if d[f] not in embedding]
            for s in tqdm(sorted(set(seqs))):
                batch_labels, batch_strs, batch_tokens = batch_converter([(s, s)])
                batch_tokens = batch_tokens.cuda()
                results = model(batch_tokens, repr_layers=[36], return_contacts=False)
                # Note: issues later on if len(s) was truncated!
                # embedding[s] = results["representations"][36][0, 1:len(s) + 1].cpu()
                assert len(batch_strs) == 1
                embedding[s] = results["representations"][36][
                    0, 1 : len(batch_strs[0]) + 1
                ].cpu()
                # trying to reduce memory requirements...
                # batch_labels = None
                # batch_strs = None
                batch_tokens = None
                results.clear()
                results = None
                torch.cuda.empty_cache()
                # o/w may need to call get_batch_converter(truncation_seq_length=80) or such
                gc.collect()
                # print('GC collected objects : %d' % gc.collect())

    model = None
    torch.cuda.empty_cache()
    print("FINAL: GC collected objects : %d" % gc.collect())

    return embedding
