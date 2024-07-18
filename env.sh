set +x

#
# remove env dsm0
#
echo "CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"
#if [ x"$CONDA_DEFAULT_ENV" = x"${CONDA_DEFAULT_ENV%dsm0}" ]; then 
if [ x"${CONDA_DEFAULT_ENV}" = "xdsm0" ]; then 
	conda deactivate
fi
if conda env list 2>&1 | grep dsm0
then
	echo "env dsm0 already exists"
	conda env remove -n dsm0
else
	echo "env dsm0 does not yet exist"
fi

#
# recreate dsm0 initial state via conda --clone
#
chem=/local/kruus/chem
mydir=`pwd`
conda create -n dsm0 --clone esm-pyt2
conda env list
set -x
. ~/bin/conda.sh
conda activate dsm0 || activate dsm0
set +x
export | grep CONDA
echo "A CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'{torch.__version__=}'"
cd $chem/esm-demo
time python demo-esmfold.py 2>&1 | tee "$mydir/demo-esmfold-1.log"
# correct output is something like:
#	
#	torch.__version__='2.0.1'
#	esm.version.version='2.0.1'
#	inference pdb for sequence[65] in 1.534 s
#	88.28911067193674
#	write pdb and struct.b_factor.mean() in 0.238 s
#	
#	real	0m37.557s
#	user	1m2.444s
#	sys	0m9.914s
echo ""
echo "*** End initial env dsm0 demo-esmfold.py test ***"
echo ""

#
# Modify/update env dsm0
#
#conda install --override-channels -c nvidia -c pytorch -c conda-forge -c bioconda -y python=3.11
cd $mydir 				# cd $chem/DSMBind
#conda update --file env-dsm0.yml
conda remove -y fairscale
#mamba env update --name dsm0 --file env-dsm1-export.yml # dms1.sh uses env-dsm0.yml, thenn conda env export
#mamba env update --name dsm0 --file env-dsm1-b.yml # dms1.sh uses env-dsm0.yml, thenn conda env export
echo ""
echo "   mamba env update --name dsm0 --file env-dsm0.yml"
# *** conda env update now began to take a HUGE time to solve the environment ***
#conda env update --name dsm0 --file env-dsm0.yml
# *** so instead, try small updates
echo 'mamba install --override-channels -c nvidia -c pytorch -c conda-forge -c bioconda -y biopython==1.79 numpy==1.23.5 ipython jupyterlab ipywidgets ipykernel nb_conda_kernels'
mamba install --override-channels -c nvidia -c pytorch -c conda-forge -c bioconda -y biopython==1.79 numpy==1.23.5 ipython jupyterlab ipywidgets ipykernel conda-forge::nb_conda_kernels conda-forge::scikit-learn conda-forge::rdkit conda-forge::chemprop
#  tqdm matplotlib pandas seaborn # already OK
#  all that changed was numpy, biopython (ca-certificates certifi)
#  Missing for DSMBing .ipynb : conda-forge::scikit-learn
echo 'pip install sru==3.0.0.dev  git+https://github.com/jonathanking/sidechainnet/'
pip install 'sru==3.0.0.dev6' 'git+https://github.com/jonathanking/sidechainnet/'
# OH.  DSMBind lacks setup.py so is NOT pip-installable
#echo 'pip install -e .'
#pip install -e .


#
# ensure updated env dsm0
#
if [ ! x"${CONDA_DEFAULT_ENV}" = "dsm0" ]; then
	if [ ! x"${CONDA_DEFAULT_ENV}" = "xbase" ]; then 
		echo "B CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV --> conda deactivate"
		#conda deactivate
		conda activate			# without args, returns to 'base'
	fi
	if [ ! x"${CONDA_DEFAULT_ENV}" = "xdsm0" ]; then 
		conda activate dsm0
	fi
fi
echo "C CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"

#
# test 'esm' in env dsm0
#
cd $chem/esm-demo
echo "D CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"
time python demo-esmfold.py 2>&1 | tee "$mydir/demo-esmfold-2.log"
echo ""
echo "*** End second env dsm0 demo-esmfold.py test ***"
echo ""

# DSMBind issues:
#   ERROR: from sru import SRUpp  :  Fix by pip install sru=3.0.0.dev6 (was 3.0.0.dev)
#   ERROR: from rdkit import Chem :  Fix by conda install --override-channels -c conda-forge -c bioconda rdkit
#   --> Install: 15 packages Change: 13 packages Upgrade: 2 packages Downgrade: 21 packages (all seem minor)
#   also missing:    chemprop (-> conda-forge, 1.61)

# Issue: during
#    embedding = load_esm_embedding(test_equibind.data + test_casf16.data + test_fep.data, ['target_seq'])
#    OutOfMemoryError: CUDA out of memory. Tried to allocate 340.00 MiB (GPU 0; 11.90 GiB total capacity; 10.83 GiB already allocated; 322.12 MiB free; 10.92 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
#    with os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator, max_split_size_mb=256"
#    see 11.03 GiB already allocated; 75.38 MiB free; 11.09 GiB reserved in total
#    ... so it need > 12Gb cuda memory.
# introduced sequence length truncation parameter --> len 350 fits into 12 Gb for embedding step.
#   torch>=1.9 deprecated symeig -- replace w/ torch.linalg.eigh, but parameters differ...
#   TypeError: linalg_eigh(): argument 'UPLO' (position 2) must be str, not bool
# torch.symeig(input, eigenvectors=False, upper=True, *, out=None)
#   instructions in http://docs.pytorch.wiki/en/generated/torch.symeig.html

# actual fix of GPU memory should be to add length restriction to 'DrugDataset',
# to filter out any data for proteins > 350 (or so) residues.

cd $mydir

