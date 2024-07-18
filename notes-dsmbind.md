<!-- Required extensions: sane_lists, mdx_math(enable_dollar_delimiter=1)-->
<!-- mdx_math is for $...$ and $$...$$ support -->
<!-- other extensions (examples) wikilinks -->

# DSMBind
This is a directory-only project that might perform better then ESMfold.
Unfortunately, the environment setup needs a layered install, that
avoids satisfying all dependencies.

After much work, `env.sh` leverages an initial `esm-pyt2` environment
(see my chem/esm/ subdirectory, env-pyt2.0.yml

The `ejk.ipynb` (copy of tutorial.ipynb) will not run on snake10:  
CUDA out of memory (12 Gb available)

The dataset is loaded into cuda memory first, and takes 6 Gb