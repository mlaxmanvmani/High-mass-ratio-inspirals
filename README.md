# gwbench
## Acknowledgment
We request that any academic report, publication, or other academic
disclosure of results derived from the use of this software acknowledge
the use of the software by an appropriate acknowledgment or citation.

The gwbench software can be cited from [arXiv:2010.15202](https://arxiv.org/abs/2010.15202), with INSPIRE BibTeX entry:
```
@article{Borhanian:2020ypi,
    author = "Borhanian, Ssohrab",
    title = "{gwbench: a novel Fisher information package for gravitational-wave benchmarking}",
    eprint = "2010.15202",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    month = "10",
    year = "2020"
}
```

## Installation via pip
```
pip install gwbench
```

## Installation from source
### Clone the gwbench repository and enter it
Clone this repository and follow the next steps.
```
git clone https://gitlab.com/sborhanian/gwbench.git
cd gwbench
```

### Using conda
#### Source Oasis Conda - *do this first; only on LIGO clusters needed*
```
source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh  
which conda
```

The last line should print `/cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/condabin/conda` or similar.

#### Setup conda virtual environment
```
conda create -y --name gwbench python=3.9  
conda activate gwbench  
conda install -y -c conda-forge --file requirements_conda.txt  
```

### Using `python -m venv` and pip
Replace `~/gwbench` with the appropriate path of choice in the following instructions:
```
python3 -m venv ~/gwbench
source ~/gwbench/bin/activate
pip install -r requirements_pip.txt
```

### Using pip or conda
Install while the virtual environment is active:
```
pip install .
```

### Uninstall
```
pip uninstall gwbench
```

### Test
```
cd example_scripts  
python test_run.py
```
