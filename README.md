# Installation

## Acquiring specific versions of Upside to reproduce a paper

Papers that use Upside have a tag that indicates the precise git version of Upside needed to reproduce the results of the paper.  The `v2.0.0` tag corresponds to the docking paper. Note that this docking version of Upside diverged from the main version at tag `docking_fork`. Refer to the readme.md of the specific tagged release for instructions for that release instead of this readme. 

## General Notes

We recommend installing within an Anaconda environment with Python 2.7 since Python 2.7 has reached End of Life from its core developer team. This install guide assumes that you are using Linux. Installation on latest versions of other OSs is not tested.

## Dependencies

Compile dependencies. Acquire these from conda channels if you are unable to install packages to your system or compile from source locally on your own.

  * CMake, 2.8+
  * C++11 compiler, such as GCC 4.8+
  * HDF5, 1.8+ installed with high-level interface compile option (which it is if installed from the default conda channel for at least v1.10.4)
  * Eigen Matrix Library, 3.0+

Python dependencies

  * Python 2.7
  * Biopython
  * Jupyter (for running the example analysis notebook)
  * MDTraj
  * NGLView (might need to run `jupyter-nbextension enable nglview --py --user` after installation)
  * Numpy
  * Scipy
  * Pandas
  * Prody
  * Progress (for progress bars)
  * PyTables

External software

- SCWRL4 (build explicit sidechains needed for CAPRI evaluation)

## Compiling Upside

Change to the `obj/` directory of the upside-docking directory and execute the following commands.

    rm -rf ../obj/* && cmake ../src && make -j

After these commands execute successfully, the `obj/` directory will contain the `upside` executable and the `libupside.so` shared library (exact name of shared library may depend on operating system).

# Usage

Refer to the "Running molecular dynamics" sections in the readme.md of the original pre-docking fork version of Upside [here](https://github.com/nffaruk/upside-docking/tree/812b7edeca3e8b5ed63d363a5e3524bd40e66695) for constant temperature and replica exchange simulations of single proteins.

## Protein Complex Simulations and Analysis

This section covers an example that demonstrates features used in the work for the docking paper. The `examples/` directory of the upside-docking directory contains a `PDB/` directory with PDB files for 2OOB, a small binary heterocomplex. The directory also contains a Python script, `nse_run_complex.py`, that runs a native state ensemble simulation of this complex using the trained binding energy terms and computes energies and CAPRI criteria metrics. The Jupyter notebook, `todo.ipynb`, includes plots of the output energies and CAPRI metrics from the simulation and shows how to load and visualize the trajectory. 

Open `nse_run_complex.py` and adjust the directory for SCWRL4, `scwrl_dir` (line 21), to where you've installed it. Also inspect for any paths that need to be changed if you've placed files outside of expected locations (e.g. it assumes you are running the example script from the `examples/` directory, and it obtains other file/module locations relative to that), and information about docking related functions and options employed.  You may want to change the number of simulation replicates, `n_rep` (line 176), and the number of subprocesses for parallel analysis, `pool_size` (line 177). These should match the number of CPU cores you've allocated for optimal efficiency.

Simply run with `python nse_run_complex.py`.  It will output starting energies, which should be approx. (in kT) : 

|    total | inter_rot | inter_env |
|---------:|----------:|----------:|
| -271.185 | -27.041   | 3.608     |

It will also output average energies and CAPRI criteria metrics for the second halves of each replicate trajectory, which will differ due to random velocity initialization and the stochastic component of Langevin dynamics. But if you set `seed = 1` in the script, the values for the first replicate should be (in kT for energies, angstroms for position deviations):

|   total | inter_rot | inter_env | IRMSD | LRMSD | f_nat |
|--------:|----------:|----------:|-------|-------|-------|
| -286.78 | -81.37    | -16.21    | 2.43  | 3.25  | 0.68  |

 

