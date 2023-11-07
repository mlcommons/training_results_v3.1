# MLPerf v3.1 NVIDIA Submission

This is a repository of NVIDIA's submission to the MLPerf v3.1 benchmark.  It
includes implementations of the benchmark code optimized for running on NVIDIA
DGX H100.  The reference implementations can be found elsewhere:
https://github.com/mlcommons/training.git

# v3.1 release

This readme was updated in October 2023, for the v3.1 round of MLPerf.

# Contents

Each implementation in the `benchmarks` subdirectory provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to build a container for the benchmark.
* Documentation on the dataset, model, and machine setup.

# Running Benchmarks

These benchmarks have been tested on the following machine configuration:

* An NVIDIA DGX SuperPOD&trade; with NVIDIA DGX H100 servers with 8x80GB H100 SXM
  gpus.
* The required software stack includes Slurm, with Enroot for running
  containers and the Slurm Pyxis plugin

Generally, a benchmark can be run with the following steps:

1. Follow the instructions in the README to download and format the input data and any required checkpoints.
2. Build the Dockerfile
3. Source the appropriate `config_*.sh` file.
4. `sbatch -N $DGXNNODES -t $WALLTIME run.sub`
