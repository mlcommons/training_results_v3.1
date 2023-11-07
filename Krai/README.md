# MLPerf v3.1 Krai Submission

This is a repository of Krai's submission to the MLPerf v3.1 benchmark. It
includes optimized implementations of the benchmark code. The reference
implementations can be found elsewhere:
https://github.com/mlcommons/training.git

# v3.1 release

This readme was updated in Sept. 2023, for the v3.1 round of MLPerf.

# Contents

The implementation(s) in the `benchmarks` subdirectory provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to run the benchmark in a container.
* Documentation on the dataset, model, and machine setup.

# Running the Benchmark

These benchmarks have been tested on the following machine configuration:

* A server with 2x NVIDIA RTX A5000s (2x24GB gpus) using MxNet 23.08 with 0.25TB SSD for dataset storage.
* A server with 2x NVIDIA RTX A5000s (2x24GB gpus) using MxNet 22.08 with 0.25TB SSD for dataset storage.
* A server with 2x NVIDIA RTX A5000s (2x24GB gpus) using MxNet 22.04 with 0.25TB SSD for dataset storage.
* A server with 2x NVIDIA RTX A5000s (2x24GB gpus) and 1TB SSD for dataset storage.
* A server with 2x NVIDIA RTX A5000s (2x24GB gpus) and 4TB HDD for dataset storage.

Please see [here](./benchmarks) for the detail instructions in running the benchmark. 

