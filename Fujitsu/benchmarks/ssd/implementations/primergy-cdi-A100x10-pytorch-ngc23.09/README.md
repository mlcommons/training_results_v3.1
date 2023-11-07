# Instructions to reproduce Fujitsu's submission results

## Setup
Build docker image according to the instruction in `../pytorch/README.md`.

## Download and build datasets
Setup datasets according to the instruction in `../pytorch/README.md`.

## Run bencmark program
Run benchmark program according to the instruction in `../pytorch/README.md`.
You can reproduce Fujitsu's results with `../pytorch/do_ssd_10gpu.sh` script. 
It loads `../pytorch/config_PG_10gpu_gbs240.sh` and you can modify benchmark behavior
by fixing the config file.

