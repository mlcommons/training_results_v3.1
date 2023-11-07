# Instructions to reproduce Fujitsu's submission results

## Setup
Build docker image according to the instruction in `../pytorch/README.md`.

## Download and build datasets
Setup datasets according to the instruction in `../pytorch/README.md`.

## Run bencmark program
Run benchmark program according to the instruction in `../pytorch/README.md`.

You can reproduce Fujitsu's results with `../pytorch/do_ssd_8gpu.sh` script with
a bit modification to load `config_PG_8gpu_gbs320.sh` instead of `config_PG_8gpu_gbs256.sh`.
