# Instructions to reproduce Fujitsu's submission results

## Setup
Build docker image according to the instruction in `../mxnet/README.md`.

## Download and build datasets
Setup datasets according to the instruction in `../mxnet/README.md`.

## Run bencmark program
Run benchmark program according to the instruction in `../mxnet/README.md`.
You can reproduce Fujitsu's results with `../mxnet/do_unet3d_10gpu.sh` script. 
It loads `../mxnet/config_PG_10gpu.sh` and you can modify benchmark behavior
by fixing the config file.

