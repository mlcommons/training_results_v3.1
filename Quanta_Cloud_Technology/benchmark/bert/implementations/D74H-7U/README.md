## Steps to launch training

### QuantaGrid D74H-7U

Launch configuration and system-specific hyperparameters for the QuantaGrid D74H-7U
submission are in the `../<implementation>/config_D74H-7U.sh` script.

Steps required to launch training on QuantaGrid D74H-7U.

1. Build the docker container and push to a docker registry

```
cd ../pytorch
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_D74H-7U.sh 
NEXP=10 CONT=<docker/registry:benchmark-tag> DATADIR_PHASE2=<path/to/packed_data> EVALDIR=<path/to/hdf5/eval_varlength> CHECKPOINTDIR_PHASE1=<path/to/ckpt/dir> ./run_with_docker.sh
