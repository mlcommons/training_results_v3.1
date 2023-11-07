## Steps to launch training

### QuantaGrid D54U-3U

Launch configuration and system-specific hyperparameters for the QuantaGrid D54U-3U
submission are in the `../<implementation>/config_D54U-3U.sh` script.

Steps required to launch training on QuantaGrid D54U-3U.

1. Build the docker container and push to a docker registry

```
cd ../pytorch
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_D54U-3U.sh 
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data> PKLPATH=<path/to/data/pkl/dir> COCOPYTDIR=<path/to/data/pyt/dir> ./run_with_docker.sh
