# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
* [MXNet 23.04-py3 NGC container](https://ngc.nvidia.com/catalog/containers/nvidia:mxnet)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) (multi-node)

# 2. Directions

## Steps to download and verify data

1. Clone the public DeepLearningExamples repository
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/MxNet/Classification/RN50v1.5
git checkout 81ee705868a11d6fe18c12d237abe4a08aab5fd6
```

2. Build a ResNet50 MXNet NGC container
```
docker build . -t nvidia_rn50_mx
```

3. Start an interactive session in the NGC container to run preprocessing
```
nvidia-docker run --rm -it --ipc=host -v <path/to/store/raw/&/processed/data>:/data nvidia_rn50_mx
```

4. Download and unpack the data
* Download **Training images (Task 1 &amp; 2)** and **Validation images (all tasks)** at http://image-net.org/challenges/LSVRC/2012/2012-downloads (require an account)
* Extract the training data:
    ```
    mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
    tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..
    ```
    
* Extract the validation data:
    ```
    mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val 
    tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    ```

5. Preprocess the dataset
```
./scripts/prepare_imagenet.sh <path/to/raw/imagenet> <path/to/save/preprocessed/data>
```

## Steps to run benchmark

## Steps to launch training on a single node

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to
run our container.

#### NVIDIA DGX H100 (single node)

Launch configuration and system-specific hyperparameters for the appropriate
NVIDIA DGX single node submission are in the `config_DGXH100.sh` script.

Steps required to launch single node training:

1. Build the container and push to a docker registry:
```
docker build --pull -t <docker/registry>/mlperf-nvidia:image_classification-mxnet .
docker push <docker/registry>/mlperf-nvidia:image_classification-mxnet
```
2. Launch the training:

```
source config_DGXH100.sh
CONT="<docker/registry>/mlperf-nvidia:image_classification-mxnet DATADIR=<path/to/data/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

## Alternative launch with nvidia-docker

When generating results for the official v3.0 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX H100 (single
node)](#nvidia-dgx-h100-single-node) explain how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
docker build --pull -t mlperf-nvidia:image_classification-mxnet .
source config_DGXH100.sh
CONT=mlperf-nvidia:image_classification-mxnet DATADIR=<path/to/data/dir> bash ./run_with_docker.sh
```

## Steps to launch training on multiple nodes

For multi-node training, we use Slurm for scheduling, and the Pyxis plugin to
Slurm to run our container, and correctly configure the environment for Pytorch
distributed execution.

### NVIDIA DGX H100 (multi node)

Launch configuration and system-specific hyperparameters for the NVIDIA DGX
H100 8 node submission is in the `config_DGXH100_8x8x50.sh` script.
Launch configuration and system-specific hyperparameters for the NVIDIA DGX
H100 submission on `NNODES` nodes are in the `config_DGXH100_<NNODES>x*.sh` scripts.

Steps required to launch multi node training on NVIDIA DGX H100

1. Build the docker container and push to a docker registry
```
docker build --pull -t <docker/registry>/mlperf-nvidia:image_classification-mxnet .
docker push <docker/registry>/mlperf-nvidia:image_classification-mxnet
```

2. Launch the training
```
source config_DGXH100_8x8x50.sh  # use appropriate config
CONT=<docker/registry>/mlperf-nvidia:image_classification-mxnet DATADIR=<path/to/data/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```
