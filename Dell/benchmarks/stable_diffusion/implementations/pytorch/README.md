# 1. Problem

This benchmark represents image generation task that using a subset
of [LAION-400_MILLION](https://laion.ai/blog/laion-400-open-dataset/) dataset. The task is carried out using
Stable Diffusion model that is mostly based on the following
paper: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf).
The specific varianet used in this benchmark is Stable Diffusion v2.0 based on
the [Stability-AI](https://github.com/Stability-AI/StableDiffusion).

## Requirements

* [NeMo Multimodal Framework Early Access Container](https://developer.nvidia.com/nemo-framework-multimodal-early-access)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) (multi-node)

# 2. Directions

## Downloading the dataset

The benchmark employs two datasets:

1. Training: a subset of [laion-400m](https://laion.ai/blog/laion-400-open-dataset)
2. Validation: a subset of [coco-2014 validation](https://cocodataset.org/#download)

### Laion 400m

The benchmark uses a CC-BY licensed subset of the Laion400 dataset.

The LAION datasets comprise lists of URLs for original images, paired with the ALT text linked to those images. As
downloading millions of images from the internet is not a deterministic process and to ensure the replicability of the
benchmark results, hence we download it from MLCommons storage. The dataset can be downloaded using a script:

```
scripts/datasets/laion400m-filtered-download-moments.sh --output-dir /datasets/laion-400m/webdataset-moments-filtered
```

### COCO-2014

The COCO-2014-validation dataset consists of 40,504 images and 202,654 annotations. However, our benchmark uses only a
subset of 30,000 images and annotations chosen at random with a preset seed. It's not necessary to download the entire
COCO dataset as our focus is primarily on the labels (prompts) and the inception activation for the corresponding
images (used for the FID score).

We download this dataset from MLCommons storage using the following scripts:

```bash
scripts/datasets/coco2014-validation-download-prompts.sh --output-dir /datasets/coco2014
scripts/datasets/coco2014-validation-download-stats.sh --output-dir /datasets/coco2014
```

While the benchmark code can work with raw images, we use the preprocessed inception weights to save on
computational resources.

## Downloading the checkpoints

The benchmark utilizes several network architectures for both the training and validation processes:

1. **Stable Diffusion**: This component leverages StabilityAI's 512-base-ema.ckpt checkpoint from HuggingFace. While the
   checkpoint includes weights for the UNet, VAE, and OpenCLIP text embedder, the UNet weights are not used and are
   discarded when loading the weights. The checkpoint can be downloaded with the following command:

```bash
scripts/checkpoints/download_sd.sh --output-dir /checkpoints/sd
```

2. **Inception**: The Inception network is employed during validation to compute the Fréchet Inception Distance (FID)
   score. The necessary weights can be downloaded with the following command:

```bash
scripts/checkpoints/download_inception.sh --output-dir /checkpoints/inception
```

3. **OpenCLIP ViT-H-14 Model**: This model is utilized for the computation of the CLIP score. The required weights can
   be downloaded using the command:

```bash
scripts/checkpoints/download_clip.sh --output-dir /checkpoints/clip
```

The aforementioned scripts will handle both the download and integrity verification of the checkpoints.

## Steps to download and verify data

1. Build and run the dataset preprocessing Docker container.

 ```bash
 docker build -t "<docker/registry>/mlperf-nvidia:stable_diffusion-pyt"
 docker run --ipc=host -it --rm --runtime=nvidia -v DATADIR:/datasets -v CHECKPOINTS:/checkpoints "<docker/registry>/mlperf-nvidia:stable_diffusion-pyt"
 ```

Where

* `DATADIR` is the target directory used to store the datasets that will be downloaded
* `CHECKPOITS` is the target directory used to store the checkpoints that will be downloaded

2. Download and preprocess the data using scripts provided above

## Steps to launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

### NVIDIA DGX H100

### Steps to launch training on multiple nodes

For multi-node training, we use Slurm with the Pyxis extension, and Slurm's MPI
support to run our container.

1. Build the container and push to a docker registry:

```
docker build --pull -t <docker/registry>/mlperf-nvidia:stable_diffusion-pyt .
docker push <docker/registry>/mlperf-nvidia:stable_diffusion-pyt
```

2. Launch the training:

```
source config_DGXH100_08x08x32.sh # or any other config
export DATADIR="/datasets"
export CHECKPOINTS="/checkpoints"
export NEMOLOGS="/nemologs"
export LOGDIR="/logdir"

CONT="<docker/registry>/mlperf-nvidia:stable_diffusion-pyt" sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh` in form of
environment variables. There are multiple options, but the smallest recommended config is `DGXH100_08x08x32.sh`.

# 3. Quality

## Quality metric

The quality metric in this benchmark is FID score and CLIP score.

## Quality target

* The target FID is below or equal to 90.
* The target CLIP is above or equal to 0.15.

## Evaluation frequency

The evaluation schedule is the following:

- Evaluate every 512000 training samples

Evaluation time does not count into the total runtime. It can be performed after the training is finished.
