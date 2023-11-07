# 1. Problem

This benchmark represents a 3D medical image segmentation task using [2019 Kidney Tumor Segmentation Challenge](https://kits19.grand-challenge.org/) dataset called [KiTS19](https://github.com/neheller/kits19). The task is carried out using a [U-Net3D](https://arxiv.org/pdf/1606.06650.pdf) model variant based on the [No New-Net](https://arxiv.org/pdf/1809.10483.pdf) paper.

## Requirements
* [PyTorch 21.02-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) (data preprocessing)
* [MXNet 23.04-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-mxnet)
* Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot) (multi-node)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (single-node)

## Dataset

The data is stored in the [KiTS19 github repository](https://github.com/neheller/kits19).

## Publication/Attribution
Heller, Nicholas and Isensee, Fabian and Maier-Hein, Klaus H and Hou, Xiaoshuai and Xie, Chunmei and Li, Fengyi and Nan, Yang and Mu, Guangrui and Lin, Zhiyong and Han, Miofei and others.
"The state-of-the-art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 Challenge".
Medical Image Analysis, 101821, Elsevier (2020).

Heller, Nicholas and Sathianathen, Niranjan and Kalapara, Arveen and Walczak, Edward and Moore, Keenan and Kaluzniak, Heather and Rosenberg, Joel and Blake, Paul and Rengel, Zachary and Oestreich, Makinna and others.
"The kits19 challenge data: 300 kidney tumor cases with clinical context, ct semantic segmentations, and surgical outcomes".
arXiv preprint arXiv:1904.00445 (2019).

# 2. Directions

## Steps to download and verify data

1. Build and run the dataset preprocessing Docker container.
    
    ```bash
    docker build -t preprocessing -f Dockerfile_pyt .
    docker run --ipc=host -it --rm --runtime=nvidia -v DATADIR:/data preprocessing:latest 
    ```
   Where DATADIR is the target directory used to store the dataset after preprocessing.

   
2. Download and preprocess the data

    ```bash
    bash download_dataset.sh 
    ```

## Steps to launch training on a single node

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

## Steps to run benchmark with PRIMERGY CDI

Fujitsu has measured this benchmark results with the following steps:

1. Build docker image
This benchmark program is run in docker environment. You have to build docker image the following command.

```
docker build --pull -t mlperf-nvidia:image_segmentation-mxnet .
```

2. Set user password for sudo command
The execution script clears CPU cache before running benchmark program using sudo command. 
User password is required for the sudo command and should be stored password.txt file.

```
echo <user password here> > password.txt
```

3. Configure benchmark execution scripts
Benchmarks are run with the following benchmark execution scripts,
- `do_unet3d_8gpu.sh` and
- `do_unet3d_10gpu.sh`.

These scripts set the number of benchmark runs, the place of dataset, the place of benchmark logs and etc.,
and you should fix these scripts to adapt them to your environment.

4. Run benchmark execution script
Finaly you run benchmark execution script and get benchmark results.

```
bash do_unet3d_8gpu.sh
```


### Hyperparameter settings

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh`.

 
# 3. Quality

## Quality metric

The quality metric in this benchmark is mean (composite) DICE score for classes 1 (kidney) and 2 (kidney tumor). 
The metric is reported as `mean_dice` in the code.

## Quality target

The target `mean_dice` is 0.908.

## Evaluation frequency

The evaluation schedule is the following:
- for epochs 1 - 999: Do not evaluate
- for epochs >= 1000: Evaluate every 20 epochs

## Evaluation thoroughness

The validation dataset is composed of 42 volumes. They were pre-selected, and their IDs are stored in the `evaluation_cases.txt` file.
A valid score is obtained as an average `mean_dice` score across the whole 42 volumes. Please mind that a multi-worker training in popular frameworks is using so-called samplers to shard the data.
Such samplers tend to shard the data equally across all workers. For convenience, this is achieved by either truncating the dataset, so it is divisible by the number of workers,
or the "missing" data is duplicated using existing samples. This most likely will influence the final score - a valid evaluation is performed on exactly 42 volumes and each volume's score has a weight of 1/42 of the total sum of the scores.
