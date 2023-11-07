# Running Habana MLPerf™ Benchmarks

This directory provides instructions to reproduce Habana's results for [MLPerf Training v3.1](https://habana.ai/since-habanas-last-mlperf-submission/) **on 1 to 48 servers configurations with 8 Gaudi2 cards each.**

For more information on training deep learning models using Gaudi, refer to [developer.habana.ai](https://developer.habana.ai/resources/)

MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries. All rights reserved. Unauthorized use is strictly prohibited.

- [Running Habana MLPerf™ Benchmarks](#running-habana-mlperf-benchmarks)
  - [Setup](#setup)
    - [Prepare MLPerf Directory](#prepare-mlperf-directory)
    - [Build and Deploy HabanaLabs MLPerf Training 3.1 Container](#build-and-deploy-habanalabs-mlperf-training-31-container)
    - [Training Data for TensorFlow BERT](#training-data-for-tensorflow-bert)
    - [Training Data for PyTorch BERT](#training-data-for-pytorch-bert)
    - [Training Data for ResNet50](#training-data-for-resnet50)
    - [Training Data for GPT3-175B](#training-data-for-gpt3-175b)
    - [Training Data for Pytorch Stable Diffusion](#training-dataset-preparation-for-pytorch-stable-diffusion)
  - [Training BERT](#training-bert)
  - [Training ResNet50](#training-resnet50)
  - [Training GPT3-175B](#training-gpt3-175b)
  - [Training Pytorch Stable Diffusion](#training-pytorch-stable-diffusion)

## Setup

### Prepare MLPerf Directory

On each compute node, perform the following:

1. Follow the instructions provided in the [Gaudi Installation
Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the
environment including the `$PYTHON` environment variable.
The guide will walk you through the process of setting up your system to run the benchmarks on Gaudi.

2. Create directories for scratch and dataset folders:
    ```
    export MLPERF_ROOT=/path/to/mlperf/root
    export SCRATCH_DIR=$MLPERF_ROOT/scratch
    export DATASETS_DIR=$MLPERF_ROOT/datasets
    mkdir -p $MLPERF_DIR/Habana
    mkdir -p $SCRATCH_DIR
    mkdir -p $DATASETS_DIR
    ```

    **Note:** If training is to be conducted on multiple nodes, it is essential to place the $DATASETS_DIR on a shared filesystem that is accessible by all the nodes. This allows for dataset preparation to be performed only once in the `Training Data for <configuration>` sections, enabling all nodes to access the prepared dataset during training.

3. This README is located in `benchmarks` directory corresponding to Habana's submission.
Download this whole `benchmarks` folder along with all subfolders and copy it under `$MLPERF_DIR/Habana`

### Build and Deploy HabanaLabs MLPerf Training 3.1 Container

To build MLPerf training 3.1 container, perform the following:

1. Choose gaudi-docker-mlperf/ver3.1 release container, depending on the framework and model to be used:
   1. For Tensorflow Resnet, Bert models:
    ```
    export DOCKER_IMAGE=vault.habana.ai/gaudi-docker-mlperf/ver3.1/tensorflow-installer-tf-cpu-2.13.0:1.13.99-32
    ```
   2. For PyTorch Resnet, Bert models:
    ```
    export DOCKER_IMAGE=vault.habana.ai/gaudi-docker-mlperf/ver3.1/pytorch-installer-2.0.1:1.13.99-32
    ```
   3. For PyTorch GPT3 model:
    ```
    export DOCKER_IMAGE=vault.habana.ai/gaudi-docker-mlperf/ver3.1/pytorch-installer-2.0.1:1.13.99-41
    ```
    4. For PyTorch Stable Diffusion model:
    ```
    export DOCKER_IMAGE=vault.habana.ai/gaudi-docker-mlperf/ver3.1/pytorch-installer-2.0.1:1.12.0-381
    ```

2. Create `mlperf3.1` container by running the following command.

    ```bash
    export CONTAINER_NAME=mlperf3.1
    docker run --privileged --security-opt seccomp=unconfined \
      --name $CONTAINER_NAME -td                              \
      -v /dev:/dev                                            \
      --device=/dev:/dev                                      \
      -e LOG_LEVEL_ALL=6                                      \
      -v /sys/kernel/debug:/sys/kernel/debug                  \
      -v /tmp:/tmp                                            \
      -v $MLPERF_DIR:/root/MLPERF                             \
      -v $DATASETS_DIR:/root/datasets                         \
      -v $SCRATCH_DIR:/root/scratch                           \
      --cap-add=sys_nice --cap-add=SYS_PTRACE                 \
      --user root --workdir=/root --net=host                  \
      --ulimit memlock=-1:-1 ${DOCKER_IMAGE}
    ```

3. Start the docker.

    ```bash
    docker exec $CONTAINER_NAME bash -c "service ssh start"
    docker exec -it $CONTAINER_NAME bash
    ```

    **Note:** The following two steps are only necessary for training on multiple nodes.

4. In the docker, create `/root/shared/hosts` file that contains a list of all host IPs in the cluster. Add one IP per line. Below is an example for 4 nodes (32 devices).
    ```
    mkdir /root/shared
    echo '10.10.100.101' > /root/shared/hosts
    echo '10.10.100.102' >> /root/shared/hosts
    echo '10.10.100.103' >> /root/shared/hosts
    echo '10.10.100.104' >> /root/shared/hosts
    ```

5. SSH is used to spawn local and remote processes. In order to allow communication between machines it is required to provide a passwordless _ssh_ communication and set default port for connection. It has to be done on all of the machines:
    ```
    mkdir .ssh
    printf 'Host *\n    StrictHostKeyChecking no\nPort 3022' >> .ssh/config
    ```
    It also may be necessary to setup SSH keys and add them to `~/.ssh/authorized_keys`.

### Training Data for TensorFlow BERT

1. Log into mlperf3.1 TensorFlow container and install the requirements:
    <!-- DATASET download_mlperf_bert_tensorflow -->
    <!-- DATASET process_mlperf_bert_tensorflow -->
    ```bash
    export BERT_PATH=/root/MLPERF/Habana/benchmarks/bert/implementations/TensorFlow/nlp/bert
    cd $BERT_PATH
    pip install -r requirements.txt
    ```
    <!-- /DATASET process_mlperf_bert_tensorflow -->
    <!-- /DATASET download_mlperf_bert_tensorflow -->

2. Download the required files from Google drives.
    <!-- DATASET download_mlperf_bert_tensorflow -->
    ```bash
    export TENSORFLOW_BERT_DATA=/root/datasets/tensorflow_bert
    bash pretraining/prepare_dataset.sh \
      --data-path $TENSORFLOW_BERT_DATA \
      --only-download
    ```
    <!-- /DATASET download_mlperf_bert_tensorflow -->

    After completing this step, there should be a `$TENSORFLOW_BERT_DATA/input` folder containing the following files:
    ```
    bert_config.json
    model.ckpt-28252.data-00000-of-00001
    model.ckpt-28252.index
    model.ckpt-28252.meta
    results_text.tar.gz
    vocab.txt
    ```

3. Prepare the packed dataset by running the command below:
    <!-- DATASET process_mlperf_bert_tensorflow -->
    ```bash
    bash pretraining/prepare_dataset.sh \
      --scripts-path $BERT_PATH \
      --data-path $TENSORFLOW_BERT_DATA \
      --only-preprocessing \
      --jobs-limit 25
    ```
    <!-- /DATASET process_mlperf_bert_tensorflow -->

    This step will take multiple hours to complete.
    The exact time depends on the machine setup and the speed of storage that contains the dataset.
    The `--jobs-limit` option limits the number of pararell processes for converting and packing tfrecords.
    This step is resource consuming,
    and the machine running it must have a minimum of 32 CPUs and 755GB of RAM to ensure proper functioning.

4. `$TENSORFLOW_BERT_DATA` should now contain following folders:
    ```
    checkpoint
    eval_dataset
    input
    packed_data_500
    unpacked_data
    ```

    `input` folder can be removed if the preprocessing has been successfully completed.
    By default, TensorFlow BERT uses only packed data for training,
    as described in the scenario mentioned described [here](#training-for-tensorflow-bert).
    In such cases, the `unpacked_data` is unnecessary and can be deleted.

### Training Data for PyTorch BERT

#### Dataset Preparation

Log into mlperf3.1 PyTorch container and run:
<!-- DATASET download_mlperf_bert_pytorch -->
<!-- DATASET process_mlperf_bert_pytorch -->
```bash
cd /root/MLPERF/Habana/benchmarks/bert/implementations/PyTorch
pip install -r requirements.txt
export PYTORCH_BERT_DATA=/root/datasets/pytorch_bert
```
<!-- /DATASET process_mlperf_bert_pytorch -->
```bash
bash input_preprocessing/prepare_data.sh -o $PYTORCH_BERT_DATA
```
<!-- /DATASET download_mlperf_bert_pytorch -->

At this stage, ```$PYTORCH_BERT_DATA/phase1``` checkpoint and  ```$PYTORCH_BERT_DATA/hdf5/eval_varlength``` evaluation data are ready, while ```$PYTORCH_BERT_DATA/hdf5/training_4320/hdf5_4320_shards_uncompressed``` training data requires packing as described in the following section.

#### Training Data Packing

Once the training data is ready, pack it using a similar code as described in [GraphCore for v1.0 Submission](https://github.com/mlcommons/training_results_v1.0/tree/master/Graphcore/benchmarks/bert/implementations/popart/bert_data).

<!-- DATASET process_mlperf_bert_pytorch -->
```bash
mkdir $PYTORCH_BERT_DATA/packed
python3 pack_pretraining_data_pytorch.py \
    --input_dir=$PYTORCH_BERT_DATA/hdf5/training-4320/hdf5_4320_shards_uncompressed \
    --output_dir=$PYTORCH_BERT_DATA/packed \
    --max_predictions_per_seq=76
```
<!-- /DATASET process_mlperf_bert_pytorch -->

For further details, refer to [Packing: Towards 2x NLP BERT Acceleration](https://arxiv.org/abs/2107.02027).

### Training Data for ResNet50

The instructions for the ImageNet dataset is applicable for both PyTorch and TensorFlow ResNet50.

 1. Sign up with [image-net.org](http://image-net.org/download-images) and acquire the rights to download original images.
 2. Follow the link to the 2012 ILSVRC and download ILSVRC2012_img_val.tar and ILSVRC2012_img_train.tar.
 Place the files in the folder that will be mapped in mlperf3.1 container (for example, `$DATASETS_DIR`).
 3. Run the script below in mlperf3.1 container (PyTorch or TensorFlow) to unpack the dataset:

    ```
    bash /root/MLPERF/Habana/benchmarks/resnet/scripts/unpack_imagenet.sh \
        --train-archive /path/to/ILSVRC2012_img_train.tar \
        --validation-archive /path/to/ILSVRC2012_img_val.tar \
        --output-path /root/datasets/imagenet \
        --jobs-number 16
    ```

    The script unpacks training and validation packages in parallel.
    In addition, when upacking subarchives from ILSVRC2012_img_train.tar,
    `--jobs-number` defines number of pararell processes allocated for the task.
    Scripts runtime is dependent in large part on the data access speed of the storage where $DATASETS_DIR is located.

### Training Data for GPT3-175B

#### Dataset Preparation for GPT3-175B

Dataset preparation should be done in the following docker:

```
docker run --ipc=host -it -v $DATASETS_DIR:/root/datasets -v $MLPERF_DIR:/root/MLPERF nvcr.io/nvidia/pytorch:22.11-py3 bash
```

MLPerf GPT3 is trained using C4/en/3.0.1 dataset. It can be downloaded from https://huggingface.co/datasets/allenai/c4. Instruction is clear on how to select precisely the files for downloading.

```
apt-get update
apt-get install git-lfs
mkdir -p /root/datasets/gpt3
cd /root/datasets/gpt3
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/allenai/c4
cd c4
git lfs pull --include "en/*"
```

Out of all the files, only 256 will be required for training, and 8 for validation.
You can merge them into three .json.gz files using the following commands, which are taken from https://github.com/mlcommons/training/blob/master/large_language_model/megatron-lm/README.md.

```
# create softlinks to store each shard before merging
mkdir -p softlinks
for shard in {6..7}; do
  start=$((shard * 128))
  end=$((shard * 128 + 127))
  mkdir -p softlinks/en_$shard
  for ind in $(seq -f "%05g" $start $end); do
    ln -s ../../en/c4-train.${ind}-of-01024.json.gz softlinks/en_${shard}/c4-train.${ind}-of-01024.json.gz
  done
done

# merge
mkdir -p en_merge
for shard in {6..7}; do
  cat softlinks/en_${shard}/*gz > en_merge/c4-train.en_${shard}.json.gz
done
cat en/c4-validation.0000* > en_merge/c4-validation.json.gz
```

To tokenize the prepared files, you need to download the tokenizer model, vocab_c4_en_301_5Mexp2_spm.model, and the vocabulary file, vocab_c4_en_301_5Mexp2_spm.vocab, from the following location:
https://console.cloud.google.com/storage/browser/mlperf-llm-public2;tab=objects?prefix=&forceOnObjectsSortingFiltering=false.
Please note that registration is required to access these files. Tokenization can be performed using the following commands.
Please be aware that this conversion process may take several hours.

```
git clone https://github.com/NVIDIA/NeMo.git
cd NeMo && git checkout f3ad584b94170bc3ea197df29eb9ef9c96061730 && bash ./reinstall.sh && cd ..

mkdir -p preprocessed_c4_spm
for shard in {6..7}; do
python3 NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input en_merge/c4-train.en_${shard}.json.gz \
    --tokenizer-library sentencepiece \
    --tokenizer-model vocab_c4_en_301_5Mexp2_spm.model \
    --output-prefix preprocessed_c4_spm/c4_en_${shard}_c4_spm \
    --dataset-impl mmap \
    --workers 128
done

python3 NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input en_merge/c4-validation.json.gz \
    --tokenizer-library sentencepiece \
    --tokenizer-model vocab_c4_en_301_5Mexp2_spm.model \
    --output-prefix preprocessed_c4_spm/c4_en_validation_c4_spm \
    --dataset-impl mmap \
    --workers 128
```

The resulting files to be used during training are as follows:
* ```preprocessed_c4_spm/c4_en_6_c4_spm_text_document.bin```
* ```preprocessed_c4_spm/c4_en_6_c4_spm_text_document.idx```
* ```preprocessed_c4_spm/c4_en_7_c4_spm_text_document.bin```
* ```preprocessed_c4_spm/c4_en_7_c4_spm_text_document.idx```
* ```preprocessed_c4_spm/c4_en_validation_c4_spm_text_document.bin```
* ```preprocessed_c4_spm/c4_en_validation_c4_spm_text_document.idx```

In addition to the dataset, GPT3 implementation requires https://huggingface.co/gpt2/resolve/main/vocab.json and https://huggingface.co/gpt2/resolve/main/merges.txt files:

```
wget "https://huggingface.co/gpt2/resolve/main/vocab.json" -P preprocessed_c4_spm
wget "https://huggingface.co/gpt2/resolve/main/merges.txt" -P preprocessed_c4_spm
```

In order to exclude graph compilation time from Time To Train, you need to prepare a synthetic dataset for device warmup:
```
python3 /root/MLPERF/Habana/benchmarks/gpt3/Megatron-DeepSpeed/tools/create_synthetic_dataset.py \
    --valid_files_path preprocessed_c4_spm/c4_en_validation_c4_spm_text_document \
    --output_path preprocessed_c4_spm/
```

The commandline above will create synthetic files:
* ```preprocessed_c4_spm/synthetic_text_document.bin```
* ```preprocessed_c4_spm/synthetic_text_document.idx```

#### Checkpoint Preparation for GPT3-175B

Log into mlperf3.1 PyTorch container. Install DeepSpeed and other requirements:
```
pip install /root/MLPERF/Habana/benchmarks/gpt3/deepspeed-fork
pip install -r /root/MLPERF/Habana/benchmarks/gpt3/Megatron-DeepSpeed/requirements.txt
```

The checkpoint for MLPerf GPT3 in the paxml format can be downloaded from
[gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000](gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000).
The common_bf16.json can be downloaded from: https://github.com/ShriyaPalsamudram/training/tree/LLM-NVIDIA-reference-draft/large_language_model/megatron-lm/scripts.
At one stage, there will be a merged directory and a universal directory, each requiring 2 TB of disk space for 96L. Therefore, to complete all the steps, it is necessary to have over 4TB of free disk space.
Additionally, the machine must have a minimum of 32 CPUs and 755GB of RAM to ensure proper functioning.
Before the checkpoint can be used, it must be converted by following the steps below:

1. Convert the paxml checkpoint to Megatron distributed using /root/MLPERF/Habana/benchmarks/gpt3/tools/convert_checkpoint/convert_paxml_optimizer.py

    ```
    python3 /root/MLPERF/Habana/benchmarks/gpt3/tools/convert_checkpoint/convert_paxml_optimizer.py \
            --google_ckpts checkpoint_00004000/ \
            --output_dir megatron_merged_ckpt \
            --num_layers 96 \
            --params_file common_bf16.json \
            --pool 1
    ```

2. Convert Megatron merged checkpoint to DeepSpeed universal.

    To generate the mp-rank-files required in megatron_optim_merged_to_ds_universal_convert.py, the user needs to run GPT-3, which will generate these files based on the configuration used in the run.
    This can be obtained by running a single step of GPT-3 and saving the checkpoint.
    Please note that only this particular step of checkpoint peparation must be done using 8 HLS2 machines. The remaining steps can be performed on a CPU-only machine.
    Please make sure /root/shared/hosts file contains a list of 8 IPs for HLS2 machines and SSH communication is properly configured.
    For further details, refer to points 4 and 5 [here](#build-and-deploy-habanalabs-mlperf-training-31-container).
    Once the setup is ready, proceed to run the single step for GPT3 as follows:
    ```
    mkdir checkpoint_with_mp_rank_files
    bash /root/MLPERF/Habana/benchmarks/gpt3/run_gpt.sh --hosts /root/shared/hosts --data-dir /root/datasets/ --output-dir /root/scratch --num-nodes 8 --data-parallel-size 1 --start-from-ckpt false --save-checkpoints-dir checkpoint_with_mp_rank_files --exit-interval 1 --global-batch-size 2048
    ```

    Run megatron_optim_merged_to_ds_universal_convert.py to create the universal checkpoint:

    ```
    mkdir -p /root/datasets/gpt3/universal-checkpoint
    python3 /root/MLPERF/Habana/benchmarks/gpt3/tools/convert_checkpoint/megatron_optim_merged_to_ds_universal_convert.py \
        --o /root/datasets/gpt3/universal-checkpoint/ --ds-mp-rank-files-dir checkpoint_with_mp_rank_files --megatron-lm-merged-input-dir megatron_merged_ckpt \
        --tp 8 --pp 8 --nl 96 --iteration 3000 --global-batch-size 2048 --seq_length 2048 --lr-decay-samples 166809600 --lr-warmup-samples 407040 \
        --pool 64 --model-parallel-same-config False --update-only-mp-rank-files False
    ```

### Training Dataset Preparation for Pytorch Stable Diffusion

The instruction for preparing the dataset is based on original MLCommons instruction.
Please follow instructions under the following link:
https://github.com/mlcommons/training/tree/master/stable_diffusion#laion-400m


#### Generate the synthetic dataset for Stable Diffusion (warmup)

Uncompress one data tar file and keep it in the input directory path
Set environment variables for input and output path and run the below script to generate the synthetic data at the output directory

```
export DATASET_PATH_UNCOMPRESSED=<input_path>
export DATASET_PATH_OUTPUT=<output_path>
bash /root/MLPERF/Habana/benchmarks/stable_diffusion/scripts/prepare_synthetic_data.sh
```

After synthetic data preparation then copy it to the path used via WARMUP_FILE

```
export WARMUP_FILE=<output_path>/SD_synthetic_data_10001.tar
```

## Training BERT

### Training TensorFlow BERT

1. Inside the mlperf3.1 TensorFlow container, install BERT requirements.
    ```bash
    export BERT_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/bert/implementations
    pip install -r $BERT_IMPLEMENTATIONS/TensorFlow/nlp/bert/requirements.txt
    ```

2. Run the training.
    ```bash
    cd $BERT_IMPLEMENTATIONS/HLS-Gaudi2-TF
    ./launch_bert_hvd.sh --config defaults.cfg
    ```
    Training results will be stored in `/tmp/bert_pretrain/phase_2` folder.

### Training PyTorch BERT

1. Inside the mlperf3.1 PyTorch container, install BERT requirements.
    ```bash
    export BERT_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/bert/implementations
    pip install -r $BERT_IMPLEMENTATIONS/PyTorch/requirements.txt
    ```

2. Run the training.
    ```bash
    export PYTORCH_BERT_DATA=/root/datasets/pytorch_bert
    cd $BERT_IMPLEMENTATIONS/HLS-Gaudi2-PT
    ./launch_bert_pytorch.sh --data-dir $PYTORCH_BERT_DATA
    ```
    Training results will be stored in `/tmp/BERT_PRETRAINING/results/checkpoints` folder.


## Training ResNet50

### Training TensorFlow ResNet50

1. Inside the mlperf3.1 TensorFlow container, install Resnet50 requirements.
    ```bash
    export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
    pip install -r $RESNET_IMPLEMENTATIONS/TensorFlow/computer_vision/Resnets/resnet_keras/requirements.txt
    ```

2. Run the training.
    ```bash
    cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-TF
    ./launch_keras_resnet_hvd.sh --config $(pwd)/batch_256.cfg --jpeg-data-dir /root/datasets/imagenet --log_dir /tmp/resnet_log
    ```
    Training results will be stored in `/tmp/resnet_log` folder.

### Training PyTorch ResNet50

1. Inside the mlperf3.1 PyTorch container, install Resnet50 requirements.
    ```bash
    export RESNET_IMPLEMENTATIONS=/root/MLPERF/Habana/benchmarks/resnet/implementations
    pip install -r $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-PT/PyTorch/requirements.txt
    ```

2. Run the training.
    ```bash
    cd $RESNET_IMPLEMENTATIONS/HLS-Gaudi2-PT
    ./launch_resnet.sh --config batch_256.cfg --data-dir /root/datasets/imagenet
    ```
    Training results will be stored in `/tmp/resnet_log` folder.

## Training GPT3-175B

All the training steps for GPT3-175B should be performed in mlperf3.1 PyTorch container.

### Installing Requirements

The following requirements need to be installed on all machines participating in the training:
```
pip install /root/MLPERF/Habana/benchmarks/gpt3/deepspeed-fork
pip install -r /root/MLPERF/Habana/benchmarks/gpt3/Megatron-DeepSpeed/requirements.txt
```

### Run and time

The latest Intel-HabanaLabs's software supports 8-bit floating-point precision (FP8) training for GPT3 model and MLPerf3.1 submissions for GPT3 have been conducted using FP8 precision.
Running the GPT3 model requires multiple machines. For example, 32 HLS2 machines: `HLS-Gaudi2-N32-PT system` or 48 HLS2 machines `HLS-Gaudi2-N48-PT system`.

Please set the paths for the dataset and the universal checkpoint, which should be created during [setup phase](#training-data-for-gpt3-175b).
```
export DATASET_DIR=/root/datasets/gpt3/c4/preprocessed_c4_spm
export CHECKPOINT_DIR=/root/datasets/gpt3/universal-checkpoint
```

Please make sure /root/shared/hosts file contains a list of IPs for HLS2 machines, and that SSH communication is properly configured.
For further details, refer to points 4 and 5 [here](#build-and-deploy-habanalabs-mlperf-training-31-container).

#### Running GPT3 on HLS-Gaudi2-N32-PT System
```
bash /root/MLPERF/Habana/benchmarks/gpt3/run_gpt.sh --data-dir $DATASET_DIR/ --universal-ckpt-path $CHECKPOINT_DIR/ \
--hosts /root/shared/hosts --output-dir /root/scratch --num-nodes 32 --data-parallel-size 4 --save-checkpoints false --mllog-output-path /root/scratch/result.txt --train-samples 6782976 --use-fp8-transformer-engine --global-batch-size 2048 --micro-batch-size 2 --eval-interval 12 --device-warmup true --device-warmup-dataset-path $DATASET_DIR/synthetic_text_document
```

#### Running GPT3 on HLS-Gaudi2-N48-PT System
```
bash /root/MLPERF/Habana/benchmarks/gpt3/run_gpt.sh --data-dir $DATASET_DIR/ --universal-ckpt-path $CHECKPOINT_DIR/ \
--hosts /root/shared/hosts --output-dir /root/scratch --num-nodes 48 --data-parallel-size 8 --pipeline-model-parallel-size 6 --save-checkpoints false --mllog-output-path /root/scratch/result.txt --train-samples 6782976 --global-batch-size 2048 --micro-batch-size 2 --eval-interval 12 --device-warmup true --device-warmup-dataset-path $DATASET_DIR/synthetic_text_document --use-fp8-transformer-engine
```

Training results will be stored in `/root/scratch` folder.

The `--save-checkpoints` is set to `false` as 96l checkpoints take a lot of disc space. In order to save the checkpoint after the run or save it with some frequency, please use `--save-checkpoints true` and manipulate `--save-interval` parameter.
The script will start from universal checkpoint and train up to 312 steps or the time, when validation log perplexity is below 2.69. According to the convergence point of GPT3 on HLS system, it should approximately run for 288 steps in order to reach 2.69 validation log perplexity. To reduce number of steps, you can use `--exit-interval` parameter or reduce train samples by `--train-samples` parameter.

## Training Pytorch Stable Diffusion

#### Run the traning and validation steps
Following environment variables will be used to specify before training:

```
DATASET_PATH:= to the path where preprocessed data is located
ANNOTATION_FILE:= is the annotation file used for validation
FID_GT_PATH:= is the path for npz file used for inception
RESULTS_DIR:= to the path you want to save the results and checkpoint
POSTFIX_LOG_DIR:= postfix for logdir
WARMUP_FILE:= is the file used only in the warmup of the training
BASE_CKPT:= is the base checkpoint
```

```
For example:
export DATASET_PATH="/data/mlperf_models/stable_diffusion/training/laion-400m/webdataset-moments-filtered/{00000..00831}.tar"
export ANNOTATION_FILE="/data/val/val2014_30k.tsv"
export FID_GT_PATH="/data/val/val2014_30k_stats.npz"
export RESULTS_DIR="/tmp/"
export WARMUP_FILE="/data/warmup/00001.tar"
export BASE_CKPT="/data/checkpoint/512-base-ema.ckpt"
```

### Running Stable Difussion training on HLS-Gaudi2-N8-PT System

Create launcher to access all the available workers from the launcher so it can be executed on all available workers

```
cd <MLPERF_DIR>/benchmarks/stable_diffusion

DATASET_PATH=${DATASET_PATH} ANNOTATION_FILE=${ANNOTATION_FILE} FID_GT_PATH=${FID_GT_PATH}  \
   RESULTS_DIR=${RESULTS_DIR} POSTFIX_LOG_DIR=${POSTFIX_LOG_DIR} WARMUP_FILE=${WARMUP_FILE} \
   BASE_CKPT=${BASE_CKPT} /root/MLPERF/Habana/benchmarks/scripts/stable_diffusion_multi_hls_launch.sh
```

#### The script above (stable_diffusion_multi_hls_launch.sh) will follow below steps in sequence on each worker:

#### Step1: Initialization
```
bash /root/MLPERF/Habana/benchmarks/scripts/scripts/run_init.sh
```
#### Step2: Training
Each worker will have training command:

```
For example command will look like this on individual worker:
MASTER_PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} NODE_RANK={NODE_RANK} python3 -u root/MLPERF/Habana/benchmarks/stable_diffusion/main.py  \
   lightning.trainer.num_nodes=8 data.params.train.params.urls=${DATASET_PATH} lightning.modelcheckpoint.params.every_n_train_steps=1000 \
   lightning.trainer.max_steps=5000 lightning.trainer.val_check_interval=<Greater_than_max_steps_to_avoid_online_val>                    \
   lightning.modelcheckpoint.params.save_last=False model.params.hpu_graph=True -m train --ckpt {BASE_CKPT}                              \
   -b configs/train_08x08x08.yaml -l {RESULTS_DIR} --autocast --warmup {WARMUP_FILE} --async_checkpoint -n {POSTFIX_LOG_DIR}
```

#### Step3: Run offline validation in a loop of 5 consecutive checkpoints generated
```
For example:
MASTER_PORT=${MASTER_PORT} MASTER_ADDR=${MASTER_ADDR} NODE_RANK={NODE_RANK} python3 -u root/MLPERF/Habana/benchmarks/stable_diffusion/main.py  \
   lightning.trainer.num_nodes=8 data.params.validation.params.annotations_file=${ANNOTATION_FILE}                                       \
   model.params.validation_config.fid.gt_path=${FID_GT_PATH} model.params.load_unet=True -m validate                                     \
   --ckpt {RESULTS_DIR}/checkpoints/'epoch=000000-step=00000x000.ckpt' -b {BASE_CKPT} -b configs/train_08x08x08.yaml          \
   --current_validation_iter {Specify_Current_Iteration_number} --validation_iters {Specify_total_iteration_number}
```

