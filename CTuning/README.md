# MLPerf v3.1 CTuning Submission

CTuning submission for MLCommons Training v3.1 is using the [Nvidia MXNet training code](https://github.com/mlcommons/training_results_v2.1/tree/main/NVIDIA/benchmarks/resnet/implementations/mxnet-22.04) for ResNet benchmark. The code repository is forked [here](https://github.com/ctuning/training_results_v2.1/tree/main/NVIDIA/benchmarks/resnet/implementations/mxnet-22.04) with minor changes to automatically run the training benchmark with the required configuration. 

## Automation via CM

This MLPerf training submission is fully automated via [MLCommons CM](https://github.com/mlcommons/ck). To start with please [install CM](https://github.com/mlcommons/ck/blob/master/docs/installation.md).

## Get the raw dataset

The Imagenet train dataset is not publicly available. So, please download the tar file from [here](https://www.image-net.org/download.php) and register in CM automation. 

```
cm run script --tags=get,dataset,imagenet,train --input=/mount/ILSVRC2012_img_train.tar -j
```

The validation dataset will be automatically installed by CM as part of the workflow run.

## Run Command

The below command will prepare the training dataset and then do the training by launching the Nvidia docker container.

```
cm run script --tags=training,reproduce,mlperf,nvidia,_resnet --system_conf_name=config_A10_1x2x204 \
--version=r2.1 \
--adr.prepare-training-data.tags=_mxnet.22-08 \
--adr.nvidia-training-code.tags=_ctuning \
--results_dir=$HOME/results
```

After the training is done, MLPerf training log files can be seen under `$HOME/results` directory. These can be copied to the `results/a10x2-mxnet_22.08/resnet` folder and `benchmarks/resnet/implementations/mxnet` and `systems` folders can be created similar to this Github directory directory all under `submissions_training_v3.1/CTuning` directory.

## Check the submission

The validation of the submission can be verified by running the MLCommons Training submission checker as follows:

```
cm run script --tags=run,mlperf,training,submission,checker --input=`pwd`/submissions_training_v3.1/CTuning
```

This should print on console

```
0  https://github.com/mlcommons/training_results_v3.1/blob/master/CTuning/benchmarks
INFO - Running repository checks.
INFO - Running git-unfriendly file name checks.
INFO - Running large file checks.
INFO - ALL CHECKS PASSED.
```
