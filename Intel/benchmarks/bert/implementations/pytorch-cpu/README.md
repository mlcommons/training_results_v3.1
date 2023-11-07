>_All workload code and resources used for the BERT workload may be found in the MLPerf Training v3.0 submission repository [here](https://github.com/mlcommons/training_results_v3.0/tree/main/Intel/benchmarks/bert/implementations/pytorch-cpu). The README content below is presented for easy reference._

# Download and prepare data

## Dataset

Please download and prepare data as demonstrated: https://github.com/mlcommons/training_results_v2.0/tree/main/NVIDIA/benchmarks/bert/implementations/pytorch#download-and-prepare-the-data. Don't use packed_data.

## Checkpoint
```
$python convert_checkpoint_tf2torch.py \
        --tf_checkpoint <path to model.ckpt-28252> \
        --bert_config_path <path to bert_config.json> \
        --output_checkpoint <path to pretraining pytorch_model.bin>
```

# Prepare enviroment

><picture><img alt="Warning" src="warning.svg"></picture><br>
>The environment under test used a publicly-sourced PyTorch v1.12.0 component which was subsequently discovered to contain a security vulnerability [CVE-2022-45907](https://nvd.nist.gov/vuln/detail/CVE-2022-45907). In compliance with MLCommons submission requirements, setup_conda.sh remains as-run for auditing purposes. However, **it is highly recommended that PyTorch v1.13.1 or later be specified for any other purpose.**

1) Create new conda env 
```
# It creates an env.sh script for activating conda env
$bash setup_conda.sh
$source <conda-install-dir>/bin/activate pt1120
```


2) Install Intel(R) Tensor Processing Primitives extension for PyTorch

Download tpp-pytorch-extension from https://github.com/libxsmm/tpp-pytorch-extension.git to local.  

Install the tpp-pytorch-extension:
```
$pushd <path-to-tpp>
$git submodule update --init
$python setup.py install
$popd
#install torch_ccl
$bash install_torch_ccl.sh
```

3) Install task specific requirements
><picture><img alt="Warning" src="warning.svg"></picture><br>
>The environment under test used a publicly-sourced Pillow v9.2.0 component which was subsequently discovered to contain a security vulnerability [CVE-2022-45199](https://nvd.nist.gov/vuln/detail/CVE-2022-45199). In compliance with MLCommons submission requirements, setup_conda.sh remains as-run for auditing purposes. However, **it is highly recommended that Pillow v9.3.0 or later be specified for any other purpose.**

```
$pip install -r requirements.txt
```

4) Create links to path to MLPerf BERT pretraining dataset and checkpoint:
```
$ln -s <path-to-dataset> ./mlperf_dataset
$ln -s <path-to-checkpoint> ./ref_checkpoint
```

# Run pretraining

1) Set system to performance mode
```
$echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
$sudo echo 0 > /proc/sys/kernel/numa_balancing
$sudo cpupower frequency-set -g performance
```

2) Clear cache (should be conducted before each workload run for accurate benchmarking)
```
$echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
$echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
$echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
$echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
$echo 1 > /proc/sys/vm/compact_memory; sleep 1
$echo 3 > /proc/sys/vm/drop_caches; sleep 1
```
For auditing purposes, the steps above require sudo access on each cluster node. Intel implements a practice of 'least privilege', requiring a dedicated script to execute with sudo privilege by standard users. This is accomplished by running 'clean.sh', which runs the encapsulated steps above as 'spr_perf_mode.sh' on each node:
```
CONFIG=<CONFIG_LABEL>
export DIR=<DIR_OF_CLEAR_CACHE_RESET_PERF_SCRIPTS>
bash clean.sh
mkdir -p ${CONFIG}
bash run_16node.sh
mv bert.log ${CONFIG}/
cp output.log ${CONFIG}/
```

3) Run Workload 
```
export LD_LIBRARY_PATH=<path-to-tpp>/build/temp.linux-x86_64-cpython-38/libxsmm/lib/:$LD_LIBRARY_PATH

#For closed division
#8 nodes
$bash run_8node.sh
#16 nodes
$bash run_16node.sh


#For open division
#8 nodes
$bash run_8node_open.sh
#16 nodes
$bash run_16node_open.sh
```

4) Log post-processing
To reflect the 'Clear cache' step above, the following line at the begining of each result_*.txt for log format compliance:
``` 
[0] :::MLLOG {"namespace": "", "time_ms": 0, "event_type": "POINT_IN_TIME", "key": "cache_clear", "value": true, "metadata": {"file": "clean.sh", "lineno": 5}}
```
Subsequently, user identifiers (such as user home path) were replaced where they occured in logs using basic 'sed' function similar to:
```
sed -i 's/<IDENTIFYING_CONTENT>\/script\.py/\.\/script\.py/g'
```
