>_All workload code and resources used for the RetinaNet workload may be found in the MLPerf Training v3.0 submission repository [here](https://github.com/mlcommons/training_results_v3.0/tree/main/Intel/benchmarks/ssd/implementations/pytorch-cpu). The README content below is presented for easy reference._

# Download and prepare data

## Dataset

Please download and prepare dataset as https://github.com/mlcommons/training/tree/master/single_stage_detector#download-dataset

## Pretrained backbone

Please download as https://github.com/mlcommons/training/tree/master/single_stage_detector#download-the-pretrained-backbone

# Running the model

## Prepare environment

### Requirements:
+ Install GCC11.2
+ Install Intel(R) Ethernet Fabric Suite FS Package [v11.4.1.0.22](https://www.intel.com/content/www/us/en/download/19816/intel-ethernet-fabric-suite-fs-package.html?wapkw=Intel%20Ethernet%20Fabric%20Suite)

### Setup Conda Environment and Build Dependencies
+ Download and install Anaconda3
  ```
  wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
  bash Anaconda3-2022.05-Linux-x86_64.sh
  ```
+ Setup conda environment to install requirements, and build the src code
  ```
  CUR_DIR=$(pwd)
  git clone <path/to/this/repo>
  cd <path/to/this/repo>/Intel/benchmarks/ssd/implementations/pytorch-cpu/
  bash prepare_env.sh
  conda activate retinanet-train
  ```

## Run Benchmark

### Multi-node run on 16 nodes:  

```
#1) Set system to performance mode
$sudo echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo
$sudo echo 0 > /proc/sys/kernel/numa_balancing
$sudo cpupower frequency-set -g performance

#2) Clear cache
$echo never  > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
$echo never  > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
$echo always > /sys/kernel/mm/transparent_hugepage/enabled; sleep 1
$echo always > /sys/kernel/mm/transparent_hugepage/defrag; sleep 1
$echo 1 > /proc/sys/vm/compact_memory; sleep 1
$echo 3 > /proc/sys/vm/drop_caches; sleep 1

#3) Run the bash file 
$bash run_16node.sh
```
