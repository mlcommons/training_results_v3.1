## Setup from Source ##

### Prepare a conda env ###

   ```bash
   conda create -n dlrm_v2 python=3.9
   conda activate dlrm_v2
   ```

### Instal PyTorch ###

   ```bash
   pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
   ```

### Instlal Intel® Extension for PyTorch ###

   ```bash
   git clone https://github.com/intel/intel-extension-for-pytorch.git ipex
   cd ipex
   git checkout release/1.13
   git apply {$PWD}/../ipex.patch
   pip install -r requirements.txt
   git submodule sync
   git submodule update --init --recursive
   cd third_party/ideep/mkl-dnn/third_party/oneDNN/
   git fetch origin
   git checkout origin/rls-v2.7
   cd ../../../../..
   python setup.py install
   ```

### Install Intel® oneCCL Bindings for PyTorch ###
   ```bash
   git clone https://github.com/intel/torch-ccl.git
   cd torch-ccl
   git checkout ccl_torch1.13
   git submodule sync
   git submodule update --init --recursive
   cp /usr/lib64/libfabric.so.1.18.1 third_party/oneCCL/deps/ofi/lib/libfabric.so.1
   cp /usr/lib64/libfabric.so.1.18.1 third_party/oneCCL/deps/ofi/lib/libfabric.so
   cp /usr/lib64/libfabric/libpsm3-fi.so.1.21.0 third_party/oneCCL/deps/ofi/lib/prov/libpsm3-fi.so
   sed -ie 's/option(ENABLE_OFI_OOT_PROV "Enable OFI out-of-tree providers support" FALSE)/option(ENABLE_OFI_OOT_PROV "Enable OFI out-of-tree providers 
   support" TRUE)/g' third_party/oneCCL/CMakeLists.txt
   python setup.py install
   ```

### Install TorchRec ###
   ```bash
   #pip install torchrec==0.3.2
   #pip install fbgemm-gpu-cpu==0.3.2
   ```

### Prepare the final environment packages ###
   ```bash
   pip install tqdm torchmetrics==0.11.0
   pip install -e git+https://github.com/mlperf/logging#egg=mlperf-logging
   conda install intel-openmp==2023.1.0
   conda install -c conda-forge  gperftools
   pip install iopath pyre_extensions
   ```

### Prepare the dataset ###
   Follow the [MLCommons reference instructions](https://github.com/mlcommons/training/tree/00f04c57d589721aabce4618922780d29f73cf4e/recommendation_v2/torchrec_dlrm#create-the-synthetic-multi-hot-dataset) to create the 1TB Criteo dataset.

## Run DLRM_v2 training ##

### Setup Dataset ###

    ```shell
    export CONDA_ENV=dlrm_v2
    export SINGLEHOT_PATH=<path/to/singlehot/dataset>
    export MULTIHOT_PATH=<path/to/multihot/dataset>
    ```

### Setup environment ###

    ```bash
    export LOCAL_RANK=0
    export MASTER_ADDR=0
    export MASTER_PORT=1081
    export RANK=0
    export WORLD_SIZE=1
    ```

### Run single-hot ###

    ```bash
    vim run_mlperf.sh
    # change MULTIHOTDATA to 0
    ./run_mlperf.sh
    ```
    
### Run multi-hot ###

    ```bash
    vim rum_mlperf.sh
    # change MULTIHOTDATA to 1
    ./run_mlperf.sh
    ```
### Run multi-node test ###
    
    ```
    # change MULTIHOTDATA to 1 for multihot
    # change MULTIHOTDATA to 0 for onehot
    bash run_mlperf_multi_rank-64k-sota.sh <rank> <node> <batch_size> <run#>
    ```
