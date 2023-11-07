wget -c https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O anaconda3.sh
chmod +x anaconda3.sh
./anaconda3.sh -b -p ./anaconda3
export WORKDIR=$PWD
export PATH=$WORKDIR/anaconda3/bin:$PATH
conda create -n dlrm_v2 python=3.9
conda update -n base -c defaults conda --yes
source activate dlrm

source /opt/rh/gcc-toolset-11/enable
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
git clone https://github.com/intel/intel-extension-for-pytorch.git ipex
cd ipex
git checkout release/1.13
pip install -r requirements.txt
git submodule sync
git submodule update --init --recursive
cd third_party/ideep/mkl-dnn/third_party/oneDNN/
git fetch origin
git checkout origin/rls-v2.7
cd ../../../../..
python setup.py install

pip install https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/torch_ccl/cpu/oneccl_bind_pt-1.13.0%2Bcpu-cp39-cp39-linux_x86_64.whl

pip install torchrec==0.3.2
pip install fbgemm-gpu-cpu==0.3.2

pip install tqdm torchmetrics==0.11.0
pip install -e git+https://github.com/mlperf/logging#egg=mlperf-logging
conda install intel-openmp==2023.1.0
conda install -c conda-forge  gperftools
