#source config_PG_8gpu_gbs320.sh
source config_PG_8gpu_gbs256.sh

CONT=nvcr.io/nvdlfwea/mlperfv31/ssd:20230913.pytorch
DATADIR=/mnt/data4/work/ssd-openimages
LOGDIR=$(realpath ../logs-ssd-8gpu.gbs256)
TORCH_HOME=$(realpath ./torch-model-cache)
NEXP=1
num_of_run=5

export MLPERF_SUBMISSION_ORG=Fujitsu
export MLPERF_SUBMISSION_PLATFORM=PRIMERGY-CDI

for idx in $(seq 1 $num_of_run); do
    CONT=$CONT DATADIR=$DATADIR LOGDIR=$LOGDIR BACKBONE_DIR=$TORCH_HOME NEXP=$NEXP bash run_with_docker.sh
done
