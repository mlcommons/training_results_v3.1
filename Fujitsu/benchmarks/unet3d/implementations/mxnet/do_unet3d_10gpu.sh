container_name=nvcr.io/nvdlfwea/mlperfv31/unet3d:20230926.mxnet
#container_name=unet3d_mxnet:latest
data_dir=/mnt/data4/work/3d-unet/data-dir
result_dir=$(realpath ../logs-unet3d-10gpu)

#source config_GX2460.sh # or config_DGX1_conv-dali_1x8x4.sh or config_DGXA100_conv-dali_1x8x7.sh
source config_PG_10gpu.sh
export MLPERF_SUBMISSION_ORG=Fujitsu
export MLPERF_SUBMISSION_PLATFORM=PRIMERGY-CDI
num_of_run=40

for idx in $(seq 1 $num_of_run); do
    CONT=$container_name DATADIR=$data_dir LOGDIR=$result_dir NEXP=1 ./run_with_docker.sh
done
