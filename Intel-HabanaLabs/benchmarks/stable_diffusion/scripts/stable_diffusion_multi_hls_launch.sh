#!/bin/bash

export MASTER_ADDR=$(head -n 1 /etc/mpi/hostfile)
export MASTER_PORT=56789
echo "MASTER_ADDR is $MASTER_ADDR"

export PYTHONPATH=/usr/lib/habanalabs:/root/repos/pytorch-training-tests:/root/repos/event_tests_plugin;
export PYTORCH_TESTS_ROOT=/root/repos/pytorch-training-tests;
export MODEL_GARDEN_ROOT=/root/repos/model_garden;
export AUTOMATION_ROOT=/root/repos/automation;
export HABANA_SOFTWARE_STACK=/root;
export SOFTWARE_DATA=/software/data;
export SOFTWARE_LFS_DATA=/mnt/weka/data;
export SOFTWARE_DATA=${SOFTWARE_DATA:-$DOCKER_HOME/software/data};
export SOFTWARE_LFS_DATA=${SOFTWARE_LFS_DATA:-$DOCKER_HOME/software/lfs/data};
export HABANA_SOFTWARE_STACK="/root/repos";
source ${HABANA_SOFTWARE_STACK}/pytorch-training-tests/tests/torch_training_tests/k8s_container/common/mpio_helper.sh;


source ${HABANA_SOFTWARE_STACK}/pytorch-training-tests/tests/torch_training_tests/k8s_container/common/mpio_helper.sh
cd $MODEL_GARDEN_ROOT/internal/MLPERF/Habana/benchmarks/stable_diffusion

#Install requirements on each card
run_per_ip python3 -m pip install -r $MODEL_GARDEN_ROOT/internal/MLPERF/Habana/benchmarks/stable_diffusion/requirements.txt

#Start Init and clear chache for MLLogger
run_per_ip bash $MODEL_GARDEN_ROOT/internal/MLPERF/Habana/benchmarks/stable_diffusion/scripts/run_init.sh

#Run Command
mpirun --allow-run-as-root  -N 1 -x MASTER_ADDR -x MASTER_PORT -x PYTHONPATH -x MODEL_GARDEN_ROOT \
	-x DATASET_PATH -x ANNOTATION_FILE -x FID_GT_PATH -x RESULTS_DIR -x POSTFIX_LOG_DIR -x WARMUP_FILE \
	-x BASE_CKPT --bind-to None -mca plm_rsh_no_tree_spawn 1 -mca btl_tcp_if_include eth0 --merge-stderr-to-stdout \
	$MODEL_GARDEN_ROOT/internal/MLPERF/Habana/benchmarks/stable_diffusion/scripts/stable_diffusion_multi_hls_train.sh
