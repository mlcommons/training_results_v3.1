#!/bin/bash

# Try to get WORLD_SIZE and NODE_RANK from the MPI env
if [ -z ${OMPI_COMM_WORLD_SIZE} ]; then WORLD_SIZE=1; else WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}; fi
if [ -z ${OMPI_COMM_WORLD_RANK} ]; then NODE_RANK=0; else NODE_RANK=${OMPI_COMM_WORLD_RANK}; fi

ENV_ARGS="MASTER_PORT=$MASTER_PORT \
          MASTER_ADDR=$MASTER_ADDR \
          NODE_RANK=$NODE_RANK \
	  PT_HPU_POOL_MEM_ACQUIRE_PERC=99"


echo "MASTER_ADDR is $MASTER_ADDR"
echo "MASTER_PORT is $MASTER_PORT"

export PYTHONPATH=$MODEL_GARDEN_ROOT/internal/MLPERF/Habana/benchmarks/stable_diffusion:$PYTHONPATH
export PT_ENABLE_INT64_SUPPORT="true"


unset $(env | grep "OMPI_" | cut -d= -f1)
unset $(env | grep "PMIX_" | cut -d= -f1)

cmd="${ENV_ARGS} python3 -u $MODEL_GARDEN_ROOT/internal/MLPERF/Habana/benchmarks/stable_diffusion/main.py \
        lightning.trainer.num_nodes=8 \
        lightning.modelcheckpoint.params.every_n_train_steps=1000 \
        lightning.trainer.max_steps=5000 \
        lightning.trainer.val_check_interval=10000 \
        lightning.modelcheckpoint.params.save_last=False \
        data.params.train.params.urls=${DATASET_PATH} \
        model.params.hpu_graph=True \
        -m train \
        --ckpt ${BASE_CKPT} \
        -b $MODEL_GARDEN_ROOT/internal/MLPERF/Habana/benchmarks/stable_diffusion/configs/train_08x08x08.yaml \
        -l ${RESULTS_DIR} \
        --autocast \
        --warmup ${WARMUP_FILE} \
        --async_checkpoint \
        -n ${POSTFIX_LOG_DIR}"

echo ${cmd}
eval ${cmd}

path=${RESULTS_DIR}

str=${POSTFIX_LOG_DIR}

declare -A dir_cr_times

for dir in "$path"/*/;
do
  if [ -d "$dir" ] && [[ "$dir" == *"$str"* ]] ; then
    cr_time=$(stat -c %Y "$dir")
    dir_cr_times["$dir"]=$cr_time
  fi
done

sorted_dirs=($(for dir in "${!dir_cr_times[@]}"; do
  echo "$dir_cr_times[$dir]"
done | sort -n -r -k2 | awk '{print $1}'))

for dir in "${sorted_dirs[@]}"; do
  d_p=${dir%"]"}
  d_p=${d_p#"["}
  echo ${d_p}
  BACKUPDIR=${d_p}
  break
done


for n in {1..5};
do
    c1="'epoch=000000-step=00000"
    c2=$n
    c3="000.ckpt'"
    ckpt_file="${c1}${c2}${c3}"
    ckp1="checkpoints/"
    CKPT_FULL_PATH="${BACKUPDIR}${ckp1}${ckpt_file}"
    echo $CKPT_FULL_PATH
    cmd1="${ENV_ARGS} python3 -u $MODEL_GARDEN_ROOT/internal/MLPERF/Habana/benchmarks/stable_diffusion/main.py \
        lightning.trainer.num_nodes=8 \
        model.params.load_unet=True \
        data.params.validation.params.annotations_file=${ANNOTATION_FILE} \
        model.params.validation_config.fid.gt_path=${FID_GT_PATH} \
        -m validate \
        --ckpt ${CKPT_FULL_PATH} \
        -b $MODEL_GARDEN_ROOT/internal/MLPERF/Habana/benchmarks/stable_diffusion/configs/train_08x08x08.yaml \
        --current_validation_iter $n \
        --validation_iters 5"

    echo ${cmd1}
    eval ${cmd1}
done
