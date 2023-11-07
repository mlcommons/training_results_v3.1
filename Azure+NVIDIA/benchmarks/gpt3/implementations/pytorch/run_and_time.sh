set -e
[ "${DEBUG}" = "1" ] && set -x
: "${SEED:?SEED not set}"
: "${WALLTIME:=?WALLTIME not set}"
: "${LOCAL_RANK:=${SLURM_LOCALID}}"
: "${LOGGER:=""}"
: "${MULTI_NODE:=''}"
: "${OMPI_COMM_WORLD_LOCAL_RANK:=""}"
: "${SLURM_JOB_ID:=$RANDOM}"
: "${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK}}"
: "${SLURM_NODEID:=0}"
: "${SLURM_NTASKS_PER_NODE:=$DGXNGPU}"
: "${UNITTEST:=0}"
: "${NVTX_FLAG:=0}"
: "${TIME_TAGS:=0}"
: "${LOAD_CHECKPOINT:=""}"
: "${SHARE_RERUNS:=0}"
echo "LOAD_CHECKPOINT=${LOAD_CHECKPOINT}"
: "${USE_DIST_OPTIMIZER:=True}"
: "${CKPT_EVERY_VALIDATION:=False}"
: "${WALLTIME_EXIT_MINUTES:=0}"
: "${EXTRA_ARGS:=$@}"
echo RANK="${RANK}", LOCAL_RANK="${LOCAL_RANK}", MASTER_ADDR="${MASTER_ADDR}", MASTER_PORT="${MASTER_PORT}", WORLD_SIZE="${WORLD_SIZE}", UCX_NET_DEVICES="${UCX_NET_DEVICES}", NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}", NCCL_IB_HCA="${NCCL_IB_HCA}", NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY}", NCCL_IB_PCI_RELAXED_ORDERING="${NCCL_IB_PCI_RELAXED_ORDERING}", SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING="${SHARP_COLL_ENABLE_PCI_RELAXED_ORDERING}", UCX_VFS_ENABLE="${UCX_VFS_ENABLE}"
readonly node_rank="${SLURM_NODEID:-0}"
readonly local_rank="${LOCAL_RANK:=${SLURM_LOCALID:=${OMPI_COMM_WORLD_LOCAL_RANK:-}}}"
if [ "${NEMO_RESULTS_IN_TMP:-0}" -eq 1 ]; then
  readonly _explicit_log_dir=/tmp/${NEMO_RESULTS_SUBDIR:-""}
else
  readonly _explicit_log_dir=/results/${NEMO_RESULTS_SUBDIR:-""}
fi
if [ -n "${LOAD_CHECKPOINT}" ]; then
  if [ ${SHARE_RERUNS:-0} -eq 1 ] && [ -d "${_explicit_log_dir}/checkpoints" ] && [ -n "$(ls -A "${_explicit_log_dir}/checkpoints")" ]
  then
    [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ] && echo \
      "Detected a shared rerun." \
      "Resuming from previous run checkpoint stored in ${_explicit_log_dir}/checkpoints" \
      "instead of the initial checkpoint ${LOAD_CHECKPOINT}"
      unset LOAD_CHECKPOINT
  fi
else
    unset LOAD_CHECKPOINT
fi
if [ -n "${NEMO_RESULTS_SUBDIR}" ]; then
  EXTRA_ARGS+=" exp_manager.explicit_log_dir=\"${_explicit_log_dir}\""
fi
if [ "${TRAIN_ONLY:-0}" -eq 1 ]; then
  EXTRA_ARGS+=" data_prefix@model.data.data_prefix=train_only_c4"
elif [ "${USE_SYNTHETIC_DATASET:-0}" -eq 1 ]; then
  EXTRA_ARGS+=" data_prefix@model.data.data_prefix=synthetic model.data.data_impl=mock"
fi
[ "$INTERLEAVED_PIPELINE" == "0" ] && export INTERLEAVED_PIPELINE=null
if [ "$local_rank" -eq 0 ]
then
  echo "Hello from: " $(hostname)
fi
if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]
then
  echo "Run vars: id $SLURM_JOB_ID gpus $SLURM_NTASKS_PER_NODE mparams $MULTI_NODE"
  START=$(date +%s)
  START_FMT=$(date +%Y-%m-%d\ %r)
  echo "STARTING TIMING RUN AT ${START_FMT}"
fi
if [ "$USE_DIST_OPTIMIZER" = True ]; then
  EXTRA_ARGS+=" optim@model.optim=distributed_fused_adam"
fi
if [ "$CKPT_EVERY_VALIDATION" = True ]; then
  EXTRA_ARGS+=" exp_manager.checkpoint_callback_params.every_n_epochs=1"
  EXTRA_ARGS+=" exp_manager.checkpoint_callback_params.save_last=False"
fi
if [ "${WALLTIME_EXIT_MINUTES:-0}" -gt 0 ]; then
  [ "${NEXP:-1}" -gt 1 ] && echo "Warning: NEXP>1 and WALLTIME_EXIT_MINUTES>0 makes little sense (max_time for each run is set based on total WALLTIME)."
  
  max_time_minutes=$(( $WALLTIME - ${WALLTIME_EXIT_MINUTES}))
  [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ] && echo "Setting max_time to $max_time_minutes minutes"
  EXTRA_ARGS+=" +trainer.max_time=00:00:${max_time_minutes}:00"
fi
if [ "${PRINT_CONFIG_ONLY:-False}" = True ]; then
  EXTRA_ARGS+=" -c job --resolve"
fi
if [ ${NVTX_FLAG} -gt 0 ]; then
 NSYSCMD=" nsys profile --sample=none --cpuctxsw=none  --trace=cuda,nvtx  --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --output /results/language_model_pytorch_${DGXNNODES}x${DGXNGPU}x${MINIBATCHSIZE}_${DATESTAMP}_${SLURM_PROCID}_${SYNTH_DATA}.nsys-rep "
fi
[ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ] && echo "running LLM benchmark"
declare -a CMD
IB_BIND=''
if [[ "${SLURM_JOB_NUM_NODES:-1}" -gt 1 && "${ENABLE_IB_BINDING:-}" == "1" ]]; then
    IB_BIND='--ib=single'
fi
CPU_EXCLUSIVE=''
if [[ "${ENABLE_CPU_EXCLUSIVE:-1}" == "1" ]]; then
    CPU_EXCLUSIVE='--cpu=exclusive'
fi
if [[ -n "${SLURM_LOCALID-}" ]] && [[ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]]; then
    CMD=( 'bindpcie' ${CPU_EXCLUSIVE} ${IB_BIND} '--' ${NSYSCMD} 'python' '-u')
else
    CMD=( ${NSYSCMD} 'python' '-u' )
fi
if [ "$LOGGER" = "apiLog.sh" ];
then
  LOGGER="${LOGGER} -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
  if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ];
  then
    LOGGER=$LOGGER
    echo "Using LOGGER=${LOGGER}"
  else
    LOGGER=""
  fi
fi
[ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ] && echo "Extra args: $EXTRA_ARGS"
${LOGGER:-} ${CMD[@]} /workspace/llm/megatron_gpt_pretraining_custom.py \
	$EXTRA_ARGS \
	; ret_code=$?
set +x
sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi
if [ "$node_rank" -eq 0 ] && [ "$local_rank" -eq 0 ]
then
  END=$(date +%s)
  END_FMT=$(date +%Y-%m-%d\ %r)
  echo "ENDING TIMING RUN AT ${END_FMT}"
  RESULT=$(( ${END} - ${START} ))
  RESULT_NAME="large language model"
  echo "RESULT,${RESULT_NAME},${SEED},${RESULT},${USER},${START_FMT}"
fi
