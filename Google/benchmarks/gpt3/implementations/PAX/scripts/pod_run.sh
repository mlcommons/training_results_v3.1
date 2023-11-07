echo "Adjust RTO and apply non cache copy"
first_line_res=$(ip route show | head -n 1)
if [[ "$(echo "$first_line_res" | grep "rto_min lock 5ms" | wc -l)" -eq 0 ]]; then
  # disable check given the error from double quotes added to ${first_line_res}
  # shellcheck disable=SC2086
  ip route change ${first_line_res} rto_min 5ms
fi
dev_name=$(echo "$first_line_res" | awk -F'[[:space:]]' '{ print $5 }')
echo "dev_name=${dev_name}"
ethtool -K "${dev_name}" tx-nocache-copy on

if [[ "${EXP_NAME}" =~ .*(int8|Int8).* ]]; then
  export ENABLE_INT8="true"
else
  export ENABLE_INT8="false"
fi
echo "ENABLE_INT8=${ENABLE_INT8}"

if [[ "${ENABLE_LOCAL_AQT}" == "true" ]]; then
  export ADDED_ARGS="--mlperf_gpt_local_aqt_factor=${NUM_SLICES} "
else
  export ADDED_ARGS=""
fi
echo "ENABLE_LOCAL_AQT=${ENABLE_LOCAL_AQT} and ADDED_ARGS=\"${ADDED_ARGS}\""
echo "ENABLE_LOCAL_AQT=${ENABLE_LOCAL_AQT} and ADDED_ARGS=\"${ADDED_ARGS}\""

# hack
# insert "from paxml.tasks.lm.params import quant_aqt_v2" at the beginning of the main script after the first line of title comment
# to allow flag arguments in quant_aqt_v2
sed -i '2s/^/from paxml.tasks.lm.params import quant_aqt_v2\n/' /usr/local/lib/python3.10/site-packages/paxml/main.py

TPU_LIBRARY_PATH=/tmp/mlperf_test_script/libtpu.so \
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump_file --xla_dump_hlo_as_text" TF_CPP_MIN_LOG_LEVEL=0 LIBTPU_INIT_ARGS="--xla_tpu_enable_megascale_barrier=true --xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_enable_async_collective_permute=true --xla_jf_rematerialization_percent_shared_memory_limit=97 --xla_tpu_decompose_all_gather_einsum=true --xla_tpu_spmd_threshold_for_allgather_cse=10 --xla_tpu_prefuse_self_attention=false --xla_tpu_rwb_fusion=false --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_dcn_max_overlap_estimation=32.0 --xla_tpu_data_parallel_opt_different_sized_ops=true --xla_vf_vmem_max_overlap_to_mem_size_async_copy_ratio=10 --megascale_enable_async_host_commands=true --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_enable_flash_attention=true --xla_tpu_scavenge_vmem_for_fusions=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_spmd_rng_bit_generator_unsafe=${ENABLE_INT8} --xla_tpu_dot_dot_fusion_duplicated=true --xla_tpu_use_repeated_instance_for_preferred_prefetch_time=true --xla_tpu_enforce_prefetch_fifo_order=true --xla_tpu_enable_aggressive_broadcast_priority_update=true --xla_tpu_impure_use_global_barrier_for_multi_collective_permute=false --xla_jf_crs_combiner_threshold_count=0" TPU_PREMAPPED_BUFFER_SIZE=4294967296 TPU_NAME=local TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 JAX_USE_PJRT_C_API_ON_TPU=1 python3 /usr/local/lib/python3.10/site-packages/paxml/main.py \
  --exp=tasks.lm.params.c4_mlperf_test."${EXP_NAME}" \
  --job_log_dir="gs://${GS_PREFIX}/job_log_dir" \
  --jax_profiler_port=9999 \
  --enable_checkpoint_saving=false \
  --tensorstore_use_ocdbt=false \
  --mode=train --eval_on_test "${ADDED_ARGS}" 2>&1 | tee /tmp/large_scale_multislice_test_log
bash /tmp/mlperf_test_script/parser_metrics.sh | tee -a /tmp/large_scale_multislice_test_log

if [[ ${MEGASCALE_SLICE_ID} == "0" ]]; then
  if [[ ${TPU_WORKER_ID} == "0" ]]; then
    gsutil -m cp /tmp/large_scale_multislice_test_log "gs://${GS_PREFIX}/job_log_dir/large_scale_multislice_test_log_0_0"
    gsutil -m cp -r /tmp/xla_dump_file "gs://${GS_PREFIX}/job_log_dir/xla"
  fi
fi
