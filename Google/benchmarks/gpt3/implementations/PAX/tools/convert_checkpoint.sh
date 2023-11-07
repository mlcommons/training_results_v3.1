# setup: pip install abslpy; pip install tensorstore
CONVERT_CKPT=true
NODE_COUNT=16
BUCKET_NAME=mlperf-exp/tmp
EXP_NAME=C4SpmdGpt3AdamDataParallel16x16x16Int8

job_dir="gs://${BUCKET_NAME}/GPT3/${EXP_NAME}"
new_checkpoint_dir=${job_dir}/checkpoints

# preprocess ckpt
batch_size=$((128 * NODE_COUNT))
start_step=$((1536 * 4000 / batch_size))
initial_checkpoint_path=gs://mlperf-llm-public2/gpt3_spmd1x64x24_tpuv4-3072_v84_20221101/checkpoints/checkpoint_00004000
temp_dir=/tmp/gpt3_temp/v${RUN_NAME}_${TIMESTAMP}

num_stages=0

if [[ ! -z "${initial_checkpoint_path}" ]]; then
  mkdir -p "${temp_dir}"
  echo "${temp_dir}"

  step_str=$(printf "%08d" ${start_step})
  checkpoint_step_str=${new_checkpoint_dir}/checkpoint_${step_str}
  # Copy the checkpoint

  gsutil -m cp -r ${initial_checkpoint_path}/* "${checkpoint_step_str}"/

  # Update step counters
  mkdir  "${temp_dir}"/step
  rm -rf "${temp_dir}"/step

  echo "${temp_dir}"/step

  python3 tools/generate_ts_main.py \
  --value=${start_step} \
  --output_path="${temp_dir}"/step

  gsutil rm -r "${checkpoint_step_str}"/step
  gsutil -m cp -r "${temp_dir}"/step "${checkpoint_step_str}"/

  if [[ "${num_stages}" == 0 ]]; then
    str2="p#96#i-1"
  else
    str2="p#${num_stages}#sstage"
  fi

  for t in "no_prefix" ${str2}; do
  for i in $(seq 0 1 3); do
    # break

    var_name="opt_states_0.${t}_${i}.count"
    output_path="${temp_dir}/${var_name}"
    if [[ "${t}" == "no_prefix" ]]; then
      shape=""
    elif [[ "${num_stages}" == 0 ]]; then
      shape="96"
    else
      shape="${num_stages}"
    fi

    rm -rf "${output_path}"

    python3 tools/generate_ts_main.py \
      --value=${start_step} \
      --output_path="${output_path}" \
      --shape="${shape}"

    gsutil rm -r "${checkpoint_step_str}"/"${var_name}"
    gsutil -m cp -r "${output_path}" "${checkpoint_step_str}"/
  done; done
  touch commit_success.txt
  gsutil cp commit_success.txt "${checkpoint_step_str}"/
fi