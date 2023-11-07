# Benchmarks specific
export MAX_SYMBOL=300
export VAL_FREQUENCY=1
export LOG_FREQUENCY=1000
export DISABLE_FILE_LOGGING=true

## Opt flags
export FUSE_RELU_DROPOUT=true
export MULTI_TENSOR_EMA=true
export BATCH_EVAL_MODE=cg_unroll_pipeline
export APEX_LOSS=fp16
export APEX_JOINT=pack_w_relu_dropout
export AMP_LVL=2
export BUFFER_PREALLOC=true
export VECTORIZED_SA=true
export EMA_UPDATE_TYPE=fp16
export DIST_LAMB=true
export ENABLE_PREFETCH=true
export TOKENIZED_TRANSCRIPT=true
export VECTORIZED_SAMPLER=true
export DIST_SAMPLER=true
export FC_IMPL=apex_fused_dense
export AUDIO_RESAMPLING_DEVICE=cpu
export DELAY_ENCODER=false

