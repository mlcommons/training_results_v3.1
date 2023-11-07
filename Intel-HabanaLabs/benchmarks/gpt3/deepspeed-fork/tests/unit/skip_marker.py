# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

hpu_skip_tests = {
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[3-1024-51-16-3-True-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-128-128-2-3-True-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-2048-128-32-3-False-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-509-16-3-True-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-512-16-3-False-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-119-16-3-True-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-2560-128-40-3-False-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[64-1024-56-16-3-False-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-384-16-3-False-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1024-381-16-3-True-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[64-1024-24-16-3-False-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForward::test_forward[8-1536-128-24-3-False-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForwardSmallBatchSize::test_forward_with_small_bsz[8-3-1024-512-16-3-True-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/accelerators/test_accelerator_forward.py::TestCUDAForwardSmallBatchSize::test_forward_with_small_bsz[8-3-1024-512-16-3-False-False]":
    "Transformer Op not supported by HPU. SW-142420",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagrad::test_cpu_adagrad_opt_sparse_embedding[512-4096-16]":
    "CPU Adagrad not supported by HPU",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagrad::test_cpu_adagrad_opt[30000000]":
    "CPU Adagrad not supported by HPU",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagrad::test_cpu_adagrad_opt_sparse_embedding[4096-262144-16]":
    "CPU Adagrad not supported by HPU",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagrad::test_cpu_adagrad_opt[1048576]":
    "CPU Adagrad not supported by HPU",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagrad::test_cpu_adagrad_opt[64]":
    "CPU Adagrad not supported by HPU",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagrad::test_cpu_adagrad_opt[127]":
    "CPU Adagrad not supported by HPU",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagrad::test_cpu_adagrad_opt[1024]":
    "CPU Adagrad not supported by HPU",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagrad::test_cpu_adagrad_opt[22]":
    "CPU Adagrad not supported by HPU",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagrad::test_cpu_adagrad_opt_sparse_embedding[32-64-16]":
    "CPU Adagrad not supported by HPU",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagrad::test_cpu_adagrad_opt[55]":
    "CPU Adagrad not supported by HPU",
    "unit/ops/adagrad/test_cpu_adagrad.py::TestCPUAdagradGPUError::test_cpu_adagrad_gpu_error":
    "CPU Adagrad not supported by HPU",
}

g1_skip_tests = {
    "unit/runtime/half_precision/test_dynamic_loss_scale.py::TestUnfused::test_all_overflow":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_dynamic_loss_scale.py::TestUnfused::test_no_overflow":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_dynamic_loss_scale.py::TestUnfused::test_some_overflow":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_dynamic_loss_scale.py::TestFused::test_all_overflow":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_dynamic_loss_scale.py::TestFused::test_some_overflow":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_dynamic_loss_scale.py::TestFused::test_no_overflow":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamwFP16Basic::test":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyGrad::test[3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyGrad::test[2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyGrad::test[1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[False-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[False-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[True-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[True-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[True-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP16ZeroOneCycleCompatibility::test[False-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamwFP16EmptyGrad::test":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestAmp::test_adam_O2_empty_grad":
    "apex/amp is not installed",
    "unit/runtime/half_precision/test_fp16.py::TestAmp::test_lamb_basic":
    "apex/amp is not installed",
    "unit/runtime/half_precision/test_fp16.py::TestAmp::test_adam_O2":
    "apex/amp is not installed",
    "unit/runtime/half_precision/test_fp16.py::TestAmp::test_adam_basic":
    "apex/amp is not installed",
    "unit/runtime/half_precision/test_fp16.py::TestFP16AdamTypes::test[True-AdamW]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestFP16AdamTypes::test[False-AdamW]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestFP16AdamTypes::test[False-Adam]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestFP16AdamTypes::test[True-Adam]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZero3LazyScatter::test":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[FusedAdam-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[Adam-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[Adam-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[FusedAdam-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[FusedAdam-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[Adam-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestLambFP16::test__basic":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestLambFP16::test_empty_grad":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestAdamFP32EmptyGrad::test":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroAllowUntestedOptimizer::test[True-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroAllowUntestedOptimizer::test[False-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroAllowUntestedOptimizer::test[True-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroAllowUntestedOptimizer::test[True-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroAllowUntestedOptimizer::test[False-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroAllowUntestedOptimizer::test[False-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[9-True-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[9-True-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[10-False-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[10-True-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[10-True-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[9-True-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[9-False-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[10-False-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[9-False-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[10-True-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[9-False-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroStaticScale::test[10-False-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyPartition::test[True-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyPartition::test[False-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyPartition::test[True-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyPartition::test[False-1]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyPartition::test[True-3]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZeroEmptyPartition::test[False-2]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestZero2ReduceScatterOff::test":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestFP16OptimizerForMoE::test_fused_gradnorm":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestFP16OptimizerForMoE::test_lamb_gradnorm[True]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestFP16OptimizerForMoE::test_unfused_gradnorm":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestFP16OptimizerForMoE::test_lamb_gradnorm[False]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/half_precision/test_fp16.py::TestLambFP32GradClip::test":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/zero/test_zero_context.py::TestSerialContext::test_scatter_halftype":
    "FP16 datatype is not supported by Gaudi.",
    "unit/runtime/test_autocast.py::TestAutoCastDisable::test_missing_amp_autocast[True]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/inference/test_inference.py::TestAutoTensorParallelism::test[noCG-fp16-marian]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[EleutherAI/gpt-j-6B]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[facebook/opt-125m]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[EleutherAI/gpt-neo-125M]":
    "FP16 datatype is not supported by Gaudi.",
    "unit/inference/test_checkpoint_sharding.py::TestCheckpointShardinAutoTP::test[bigscience/bloom-560m]":
    "FP16 datatype is not supported by Gaudi.",
}
g2_skip_tests = {
    "unit/runtime/half_precision/test_fp16.py::TestAmp::test_adam_O2_empty_grad":
    "apex/amp is not installed",
    "unit/runtime/half_precision/test_fp16.py::TestAmp::test_lamb_basic":
    "apex/amp is not installed",
    "unit/runtime/half_precision/test_fp16.py::TestAmp::test_adam_O2":
    "apex/amp is not installed",
    "unit/runtime/half_precision/test_fp16.py::TestAmp::test_adam_basic":
    "apex/amp is not installed",
    "unit/runtime/half_precision/test_fp16.py::TestFP16OptimizerForMoE::test_fused_gradnorm":
    "Fused Adam not supported in HPU",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[FusedAdam-1]":
    "Fused Adam not supported in HPU",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[FusedAdam-3]":
    "Fused Adam not supported in HPU",
    "unit/runtime/half_precision/test_fp16.py::TestZeroSupportedClientOptimizer::test[FusedAdam-2]":
    "Fused Adam not supported in HPU",
}

gpu_skip_tests = {}
