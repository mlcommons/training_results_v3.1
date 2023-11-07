/**
 * Copyright (c) 2018-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include "multi_tensor_apply.cuh"
#include "compat.h"

#include <assert.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512
#define ILP 4

namespace at { namespace native { namespace syncfree {

__global__ void step_optimizer_state(
  volatile int* noop_gmem,
  float* optimizer_state_ptr,
  float* base_lrs_ptr,
  float self_warmup_factor,
  int self_warmup_method,
  int self_warmup_iters,
  float self_gamma,
  float milestone1,
  float milestone2,
  int self_dynamic_loss_scale,
  float self_scale_factor,
  int self_scale_window
  )
{
  bool overflow = (*noop_gmem) ? true : false;

  // last_epoch is actually step counter. It was named last_epoch in LR scheduler.
  // Steps are counted in a float32 variable. This counter is an integer variable.
  // Integers less than ~8 million (2^23) can be represented exactly by float32.
  // Since we don't expect > 8 million steps, using a float32 as step counter is ok.
  float last_epoch = optimizer_state_ptr[0];
  float base_lr = base_lrs_ptr[threadIdx.x];
  float* group_state = optimizer_state_ptr + 3 + threadIdx.x * 5;

  // SGD initializes momentums on first run, but this is complicated
  // by the dynamic loss scaler. What we really want is first run
  // without gradient overflow, but we also need to count the steps
  // we've taken even if they had gradient overflows. We achieve this
  // by counting down instead of up until first time when first_run is
  // true and overflow is false.
  bool first_run = (last_epoch <= 0.0f) ? true : false;
  last_epoch = first_run ? -last_epoch : last_epoch;

  float warmup_iters = (float)self_warmup_iters;
  float warmup_factor = 1.0f;
  float delta = 0.0f;
  if (last_epoch < warmup_iters) {
    switch (self_warmup_method) {
      case 1: // constant
	warmup_factor = self_warmup_factor;
	break;
      case 2: // linear
	warmup_factor = last_epoch / warmup_iters;
	warmup_factor = self_warmup_factor * (1.0f - warmup_factor) + warmup_factor;
	break;
      case 3: //mlperf_linear
	delta = (warmup_iters - last_epoch) * self_warmup_factor;
	break;
      default:
	break;
    }
  }
  float bisect_val = (last_epoch <= milestone1) ? 0.0f : ((last_epoch < milestone2) ? 1.0f : 2.0f);
  float lr = (base_lr - delta) * warmup_factor * powf(self_gamma, bisect_val);
  group_state[4] = lr;
  __syncthreads();

  //
  // update loss scaler
  //
  if (self_dynamic_loss_scale && threadIdx.x == 0) {
    float cur_scale = optimizer_state_ptr[2];
    int cur_iter = (int)last_epoch;
    int last_overflow_iter = (int)(optimizer_state_ptr[1]);
    if (overflow) {
      optimizer_state_ptr[2] = fmaxf(cur_scale/self_scale_factor, 1.0f);
      optimizer_state_ptr[1] = last_epoch;
    } else if ((cur_iter - last_overflow_iter) % self_scale_window == 0 && cur_iter > 0) {
      optimizer_state_ptr[2] = cur_scale * self_scale_factor;
    }
    optimizer_state_ptr[0] = (first_run && overflow) ? -(last_epoch + 1.0f) : last_epoch + 1.0f;
  }
}

void step_optimizer_state_cuda(
  at::Tensor noop_flag,
  at::Tensor optimizer_state,
  at::Tensor base_lrs,
  float self_warmup_factor,
  int self_warmup_method,
  int self_warmup_iters,
  float self_gamma,
  float milestone1,
  float milestone2,
  int self_dynamic_loss_scale,
  float self_scale_factor,
  int self_scale_window
  )
{
  TORCH_CHECK(noop_flag.device() == optimizer_state.device(), "expected noop flag to be on the same device as optimizer_state");
  TORCH_CHECK(noop_flag.scalar_type() == at::ScalarType::Int, "expected noop flag to be integer tensor");
  TORCH_CHECK(optimizer_state.scalar_type() == at::ScalarType::Float, "expected optimizer_state to be float tensor");
  TORCH_CHECK(base_lrs.scalar_type() == at::ScalarType::Float, "expected base_lrs to be float tensor");
  dim3 grid(1,1,1);
  dim3 block(base_lrs.numel(),1,1);
  step_optimizer_state<<<grid, block>>>(
      noop_flag.data_ptr<int>(),
      optimizer_state.data_ptr<float>(),
      base_lrs.data_ptr<float>(),
      self_warmup_factor,
      self_warmup_method,
      self_warmup_iters,
      self_gamma,
      milestone1,
      milestone2,
      self_dynamic_loss_scale,
      self_scale_factor,
      self_scale_window
      );
}

}}}
