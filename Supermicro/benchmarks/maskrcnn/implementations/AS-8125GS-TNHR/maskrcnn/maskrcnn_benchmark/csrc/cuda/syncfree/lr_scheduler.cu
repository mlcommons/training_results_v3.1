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

__global__ void step_scheduler_loss_scaler(
  volatile int* noop_gmem,
  float* properties,
  float self_warmup_factor,
  int self_warmup_method,
  int self_warmup_iters,
  float self_gamma,
  float milestone1,
  float milestone2,
  float base_lr1,
  float base_lr2,
  int self_dynamic_loss_scale,
  float self_scale_factor,
  int self_scale_window
  )
{
  // properties:
  // 0 -> (O) lr[0]
  // 1 -> (O) lr[1]
  // 2 -> (IO) cur_scale
  // 3 -> (IO) last_overflow_iter
  // 4 -> (I) last_epoch == cur_iter
  // 5 -> (I) warmup_iters

  bool overflow = (*noop_gmem) ? true : false;
  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    //
    // update learning rate
    //

    // TMJ: last_epoch and warmup_iters are integer values, stored and used as floats.
    // This is ok because they are guaranteed to be < 2 million, which can be stored
    // exactly in a float.
    float last_epoch = properties[4];
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
    float lr1 = (base_lr1 - delta) * warmup_factor * powf(self_gamma, bisect_val);
    float lr2 = (base_lr2 - delta) * warmup_factor * powf(self_gamma, bisect_val);
    properties[0] = lr1;
    properties[1] = lr2;

    //
    // update loss scaler
    //
    if (self_dynamic_loss_scale) {
      float cur_scale = properties[2];
      int cur_iter = (int)last_epoch;
      int last_overflow_iter = (int)(properties[3]);
      if (overflow) {
	properties[2] = fmaxf(cur_scale/self_scale_factor, 1.0f);
	properties[3] = last_epoch;
      } else if ((cur_iter - last_overflow_iter) % self_scale_window == 0) {
        properties[2] = cur_scale * self_scale_factor;
      }
    }

    // update last_epoch, cur_iter
    properties[4] = last_epoch + 1.0f;
  }
}

void step_scheduler_loss_scaler_cuda(
  at::Tensor noop_flag,
  at::Tensor properties,
  float self_warmup_factor,
  int self_warmup_method,
  int self_warmup_iters,
  float self_gamma,
  float milestone1,
  float milestone2,
  float base_lr1,
  float base_lr2,
  int self_dynamic_loss_scale,
  float self_scale_factor,
  int self_scale_window
  )
{
  TORCH_CHECK(noop_flag.device() == properties.device(), "expected noop flag to be on the same device as properties");
  TORCH_CHECK(noop_flag.scalar_type() == at::ScalarType::Int, "expected noop flag to be integer tensor");
  TORCH_CHECK(properties.scalar_type() == at::ScalarType::Float, "expected properties to be float tensor");
  dim3 grid(1,1,1);
  dim3 block(1,1,1);
  step_scheduler_loss_scaler<<<grid, block>>>(
      noop_flag.data_ptr<int>(),
      properties.data_ptr<float>(),
      self_warmup_factor,
      self_warmup_method,
      self_warmup_iters,
      self_gamma,
      milestone1,
      milestone2,
      base_lr1,
      base_lr2,
      self_dynamic_loss_scale,
      self_scale_factor,
      self_scale_window
      );
}

}}}
