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

#pragma once
#include "cpu/vision.h"
#ifndef _global_target_transforms_h_
#define _global_target_transforms_h_

void global_target_transforms(
	at::Tensor img_infos,
	at::Tensor indexes,
	at::Tensor bboxes_and_labels,
	at::Tensor dense_xy,
	at::Tensor min_size_choice,
	at::Tensor hflip,
	int max_size
	)
{
    global_target_transforms_cuda(img_infos, indexes, bboxes_and_labels, dense_xy, min_size_choice, hflip, max_size);
}

#endif

