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
#include <torch/extension.h>
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>
#include <algorithm>
#include <stdlib.h>
#include "cpu/vision.h"

__device__ void get_size(
	int w,
	int h,
	int size,
	int max_size,
	int& ow,
	int& oh
	)
{
    if (max_size > 0) {
	float min_original_size = static_cast<float>(min(w,h));
	float max_original_size = static_cast<float>(max(w,h));
	if (max_original_size / min_original_size * size > max_size) {
	    size = static_cast<int>(roundf(max_size * min_original_size / max_original_size));
	}
    }

    if ((w <= h && w == size) || (h <= w && h == size)) {
	ow = w;
	oh = h;
    } else if (w < h) {
	ow = size;
	oh = static_cast<int>(size*h/w);
    } else {
	oh = size;
	ow = static_cast<int>(size*w/h);
    }
}

__global__ void global_target_transforms_kernel(
	int* img_infos,
	int* indexes,
	float* bboxes_and_labels,
	float* dense_xy,
	int* min_size, // selected with torch.randperm
	int max_size,
	int8_t* hflip, // computed with (torch.rand > prob).to(dtype=torch.int8)
	int length
	)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < length) {
	// image transforms
	int* image_info = img_infos + index*4;
	int h = image_info[0];
	int w = image_info[1];
	int image_id = image_info[2];
        // get stretched size
	int ow,oh;
	get_size(w,h,min_size[index],max_size,ow,oh);
	float ratio_w = (float)ow / (float)w;
	float ratio_h = (float)oh / (float)h;
	// bbox transforms
	int box_offset = image_info[3];
	int num_boxes = (image_info[4+3] - box_offset) / 5;
	// write out image info after transformations.
	image_info[0] = oh;
	image_info[1] = ow;
	image_info[2] = image_id;
	image_info[3] = box_offset;
	for (int box = 0;  box < num_boxes;  ++box) {
	    // read original bbox
	    float* bbox = bboxes_and_labels + box_offset + box*4;
	    // resize
	    // bboxes are in xyxy format
	    float xmin = bbox[0] * ratio_w;
	    float xmax = bbox[2] * ratio_w;
	    float ymin = bbox[1] * ratio_h;
	    float ymax = bbox[3] * ratio_h;
	    // transpose
	    if (hflip[index]) {
		// FLIP_LEFT_RIGHT
		float tr_xmin = (float)ow - xmax - 1;
		float tr_xmax = (float)ow - xmin - 1;
		xmin = tr_xmin;
		xmax = tr_xmax;
	    }
	    // write out transformed bbox
	    bbox[0] = xmin;
	    bbox[2] = xmax;
	    bbox[1] = ymin;
	    bbox[3] = ymax;
	}
	// polygon transforms
	int header_size = indexes[0];
	int mask_offset = indexes[header_size+index];
	int num_masks = indexes[header_size+index+1] - mask_offset;
	for (int mask = 0;  mask < num_masks;  ++mask) {
	    int polygon_offset = indexes[mask_offset+mask];
	    int num_polygons = indexes[mask_offset+mask+1] - polygon_offset;
	    for (int poly = 0 ;  poly < num_polygons;  ++poly) {
		int sample_offset = indexes[polygon_offset+poly];
		int num_samples = (indexes[polygon_offset+poly+1] - sample_offset);
		float* polygon_values = dense_xy + sample_offset;
		for (int sample = 0;  sample < num_samples;  sample+=2) {
		    // resize
		    float x = polygon_values[sample  ] * ratio_w;
		    float y = polygon_values[sample+1] * ratio_h;
		    // transpose
		    if (hflip[index]) {
			// FLIP_LEFT_RIGHT
			x = (float)ow - x - 1;
		    }
		    // write out
		    polygon_values[sample  ] = x;
		    polygon_values[sample+1] = y;
		}
	    }
	}
    }
}

void global_target_transforms_cuda(
	at::Tensor img_infos,
	at::Tensor indexes,
	at::Tensor bboxes_and_labels,
	at::Tensor dense_xy,
	at::Tensor min_size_choice,
	at::Tensor hflip,
	int max_size
	)
{
    int length = img_infos.size(0)-1; // NB! +1 record added for num_masks calc
    auto stream = at::cuda::getCurrentCUDAStream();
    const int block_size = 256;
    int num_blocks = (length + block_size - 1) / block_size;
    global_target_transforms_kernel<<<num_blocks, block_size, 0, stream.stream()>>>(
	    img_infos.data_ptr<int>(),
	    indexes.data_ptr<int>(),
	    bboxes_and_labels.data_ptr<float>(),
	    dense_xy.data_ptr<float>(),
	    min_size_choice.data_ptr<int>(),
	    max_size,
	    hflip.data_ptr<int8_t>(),
	    length
	    );
}

