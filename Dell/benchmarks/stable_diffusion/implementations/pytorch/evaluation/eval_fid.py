# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Example usage:
   python eval_fid.py \
     --coco_images_path /path/to/coco2014_val \
     --fid_images_path /path/to/synthetic_images \
     --output_path /path/to/output/fid_scores.csv

1. `--coco_images_path`: The path to the real images directory. In this example,
    it is set to `/path/to/coco2014_val`. This path should point to the
    directory containing the COCO 2014 validation dataset images, resized
    to 256x256 pixels.

2. `--fid_images_path`: The path to the directory containing subfolders
    with synthetic images. In this example, it is set to
    `/path/to/synthetic_images`. Each subfolder should contain a
    set of synthetic images for which you want to compute FID scores
    against the real images from `--coco_images_path`.

3. `--output_path`: The path to the output CSV file where the FID scores
    will be saved. In this example, it is set to
    `/path/to/output/fid_scores.csv`. This file will contain a table with
    two columns: `cfg` and `fid`. The `cfg` column lists the
    names of the subfolders in `--fid_images_path`, and the `fid` column
    lists the corresponding FID scores between the synthetic images in
    each subfolder and the real images from `--coco_images_path`.
"""

import argparse
import csv
import os
import torch
import torch.distributed as dist

from evaluation.compute_fid import compute_fid_data
from evaluation.fid_dataset import CustomDataset

from mlperf_logging.mllog import constants
from mlperf_logging_utils import mllogger, extract_step_from_ckpt_name, extract_timestamp_from_ckpt_name
from functools import partial
from common import simple_init_distributed, barrier

print_flush = partial(print, flush=True)

if __name__ == "__main__":
    print_flush("STARTING FID EVALUATION")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_images_path", required=True, type=str)
    parser.add_argument("--coco_activations_dir", required=True, type=str)
    parser.add_argument("--fid_images_path", required=True, type=str)
    parser.add_argument("--output_path", default="./fid_scores.csv", type=str)
    args = parser.parse_args()

    global_id = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_id = int(os.environ["SLURM_LOCALID"])
    simple_init_distributed()

    device = f"cuda:{local_id}"
    torch.cuda.set_device(device)

    # Set paths for synthetic images and real images
    fid_images_path = args.fid_images_path
    real_path = args.coco_images_path

    # Create dataset and data loader for real images
    # We don't load the images because we're using saved activations

    # real_dataset = CustomDataset(real_path)
    # loader_real = torch.utils.data.DataLoader(
    #     real_dataset, batch_size=32, num_workers=0, pin_memory=True, drop_last=False
    # )

    num_images_total = 0

    subdirs_with_images = []
    for subfolder in os.listdir(fid_images_path):
        subfolder_path = os.path.join(fid_images_path, subfolder)
        if os.path.isdir(subfolder_path):
            subdirs_with_images.append(subfolder_path)
    subdirs_with_images = sorted(subdirs_with_images)

    # Multi job sharding
    subdirs_with_images = subdirs_with_images[global_id::world_size]
    if subdirs_with_images:
        print_flush(f"Assigned subfolders in {fid_images_path}:")
        for subfolder in subdirs_with_images:
            print_flush(subfolder)
    else:
        print_flush(f"No subfolders assigned in {fid_images_path} (SKIP)")

    # Iterate through subfolders in fid_images_path
    steps = []
    fids = []
    timestamps = []
    for subfolder in subdirs_with_images:
        subfolder_path = os.path.join(fid_images_path, subfolder)
        # Create dataset and data loader for synthetic images in subfolder
        synthetic_dataset = CustomDataset(subfolder_path, target_size=256)
        num_images_total += len(synthetic_dataset)

        loader_synthetic = torch.utils.data.DataLoader(
            synthetic_dataset, batch_size=32, num_workers=0, pin_memory=True, drop_last=False
        )

        print_flush(f"Will try to load coco activations from {args.coco_activations_dir}")
        print_flush(os.listdir(args.coco_activations_dir))

        # Compute FID score between synthetic images in subfolder and real images
        fid = compute_fid_data(
            args.coco_activations_dir,
            None,
            loader_synthetic,
            key_a=0,
            key_b=0,
            sample_size=None,
            is_video=False,
            few_shot_video=False,
            network="tf_inception",
            interpolation_mode="bilinear",
        )
        fids.append(fid)

        print_flush(f"The FID score between {subfolder_path} and {real_path} is {fid}")

        # We extract step and timestamp from the path
        step = extract_step_from_ckpt_name(subfolder_path)
        steps.append(step)
        timestamp = extract_timestamp_from_ckpt_name(subfolder_path)
        timestamps.append(timestamp)
        mllogger.event(
            key=constants.EVAL_ACCURACY,
            value=fid,
            metadata={
                constants.STEP_NUM: step,
                "metric": "FID",
                "path": subfolder_path,
            },
            unique=False,
        )
    print_flush(f"FID: total number of evaluated images: {num_images_total}")
    barrier()

    # Get maximum number of FIDs per process to avoid variable-sized gather
    fid_count = len(fids)
    max_fid_count = torch.tensor(fid_count, dtype=torch.int32, device=device)
    dist.all_reduce(max_fid_count, dist.ReduceOp.MAX)
    if global_id == 0:
        print_flush(f"Max FID count: {max_fid_count}")

    # Prep for gather, -1 represents an empty score
    step_tensor = torch.full([max_fid_count], -1, dtype=torch.int32, device=device)
    fid_tensor = torch.full([max_fid_count], -1, dtype=torch.float32, device=device)
    timestamp_tensor = torch.full([max_fid_count], -1, dtype=torch.int64, device=device)
    for i in range(fid_count):
        step_tensor[i] = steps[i]
        fid_tensor[i] = fids[i]
        timestamp_tensor[i] = timestamps[i]

    # Gather what should be written out to CSV file to rank 0
    step_tensor_list = [torch.zeros_like(step_tensor) for _ in range(world_size)] if global_id == 0 else None
    fid_tensor_list = [torch.zeros_like(fid_tensor) for _ in range(world_size)] if global_id == 0 else None
    timestamp_tensor_list = [torch.zeros_like(timestamp_tensor) for _ in range(world_size)] if global_id == 0 else None
    dist.gather(step_tensor, step_tensor_list, 0)
    dist.gather(fid_tensor, fid_tensor_list, 0)
    dist.gather(timestamp_tensor, timestamp_tensor_list, 0)

    # Rank 0 writes to output CSV file
    if global_id == 0:
        step_list = torch.cat(step_tensor_list).tolist()
        fid_list = torch.cat(fid_tensor_list).tolist()
        timestamp_list = torch.cat(timestamp_tensor_list).tolist()

        with open(args.output_path, "w", newline="") as csvfile:
            fieldnames = ["step", "fid", "timestamp"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, step in enumerate(step_list):
                if step != -1:
                    writer.writerow({"step": step, "fid": fid_list[i], "timestamp": timestamp_list[i]})
    barrier()
