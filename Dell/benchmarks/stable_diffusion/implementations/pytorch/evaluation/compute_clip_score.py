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
python clip_script.py --captions_path /path/to/coco2014_val/captions \
                      --fid_images_path /path/to/synthetic_images \
                      --output_path /path/to/output/clip_scores.csv

1. `--captions_path`: The path to the real images captions directory. In this example,
   it is set to `/path/to/coco2014_val/captions`. This path should point to the
   directory containing the COCO 2014 validation dataset captions.

2. `--fid_images_path`: The path to the directory containing subfolders with synthetic
   images. In this example, it is set to `/path/to/synthetic_images`. Each subfolder
   should contain a set of synthetic images for which you want to compute CLIP scores
   against the captions from `--captions_path`.

3. `--output_path`: The path to the output CSV file where the CLIP scores will be saved.
   In this example, it is set to `/path/to/output/clip_scores.csv`. This file will
   contain a table with two columns: `cfg` and `clip_score`. The `cfg`
   column lists the names of the subfolders in `--fid_images_path`, and the
   `clip_score` column lists the corresponding average CLIP scores between the synthetic
   images in each subfolder and the captions from `--captions_path`.
"""

import argparse
import csv
import os
from glob import glob
import torch.distributed as dist

from pathlib import Path

import open_clip
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from mlperf_logging.mllog import constants
from mlperf_logging_utils import mllogger, extract_step_from_ckpt_name, extract_timestamp_from_ckpt_name
from functools import partial

from common import simple_init_distributed, barrier

print_flush = partial(print, flush=True)


class CLIPEncoder(nn.Module):
    def __init__(self, clip_version="ViT-B/32", pretrained="", cache_dir=None, device="cuda"):
        super().__init__()

        self.clip_version = clip_version
        if not pretrained:
            if self.clip_version == "ViT-H-14":
                self.pretrained = "laion2b_s32b_b79k"
            elif self.clip_version == "ViT-g-14":
                self.pretrained = "laion2b_s12b_b42k"
            else:
                self.pretrained = "openai"

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.clip_version, pretrained=self.pretrained, cache_dir=cache_dir
        )

        self.model.eval()
        self.model.to(device)

        self.device = device

    @torch.no_grad()
    def get_clip_score(self, text, image):
        if isinstance(image, str):  # filenmae
            image = Image.open(image)
        if isinstance(image, Image.Image):  # PIL Image
            image = self.preprocess(image).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        if not isinstance(text, (list, tuple)):
            text = [text]
        text = open_clip.tokenize(text).to(self.device)
        text_features = self.model.encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = image_features @ text_features.T

        return similarity


if __name__ == "__main__":
    print_flush("STARTING CLIP EVALUATION")

    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_path", default="/coco2014/coco2014_val_sampled_30k/captions/", type=str)
    parser.add_argument("--fid_images_path", default=None, type=str)
    parser.add_argument("--output_path", default="./clip_scores.csv", type=str)
    parser.add_argument("--clip_version", default="ViT-H-14", type=str)
    parser.add_argument("--cache_dir", required=True, type=str)
    args = parser.parse_args()

    global_id = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_id = int(os.environ["SLURM_LOCALID"])
    simple_init_distributed()

    device = f"cuda:{local_id}"
    torch.cuda.set_device(device)

    fid_images_path = args.fid_images_path
    captions_path = args.captions_path

    print_flush("Init CLIP Encoder..")
    encoder = CLIPEncoder(clip_version=args.clip_version, cache_dir=args.cache_dir)

    captions_dict = {}
    all_caption_files = sorted(glob(f"{captions_path}/*.txt"))
    for caption_file in all_caption_files:
        with open(caption_file, "r") as f:
            caption = f.read().strip()
            captions_dict[Path(caption_file).stem] = caption

    print_flush(f"Found {len(captions_dict)} captions!")

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
    clips = []
    timestamps = []
    for subfolder in subdirs_with_images:
        subfolder_path = os.path.join(args.fid_images_path, subfolder)
        images = sorted(glob(f"{subfolder_path}/*.png"))
        texts = []
        for image in images:
            stem = Path(image).stem
            if stem not in captions_dict:
                print_flush(f"Image {image} not found in captions_dict!")
                print_flush("Content of captions_dict:")
                for key, value in captions_dict.items():
                    print_flush(f"{key}: {value}")
                raise KeyError(f"{stem} not in captions_dict")
            texts.append(captions_dict[stem])

        print_flush(images[:5], texts[:5])
        print_flush(f"Number of images text pairs: {len(images)}")

        ave_sim = 0.0
        count = 0

        pbar = tqdm(texts, disable=global_id != 0, desc="Computing CLIP")
        for text, img in zip(pbar, images):
            sim = encoder.get_clip_score(text, img).item()
            ave_sim += sim
            count += 1

        ave_sim /= count
        clips.append(ave_sim)
        print_flush(f"The CLIP similarity for CFG {subfolder}: {ave_sim}")

        step = extract_step_from_ckpt_name(subfolder_path)
        steps.append(step)
        timestamp = extract_timestamp_from_ckpt_name(subfolder_path)
        timestamps.append(timestamp)
        mllogger.event(
            key=constants.EVAL_ACCURACY,
            value=ave_sim,
            metadata={
                constants.STEP_NUM: step,
                "metric": "CLIP",
                "path": subfolder_path,
            },
            unique=False,
        )

    # Get maximum number of CLIPs per process to avoid variable-sized gather
    clip_count = len(clips)
    max_clip_count = torch.tensor(clip_count, dtype=torch.int32, device=device)
    dist.all_reduce(max_clip_count, dist.ReduceOp.MAX)
    if global_id == 0:
        print_flush(f"Max CLIP count: {max_clip_count}")

    # Prep for gather, -1 represents an empty score
    step_tensor = torch.full([max_clip_count], -1, dtype=torch.int32, device=device)
    clip_tensor = torch.full([max_clip_count], -1, dtype=torch.float32, device=device)
    timestamp_tensor = torch.full([max_clip_count], -1, dtype=torch.int64, device=device)
    for i in range(clip_count):
        step_tensor[i] = steps[i]
        clip_tensor[i] = clips[i]
        timestamp_tensor[i] = timestamps[i]

    # Gather what should be written out to CSV file to rank 0
    step_tensor_list = [torch.zeros_like(step_tensor) for _ in range(world_size)] if global_id == 0 else None
    clip_tensor_list = [torch.zeros_like(clip_tensor) for _ in range(world_size)] if global_id == 0 else None
    timestamp_tensor_list = [torch.zeros_like(timestamp_tensor) for _ in range(world_size)] if global_id == 0 else None
    dist.gather(step_tensor, step_tensor_list, 0)
    dist.gather(clip_tensor, clip_tensor_list, 0)
    dist.gather(timestamp_tensor, timestamp_tensor_list, 0)

    # Rank 0 writes to output CSV file
    if global_id == 0:
        step_list = torch.cat(step_tensor_list).tolist()
        clip_list = torch.cat(clip_tensor_list).tolist()
        timestamp_list = torch.cat(timestamp_tensor_list).tolist()

        with open(args.output_path, "w", newline="") as csvfile:
            fieldnames = ["step", "clip", "timestamp"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, step in enumerate(step_list):
                if step != -1:
                    writer.writerow({"step": step, "clip": clip_list[i], "timestamp": timestamp_list[i]})
    barrier()
