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
import os
import time

import torch
from PIL import Image

from pathlib import Path

from nemo.collections.multimodal.models.stable_diffusion.ldm.ddpm import MegatronLatentDiffusion
from nemo.collections.multimodal.parts.stable_diffusion.pipeline import pipeline
from nemo.collections.multimodal.parts.utils import setup_trainer_and_model_for_inference
from nemo.core.config import hydra_runner
import debug_layer_stats

from mlperf_logging.mllog import constants
from mlperf_logging_utils import mllogger, extract_step_from_ckpt_name


@hydra_runner()
def main(cfg):
    def model_cfg_modifier(model_cfg):
        model_cfg.precision = cfg.trainer.precision
        model_cfg.ckpt_path = None
        model_cfg.inductor = False
        model_cfg.unet_config.use_flash_attention = False
        model_cfg.unet_config.from_pretrained = None
        model_cfg.first_stage_config.from_pretrained = None

    global_id = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])

    # Avoids hunderds of warnings that happen when jobs wait for the main job
    if global_id != 0:
        time.sleep(5)

    torch.backends.cuda.matmul.allow_tf32 = True
    trainer, megatron_diffusion_model = setup_trainer_and_model_for_inference(
        model_provider=MegatronLatentDiffusion, cfg=cfg, model_cfg_modifier=model_cfg_modifier
    )

    model_ckpt = cfg.model.restore_from_path
    model_ckpt_name = Path(model_ckpt).name
    step_num = extract_step_from_ckpt_name(model_ckpt_name)

    mllogger.start(key=constants.EVAL_START, metadata={constants.STEP_NUM: step_num})
    trainer.strategy.barrier()

    model = megatron_diffusion_model.model
    model.cuda().eval()

    # Disable CUDA graph
    model.model.capture_cudagraph_iters = -1
    model.first_stage_model.capture_cudagraph_iters = -1
    model.cond_stage_model.capture_cudagraph_iters = -1

    rng = torch.Generator().manual_seed(cfg.infer.seed)

    # Read prompts from disk
    path = cfg.custom.prompts_dir
    prompts = []
    prompt_files = sorted(os.listdir(path))
    if cfg.custom.num_prompts is not None:
        prompt_files = prompt_files[: cfg.custom.num_prompts]

    for file in prompt_files:
        with open(os.path.join(path, file), "r") as f:
            prompts += f.readlines()

    sharded_prompts = prompts[global_id::world_size]
    sharded_prompt_files = prompt_files[global_id::world_size]

    print(f"Assigned {len(sharded_prompts)} prompts for this worker.")
    assert len(sharded_prompts) == len(sharded_prompt_files), f"{len(sharded_prompts)} != {len(sharded_prompt_files)}"
    cfg.infer.prompts = sharded_prompts
    trainer.strategy.barrier()

    debug_layer_stats.stat_report(model)

    output = pipeline(model, cfg, rng=rng)
    assert len(output) == len(sharded_prompt_files), f"{len(output)} != {len(sharded_prompt_files)}"

    os.makedirs(cfg.infer.out_path, exist_ok=True)
    for image, prompt_file in zip(output, sharded_prompt_files):
        if isinstance(image, list):
            assert len(image) == 1
            image = image[0]

        assert isinstance(image, Image.Image)
        output_image_name = Path(prompt_file).stem + ".png"
        image.save(os.path.join(cfg.infer.out_path, output_image_name))

    mllogger.end(key=constants.EVAL_STOP, metadata={constants.STEP_NUM: step_num})
    trainer.strategy.barrier()


if __name__ == "__main__":
    main()
