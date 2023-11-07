# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import utils
from torch.utils.data import Dataset
from engine import preprocessing, loss_preprocessing, compute_matched_idxs


def init_cache(model_ptr, dataset, device, args):
    # Create cached dataset with just one batch
    images, targets = [], []
    for i in range(args.batch_size):
        images.append(dataset[i][0])
        targets.append(dataset[i][1])

    images = list(image.to(device, non_blocking=True) for image in images)
    targets = {k: [dic[k].to(device, non_blocking=True) for dic in targets] for k in targets[0]}

    images, targets = preprocessing(images, targets, model_ptr, args.data_layout)

    with torch.cuda.amp.autocast(enabled=args.amp):
        targets['matched_idxs'] = compute_matched_idxs(targets['boxes'], model_ptr)

    return images, targets


def get_cached_dataset(model, dataset, device, args):
    cache_images, cache_targets = init_cache(model, dataset, device, args)
    virtual_dataset_len = int(args.train_sz / args.batch_size / utils.get_world_size())
    cached_dataset = CachedDataset(virtual_dataset_len, cache_images, cache_targets)
    cached_data_loader = torch.utils.data.DataLoader(cached_dataset, batch_size=None, batch_sampler=None,
                                                     num_workers=0, pin_memory=False)
    return cached_data_loader


class CachedDataset(Dataset):
    def __init__(self, virtual_dataset_len, cache_images, cache_targets):
        self.virtual_dataset_len = virtual_dataset_len
        self.cache_images = cache_images
        self.cache_targets = cache_targets

    def __len__(self):
        return self.virtual_dataset_len

    def __getitem__(self, idx):
        return self.cache_images, self.cache_targets
