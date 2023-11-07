# coding=utf-8
# Copyright (c) 2019-2023 NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 MLBenchmark Group. All rights reserved.

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

from typing import OrderedDict
import h5py
import numpy as np
import argparse
import logging
from tqdm import tqdm
from itertools import repeat, cycle
import json
import glob
import random

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(
    description="Training data sharding for BERT.")
parser.add_argument(
    '--input_hdf5',
    type=str,
    default='hdf5',
    help='Input hdf5_file path')
parser.add_argument(
    '--output_hdf5',
    type=str,
    default='',
    help='Output hdf5_dir path')
args = parser.parse_args()


input_files = sorted(glob.glob(args.input_hdf5 + '/part_0*.hdf5', recursive=False))
num_shards = len(input_files)
logging.info('n_input_shards = {}'.format(num_shards))

ifile_handles={}
for ifile_idx in tqdm(range(num_shards)):
    handle = h5py.File(f'{input_files[ifile_idx]}', 'r')
    ifile_handles[ifile_idx]=handle['input_ids'].shape[0]
    handle.close()

ind=[(i, j) for idx in range(num_shards) for i, j in zip(cycle([idx]), list(range(ifile_handles[idx]))) ]
#The seed is fixed to 24, if you want to get a different shuffling please change
random.Random(24).shuffle(ind)

master_sample_idx = 0
for ofile_idx in tqdm(range(num_shards)):
    n_samples_in_this_shard = ifile_handles[ofile_idx]
    idxs=ind[master_sample_idx:master_sample_idx+n_samples_in_this_shard]
    with open(f'{args.output_hdf5}/shard_indices_{ofile_idx:05}.lst','w') as f:
        f.write(json.dumps(idxs))
    master_sample_idx += n_samples_in_this_shard

print(f"Shuffled: {master_sample_idx=}")
