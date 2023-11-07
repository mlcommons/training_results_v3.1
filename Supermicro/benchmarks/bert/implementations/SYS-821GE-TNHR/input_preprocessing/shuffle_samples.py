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

from concurrent.futures import ProcessPoolExecutor
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

keys = ['input_ids', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids', 'next_sentence_labels']
for ifile_idx in tqdm(range(num_shards)):
    handle = h5py.File(f'{input_files[ifile_idx]}', 'r')
    ifile_handles[ifile_idx] = [np.asarray(handle[key][:]) for key in keys]
    handle.close()

def create_shard(i):
    with open(f'{args.output_hdf5}/shard_indices_{i:05}.lst','r') as f:
        ind=json.load(f)

    # max_seq_length = 512
    # max_predictions_per_seq = 76
    master_sample_idx=0
    hdf5_compression_method = False
    n_samples_in_this_shard = len(ind)

    h5_writer = h5py.File('{}/part_{:05d}_of_{:05d}.hdf'.format(args.output_hdf5, i, num_shards), 'w')
    input_ids = h5_writer.create_dataset('input_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    segment_ids = h5_writer.create_dataset('segment_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int8')), compression=hdf5_compression_method)
    masked_lm_positions = h5_writer.create_dataset('masked_lm_positions', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    masked_lm_ids = h5_writer.create_dataset('masked_lm_ids', (n_samples_in_this_shard,), dtype=h5py.vlen_dtype(np.dtype('int16')), compression=hdf5_compression_method)
    next_sentence_labels = h5_writer.create_dataset('next_sentence_labels', data=np.zeros(n_samples_in_this_shard, dtype="int8"), dtype='i1', compression=hdf5_compression_method)

    for o_sample_idx in tqdm(range(n_samples_in_this_shard), total=n_samples_in_this_shard):
        ifile, isample = ind[master_sample_idx]
        input_ids[o_sample_idx] = ifile_handles[ifile][0][isample]
        segment_ids[o_sample_idx] = ifile_handles[ifile][1][isample]
        masked_lm_positions[o_sample_idx] = ifile_handles[ifile][2][isample]
        masked_lm_ids[o_sample_idx] = ifile_handles[ifile][3][isample]
        next_sentence_labels[o_sample_idx] = ifile_handles[ifile][4][isample]
        master_sample_idx += 1

    h5_writer.flush()
    h5_writer.close()
    print(f"part_{i:05d}_of_{num_shards:05d}.hdf")

with ProcessPoolExecutor(max_workers=50) as executor:
    for partial_result in executor.map(create_shard, list(range(num_shards))):
        pass
