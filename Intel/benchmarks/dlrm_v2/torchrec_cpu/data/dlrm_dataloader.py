#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import List

import dlrm_dist_mlperf as ext_dist
from torch.utils.data import DataLoader

CAT_FEATURE_COUNT = 26

DEFAULT_CAT_NAMES = [
    'cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
    'cat_6', 'cat_7', 'cat_8', 'cat_9', 'cat_10', 'cat_11',
    'cat_12', 'cat_13', 'cat_14', 'cat_15', 'cat_16', 'cat_17',
    'cat_18', 'cat_19', 'cat_20', 'cat_21', 'cat_22', 'cat_23',
    'cat_24', 'cat_25'
]

DEFAULT_INT_NAMES = ['int_0', 'int_1', 'int_2', 'int_3', 'int_4', 'int_5', 'int_6', 'int_7', 'int_8', 'int_9', 'int_10', 'int_11', 'int_12']

#from torchrec.datasets.random import RandomRecDataset

# OSS import
# pyre-ignore[21]
# @manual=//ai_codesign/benchmarks/dlrm/torchrec_dlrm/data:multi_hot_criteo
from data.multi_hot_criteo import MultiHotCriteoIterDataPipe
from data.single_hot_criteo import InMemoryBinaryCriteoIterDataPipe
#except ImportError:
#    pass

# internal import
#try:
#    from .multi_hot_criteo import MultiHotCriteoIterDataPipe  # noqa F811
#    from data.single_hot_criteo import InMemoryBinaryCriteoIterDataPipe
#except ImportError:
#    pass

STAGES = ["train", "val", "test"]

def _get_random_dataloader(
    args: argparse.Namespace,
    stage: str,
) -> DataLoader:
    attr = f"limit_{stage}_batches"
    num_batches = getattr(args, attr)
    if stage in ["val", "test"] and args.test_batch_size is not None:
        batch_size = args.test_batch_size
    else:
        batch_size = args.batch_size
    return DataLoader(
        RandomRecDataset(
            keys=DEFAULT_CAT_NAMES,
            batch_size=batch_size,
            hash_size=args.num_embeddings,
            hash_sizes=args.num_embeddings_per_feature
            if hasattr(args, "num_embeddings_per_feature")
            else None,
            manual_seed=args.seed if hasattr(args, "seed") else None,
            ids_per_feature=1,
            num_dense=len(DEFAULT_INT_NAMES),
            num_batches=num_batches,
        ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=args.pin_memory,
        num_workers=0,
    )

def _get_in_memory_dataloader(
    args: argparse.Namespace,
    stage: str,
) -> DataLoader:
    if args.in_memory_binary_criteo_path is not None:
        dir_path = args.in_memory_binary_criteo_path
        dir_dense_labels_path = args.dense_labels_path
        sparse_part = "sparse.npy"
        datapipe = InMemoryBinaryCriteoIterDataPipe
    else:
        dir_path = args.synthetic_multi_hot_criteo_path
        dir_dense_labels_path = args.dense_labels_path
        sparse_part = "sparse_multi_hot.npz"
        datapipe = MultiHotCriteoIterDataPipe

    DAYS = args.days

    if stage == "train":
        stage_files: List[List[str]] = [
            [os.path.join(dir_dense_labels_path, f"day_{i}_dense.npy") for i in range(DAYS - 1)],
            [os.path.join(dir_path, f"day_{i}_{sparse_part}") for i in range(DAYS - 1)],
            [os.path.join(dir_dense_labels_path, f"day_{i}_labels.npy") for i in range(DAYS - 1)],
        ]
    elif stage in ["val", "test"]:
        stage_files: List[List[str]] = [
            [os.path.join(dir_dense_labels_path, f"day_{DAYS-1}_dense.npy")],
            [os.path.join(dir_path, f"day_{DAYS-1}_{sparse_part}")],
            [os.path.join(dir_dense_labels_path, f"day_{DAYS-1}_labels.npy")],
        ]
    if stage in ["val", "test"] and args.test_batch_size is not None:
        batch_size = args.test_batch_size
    else:
        batch_size = args.batch_size
    dataloader = DataLoader(
        datapipe(
            stage,
            *stage_files,  # pyre-ignore[6]
            batch_size=batch_size,
            rank=0, #ext_dist.my_rank,
            world_size=1, #ext_dist.my_size,
            #drop_last=args.drop_last_training_batch if stage == "train" else False,
            drop_last=True if stage == "train" else False,
            #drop_last=True,
            shuffle_batches=False, #args.shuffle_batches,
            shuffle_training_set=False, #args.shuffle_training_set,
            shuffle_training_set_random_seed=args.seed,
            mmap_mode=args.mmap_mode,
            hashes=args.num_embeddings_per_feature
            if args.num_embeddings is None
            else ([args.num_embeddings] * CAT_FEATURE_COUNT),
        ),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader


def get_dataloader(args: argparse.Namespace, backend: str, stage: str) -> DataLoader:
    """
    Gets desired dataloader from dlrm_main command line options. Currently, this
    function is able to return either a DataLoader wrapped around a RandomRecDataset or
    a Dataloader wrapped around an InMemoryBinaryCriteoIterDataPipe.

    Args:
        args (argparse.Namespace): Command line options supplied to dlrm_main.py's main
            function.
        backend (str): "nccl" or "gloo".
        stage (str): "train", "val", or "test".

    Returns:
        dataloader (DataLoader): PyTorch dataloader for the specified options.

    """
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    args.pin_memory = (
        (backend == "nccl") if not hasattr(args, "pin_memory") else args.pin_memory
    )

    if (
        args.in_memory_binary_criteo_path is None
        and args.synthetic_multi_hot_criteo_path is None
    ):
        return _get_random_dataloader(args, stage)
    else:
        return _get_in_memory_dataloader(args, stage)
