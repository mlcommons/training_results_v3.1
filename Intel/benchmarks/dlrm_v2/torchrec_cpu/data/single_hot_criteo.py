from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.data.datapipes as dp
from iopath.common.file_io import PathManager, PathManagerFactory
from pyre_extensions import none_throws
from torch.utils.data import IterableDataset

PATH_MANAGER_KEY="torchrec"

CAT_FEATURE_COUNT = 26

DEFAULT_CAT_NAMES = [
    'cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
    'cat_6', 'cat_7', 'cat_8', 'cat_9', 'cat_10', 'cat_11',
    'cat_12', 'cat_13', 'cat_14', 'cat_15', 'cat_16', 'cat_17',
    'cat_18', 'cat_19', 'cat_20', 'cat_21', 'cat_22', 'cat_23',
    'cat_24', 'cat_25'
]

DEFAULT_INT_NAMES = ['int_0', 'int_1', 'int_2', 'int_3', 'int_4', 'int_5', 'int_6', 'int_7', 'int_8', 'int_9', 'int_10', 'int_11', 'int_12']

class InMemoryBinaryCriteoIterDataPipe(IterableDataset):
    def __init__(
        self,
        stage: str,
        dense_paths: List[str],
        sparse_paths: List[str],
        labels_paths: List[str],
        batch_size: int,
        rank: int,
        world_size: int,
        drop_last: Optional[bool] = False,
        shuffle_batches: bool = False,
        shuffle_training_set: bool = False,
        shuffle_training_set_random_seed: int = 0,
        mmap_mode: bool = False,
        hashes: Optional[List[int]] = None,
        path_manager_key: str = "torchrec",
    ) -> None:
        self.stage = stage
        self.dense_paths = dense_paths
        self.sparse_paths = sparse_paths
        self.labels_paths = labels_paths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.shuffle_batches = shuffle_batches
        self.shuffle_training_set = shuffle_training_set
        np.random.seed(shuffle_training_set_random_seed)
        self.mmap_mode = mmap_mode
        self.hashes: np.ndarray = np.array(hashes).reshape((1, CAT_FEATURE_COUNT))
        self.path_manager_key = path_manager_key
        self.path_manager: PathManager = PathManagerFactory().get(path_manager_key)

        if shuffle_training_set and stage == "train":
            self._shuffle_and_load_data_for_rank()
            self.world_size = 1
            self.rank = 0
        else:
            m = "r" if mmap_mode else None
            self.dense_arrs: List[np.ndarray] = [
                np.load(f, mmap_mode=m) for f in self.dense_paths
            ]
            self.sparse_arrs: List[np.ndarray] = [
                np.load(f, mmap_mode=m) for f in self.sparse_paths
            ]
            self.labels_arrs: List[np.ndarray] = [
                np.load(f, mmap_mode=m) for f in self.labels_paths
            ]
        len_d0 = len(self.dense_arrs[0])
        second_half_start_index = int(len_d0 // 2 + len_d0 % 2)
        if stage == "val":
            self.dense_arrs[0] = self.dense_arrs[0][:second_half_start_index, :]
            self.sparse_arrs[0] = self.sparse_arrs[0][:second_half_start_index, :]
            self.labels_arrs[0] = self.labels_arrs[0][:second_half_start_index, :]
        elif stage == "test":
            self.dense_arrs[0] = self.dense_arrs[0][second_half_start_index:, :]
            self.sparse_arrs[0] = self.sparse_arrs[0][second_half_start_index:, :]
            self.labels_arrs[0] = self.labels_arrs[0][second_half_start_index:, :]
        if not self.mmap_mode and self.hashes is not None:
            for sparse_arr in self.sparse_arrs:
                sparse_arr %= self.hashes

        self.num_rows_per_file: List[int] = list(map(len, self.dense_arrs))
        total_rows = sum(self.num_rows_per_file)
        self.num_full_batches: int = (
            total_rows // batch_size // self.world_size * self.world_size
        )
        self.last_batch_sizes: np.ndarray = np.array(
            [0 for _ in range(self.world_size)]
        )
        remainder = total_rows % (self.world_size * batch_size)
        if not self.drop_last and 0 < remainder:
            if remainder < self.world_size:
                self.num_full_batches -= self.world_size
                self.last_batch_sizes += batch_size
            else:
                self.last_batch_sizes += remainder // self.world_size
            self.last_batch_sizes[: remainder % self.world_size] += 1

        self._num_ids_in_batch: int = CAT_FEATURE_COUNT * (batch_size + 1)
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.lengths: torch.Tensor = torch.ones(
            (self._num_ids_in_batch,), dtype=torch.int32
        )
        self.offsets: torch.Tensor = torch.arange(
            0, self._num_ids_in_batch + 1, dtype=torch.int32
        )
        self._num_ids_in_batch -= CAT_FEATURE_COUNT
        self.length_per_key: List[int] = CAT_FEATURE_COUNT * [batch_size]
        self.offset_per_key: List[int] = [
            batch_size * i for i in range(CAT_FEATURE_COUNT + 1)
        ]
        self.index_per_key: Dict[str, int] = {
            key: i for (i, key) in enumerate(self.keys)
        }

    def get_shape_from_npy(
        path: str, path_manager_key: str = PATH_MANAGER_KEY
    ) -> Tuple[int, ...]:
        path_manager = PathManagerFactory().get(path_manager_key)
        with path_manager.open(path, "rb") as fin:
            np.lib.format.read_magic(fin)
            shape, _order, _dtype = np.lib.format.read_array_header_1_0(fin)
            return shape

    def get_file_row_ranges_and_remainder(
        lengths: List[int],
        rank: int,
        world_size: int,
        start_row: int = 0,
        last_row: Optional[int] = None,
    ) -> Tuple[Dict[int, Tuple[int, int]], int]:
        if last_row is None:
            total_length = sum(lengths) - start_row
        else:
            total_length = last_row - start_row + 1

        rows_per_rank = total_length // world_size
        remainder = total_length % world_size
        rows_per_rank = np.array([rows_per_rank for _ in range(world_size)])
        rows_per_rank[:remainder] += 1
        rank_rows_bins_csr = np.cumsum([0] + list(rows_per_rank))
        rank_left_g = rank_rows_bins_csr[rank] + start_row
        rank_right_g = rank_rows_bins_csr[rank + 1] - 1 + start_row

        output = {}

        file_left_g, file_right_g = -1, -1
        for idx, length in enumerate(lengths):
            file_left_g = file_right_g + 1
            file_right_g = file_left_g + length - 1

            if rank_left_g <= file_right_g and rank_right_g >= file_left_g:
                overlap_left_g, overlap_right_g = max(rank_left_g, file_left_g), min(
                    rank_right_g, file_right_g
                )

                overlap_left_l = overlap_left_g - file_left_g
                overlap_right_l = overlap_right_g - file_left_g
                output[idx] = (overlap_left_l, overlap_right_l)

        return output, remainder

    def _load_data_for_rank(self) -> None:
        start_row, last_row = 0, None
        if self.stage in ["val", "test"]:
            samples_in_file = get_shape_from_npy(
                self.dense_paths[0], path_manager_key=self.path_manager_key
            )[0]
            start_row = 0
            dataset_len = int(np.ceil(samples_in_file / 2.0))
            if self.stage == "test":
                start_row = dataset_len
                dataset_len = samples_in_file - start_row
            last_row = start_row + dataset_len - 1

        row_ranges, remainder = get_file_row_ranges_and_remainder(
            lengths=[
                get_shape_from_npy(
                    path, path_manager_key=self.path_manager_key
                )[0]
                for path in self.dense_paths
            ],
            rank=self.rank,
            world_size=self.world_size,
            start_row=start_row,
            last_row=last_row,
        )
        self.remainder = remainder
        self.dense_arrs, self.sparse_arrs, self.labels_arrs = [], [], []
        for arrs, paths in zip(
            [self.dense_arrs, self.sparse_arrs, self.labels_arrs],
            [self.dense_paths, self.sparse_paths, self.labels_paths],
        ):
            for idx, (range_left, range_right) in row_ranges.items():
                arrs.append(
                    BinaryCriteoUtils.load_npy_range(
                        paths[idx],
                        range_left,
                        range_right - range_left + 1,
                        path_manager_key=self.path_manager_key,
                        mmap_mode=self.mmap_mode,
                    )
                )

    def _shuffle_and_load_data_for_rank(self) -> None:
        world_size = self.world_size
        rank = self.rank
        dense_arrs = [np.load(f, mmap_mode="r") for f in self.dense_paths]
        sparse_arrs = [np.load(f, mmap_mode="r") for f in self.sparse_paths]
        labels_arrs = [np.load(f, mmap_mode="r") for f in self.labels_paths]
        num_rows_per_file = list(map(len, dense_arrs))
        total_rows = sum(num_rows_per_file)
        permutation_arr = np.random.permutation(total_rows)
        self.remainder = total_rows % world_size
        rows_per_rank = total_rows // world_size
        rows_per_rank = np.array([rows_per_rank for _ in range(world_size)])
        rows_per_rank[: self.remainder] += 1
        rank_rows_bins = np.cumsum(rows_per_rank)
        rank_rows_bins_csr = np.cumsum([0] + list(rows_per_rank))

        rows = rows_per_rank[rank]
        d_sample, s_sample, l_sample = (
            dense_arrs[0][0],
            sparse_arrs[0][0],
            labels_arrs[0][0],
        )
        shuffled_dense_arr = np.empty((rows, len(d_sample)), d_sample.dtype)
        shuffled_sparse_arr = np.empty((rows, len(s_sample)), s_sample.dtype)
        shuffled_labels_arr = np.empty((rows, len(l_sample)), l_sample.dtype)

        day_rows_bins_csr = np.cumsum(np.array([0] + num_rows_per_file))
        for i in range(len(dense_arrs)):
            start = day_rows_bins_csr[i]
            end = day_rows_bins_csr[i + 1]
            indices_to_take = np.where(
                rank == np.digitize(permutation_arr[start:end], rank_rows_bins)
            )[0]
            output_indices = (
                permutation_arr[start + indices_to_take] - rank_rows_bins_csr[rank]
            )
            shuffled_dense_arr[output_indices] = dense_arrs[i][indices_to_take]
            shuffled_sparse_arr[output_indices] = sparse_arrs[i][indices_to_take]
            shuffled_labels_arr[output_indices] = labels_arrs[i][indices_to_take]
        self.dense_arrs = [shuffled_dense_arr]
        self.sparse_arrs = [shuffled_sparse_arr]
        self.labels_arrs = [shuffled_labels_arr]

    def _np_arrays_to_batch(
        self, dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray
    ):
        if self.shuffle_batches:
            shuffler = np.random.permutation(len(dense))
            dense = dense[shuffler]
            sparse = sparse[shuffler]
            labels = labels[shuffler]

        #batch_size = len(dense)
        #for i in range(len(sparse)):
        #    sparse[i] = torch.from_numpy(sparse[i]).long().reshape(-1)
        sparse=torch.from_numpy(sparse.transpose(1, 0).reshape(-1))
        dense = torch.from_numpy(dense.copy())
        #index = tuple(sparse)
        #offset = self.offsets[: CAT_FEATURE_COUNT * [batch_size] + 1]
        labels = torch.from_numpy(labels.reshape(-1).copy())
        return (dense, sparse, labels)

    def _padding(self, x_arrs, padding_batchsize, slice_, rows_to_get):
        x_shape = list(x_arrs.shape)
        x_shape[0] = padding_batchsize
        x = np.zeros(x_shape, dtype=x_arrs.dtype)
        x[:rows_to_get] = x_arrs[slice_, :]
        return x

    def __iter__(self):
        buffer: Optional[List[np.ndarray]] = None

        def append_to_buffer(
            dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray
    ) -> None:
            nonlocal buffer
            if buffer is None:
                buffer = [dense, sparse, labels]
            else:
                for idx, arr in enumerate([dense, sparse, labels]):
                    buffer[idx] = np.concatenate((buffer[idx], arr))

        file_idx = 0
        row_idx = 0
        batch_idx = 0
        buffer_row_count = 0
        cur_batch_size = (
            self.batch_size if self.num_full_batches > 0 else self.last_batch_sizes[0]
        )
        while (
            batch_idx
            < self.num_full_batches + (self.last_batch_sizes[0] > 0) * self.world_size
        ):
            if buffer_row_count == cur_batch_size or file_idx == len(self.dense_arrs):
                if batch_idx % self.world_size == self.rank:
                    yield self._np_arrays_to_batch(*none_throws(buffer))
                    buffer = None
                buffer_row_count = 0
                batch_idx += 1
                if 0 <= batch_idx - self.num_full_batches < self.world_size and (
                    self.last_batch_sizes[0] > 0
                ):
                    cur_batch_size = self.last_batch_sizes[
                        batch_idx - self.num_full_batches
                    ]
            else:
                rows_to_get = min(
                    cur_batch_size - buffer_row_count,
                    self.num_rows_per_file[file_idx] - row_idx,
                )
                buffer_row_count += rows_to_get
                slice_ = slice(row_idx, row_idx + rows_to_get)

                if batch_idx % self.world_size == self.rank:
                    if rows_to_get % 1024 != 0 and (self.stage != "train"):
                        padding_batchsize = (rows_to_get // 1024 + 1) * 1024
                        dense_inputs = self._padding(self.dense_arrs[file_idx], padding_batchsize, slice_, rows_to_get)
                        sparse_inputs = self._padding(self.sparse_arrs[file_idx], padding_batchsize, slice_, rows_to_get)
                        target_labels = self.labels_arrs[file_idx][slice_, :]
                    else:
                        dense_inputs = self.dense_arrs[file_idx][slice_, :]
                        sparse_inputs = self.sparse_arrs[file_idx][slice_, :]
                        target_labels = self.labels_arrs[file_idx][slice_, :]

                    if self.mmap_mode and self.hashes is not None:
                        sparse_inputs = sparse_inputs % self.hashes

                    append_to_buffer(
                        dense_inputs,
                        sparse_inputs,
                        target_labels,
                    )
                row_idx += rows_to_get

                if row_idx >= self.num_rows_per_file[file_idx]:
                    file_idx += 1
                    row_idx = 0

    def __len__(self) -> int:
        return self.num_full_batches // self.world_size + (self.last_batch_sizes[0] > 0)
