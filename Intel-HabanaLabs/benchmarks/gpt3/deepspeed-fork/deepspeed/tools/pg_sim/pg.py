# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import warnings
import torch
import torch.distributed as dist

from datetime import timedelta
from functools import partial
from torch.distributed.constants import default_pg_timeout
from torch.distributed.distributed_c10d import (GroupMember, BroadcastOptions, AllreduceOptions, ReduceOp)
from torch.distributed.distributed_c10d import STORE_BASED_BARRIER_PREFIX  # noqa
from deepspeed.accelerator import get_accelerator


class SimProcessGroup:
    BACKEND = "sim"
    DEFAULT_PG = None
    WORLD_SIZE = -1
    STORE = None

    def __init__(self, rank, world_size, timeout, backend):
        self.sim_rank = rank
        self.pg_world_size = world_size
        self.timeout = timeout
        self.backend = backend
        self.pg = None
        self.torch_ver_major = int(torch.__version__.split('.')[0])
        self.torch_ver_minor = int(torch.__version__.split('.')[1])

        assert self.torch_ver_major == 1, \
            f"Torch version major != 1 is not supported (version={torch.__version__})"
        assert self.torch_ver_minor >= 10, \
            f"Torch version < 1.10 is not supported (version={torch.__version__})"

        if self.torch_ver_minor < 13:
            warnings.warn(f"Torch version < 1.13 is not tested (version={torch.__version__})")

        # default is the first process group created
        if SimProcessGroup.DEFAULT_PG is None:
            SimProcessGroup.DEFAULT_PG = self

    @staticmethod
    def get_dist_group_count():
        return torch.distributed.distributed_c10d._group_count

    @classmethod
    def store_add_rest_of_world(cls, next_group):
        group = cls.get_dist_group_count() + (1 if next_group else 0)
        store_key = f"{STORE_BASED_BARRIER_PREFIX}:{group}"
        cls.STORE.add(store_key, cls.WORLD_SIZE - 1)

    def _create_pg(self):
        self.store_add_rest_of_world(next_group=False)
        pg = dist.new_group(ranks=[0], timeout=self.timeout, backend=self.backend, pg_options=None)
        return pg

    def post_create_sim_group(self):
        self.pg = self._create_pg()

    @classmethod
    def default_pg(cls):
        assert cls.DEFAULT_PG is not None
        return cls.DEFAULT_PG

    def size(self):
        return self.pg_world_size

    def rank(self):
        return self.sim_rank

    # ----------------------------------------------------
    # P2P
    #
    # P2P operations are simulated as all_reduce
    # ----------------------------------------------------
    class P2PRequestObject:
        """ Dummy p2p request object that is returned for p2p ops"""

        def __init__(self, src):
            self.src = src

        def wait(self):
            return

        def is_completed(self):
            return True

        def _source_rank(self):
            return self.src

    def _p2p_op(self, tensor_list, src=None):
        opts = AllreduceOptions()
        if self.torch_ver_minor > 10:
            opts.reduceOp = ReduceOp.SUM
        self.pg.allreduce(tensor_list, opts).wait()
        src = src if src is not None else self.sim_rank
        return SimProcessGroup.P2PRequestObject(src=src)

    def send(self, tensor_list, _group_dst_rank, _tag):
        return self._p2p_op(tensor_list)

    def recv_anysource(self, tensor_list, _tag):
        return self._p2p_op(tensor_list)

    def recv(self, tensor_list, src, _tag):
        return self._p2p_op(tensor_list, src=src)

    # ----------------------------------------------------
    # Collectives
    #
    # For some collectives, it is required to shrink the
    # input/output tensors_list to 1-element (world_size=1).
    # also, need to make all other members of tensors_list to depend
    # on the first element - to prevent incorrect graph signaling.
    # The logic of shrink and then copy is handled by:
    # - _adjust_tensors_list_to_ws1
    # - _copy_data_from_tensor_to_tensor_list
    # ----------------------------------------------------
    @staticmethod
    def _to_device(tensors, device):
        if isinstance(tensors, dict):
            return {k: SimProcessGroup._to_device(v, device) for k, v in tensors.items()}
        elif isinstance(tensors, list):
            return [SimProcessGroup._to_device(v, device) for v in tensors]
        elif isinstance(tensors, torch.Tensor):
            return tensors.to(device)
        else:
            assert False, 'Unsupported tensors type'

    def broadcast(self, tensors, opts):
        """ ignore opts.rootRank and override to be the source """
        opts.rootRank = self.sim_rank
        tensors = self._to_device(tensors, get_accelerator().current_device_name())
        return self.pg.broadcast(tensors, opts)

    def allreduce(self, tensors, opts):
        return self.pg.allreduce(tensors, opts)

    def allreduce_coalesced(self, tensors, opts):
        return self.pg.allreduce_coalesced(tensors, opts)

    def reduce(self, tensors, opts):
        if opts.rootRank == self.sim_rank:
            return self.pg.reduce(tensors, opts)

        broadcast_opts = BroadcastOptions()
        broadcast_opts.rootRank = self.sim_rank
        broadcast_opts.rootTensor = opts.rootTensor
        return self.pg.broadcast(tensors, broadcast_opts)

    def _adjust_tensors_list_to_ws1(self, tensors_list):
        """ receives list of lists of tensors and returns lists
            of list-size-1 to match the world_size=1
        """
        world1_tensors_list = []
        for i, tensors in enumerate(tensors_list):
            world1_tensors_list.append(tensors[self.sim_rank:self.sim_rank + 1])
        return world1_tensors_list

    @staticmethod
    def _copy_data_from_tensor_to_tensor_list(source_tensors, tensors_list):
        """ copy data from source tensors to all tensors in tensor list """
        for i, tensors in enumerate(tensors_list):
            for t in tensors:
                t.data[:] = source_tensors[i][0].data[:]

    def allgather(self, tensors_list, input_tensors, *kwargs):
        world1_tensors_list = self._adjust_tensors_list_to_ws1(tensors_list)
        handle = self.pg.allgather(world1_tensors_list, input_tensors, *kwargs)
        self._copy_data_from_tensor_to_tensor_list(world1_tensors_list, tensors_list)
        return handle

    def gather(self, output_tensors, input_tensors, opts):
        if opts.rootRank == self.sim_rank:
            world1_tensors_list = self._adjust_tensors_list_to_ws1(output_tensors)
            handle = self.pg.gather(world1_tensors_list, input_tensors, opts)
            self._copy_data_from_tensor_to_tensor_list(world1_tensors_list, output_tensors)
            return handle

        broadcast_opts = BroadcastOptions()
        broadcast_opts.rootRank = self.sim_rank
        return self.pg.broadcast(input_tensors, broadcast_opts)

    def scatter(self, output_tensors, input_tensors, opts):
        if opts.rootRank == self.sim_rank:
            world1_tensors_list = self._adjust_tensors_list_to_ws1(input_tensors)
            handle = self.pg.scatter(output_tensors, world1_tensors_list, opts)
            self._copy_data_from_tensor_to_tensor_list(world1_tensors_list, input_tensors)
            return handle

        broadcast_opts = BroadcastOptions()
        broadcast_opts.rootRank = self.sim_rank
        return self.pg.broadcast(output_tensors, broadcast_opts)

    def reduce_scatter(self, output_tensors, input_tensors, opts):
        world1_tensors_list = self._adjust_tensors_list_to_ws1(input_tensors)
        handle = self.pg.reduce_scatter(output_tensors, world1_tensors_list, opts)
        self._copy_data_from_tensor_to_tensor_list(world1_tensors_list, input_tensors)
        return handle

    def alltoall(self, output_tensors, input_tensors, _opts):
        world1_in_tensors_list = input_tensors[self.sim_rank:self.sim_rank + 1]
        world1_out_tensors_list = output_tensors[self.sim_rank:self.sim_rank + 1]
        world1_out_tensors_list[0].data[:] = world1_in_tensors_list[0].data[:]
        opts = AllreduceOptions()
        if self.torch_ver_minor > 10:
            opts.reduceOp = ReduceOp.SUM
        handle = self.pg.allreduce(world1_out_tensors_list, opts)
        return handle

    def barrier(self, opts):
        opts.device_ids = [self.sim_rank]
        return self.pg.barrier(opts)

    # ----------------------------------------------------
    # Create group registered function
    # ----------------------------------------------------
    @classmethod
    def create(cls, _store, rank, world_size, timeout, backend):
        return cls(rank, world_size, timeout, backend)


def install_sim_dist_backend(sim_world_size, sim_rank):

    def wrapped_dist_init_process_group(backend,
                                        init_method=None,
                                        timeout=default_pg_timeout,
                                        world_size=-1,
                                        rank=-1,
                                        store=None,
                                        group_name="",
                                        pg_options=None):
        assert world_size == -1 or world_size == sim_world_size, \
            f'Inconsistent world_size: sim={sim_world_size} dist_init={world_size}'

        assert rank == -1 or rank == sim_rank, \
            f'Inconsistent rank: sim={sim_rank} dist_init={rank}'

        if backend == 'hccl':
            import habana_frameworks.torch.distributed.hccl  # noqa: F401

        # override provided init_method/store with a dummy store
        # For debug, it is better to use FileStore:
        #   import os
        #   my_store_filename = '/tmp/my_store'
        #   os.remove(my_store_filename) if os.path.exists(my_store_filename) else None
        #   os.remove(my_store_filename)
        #   store = torch.distributed.FileStore(my_store_filename, world_size)
        store = torch.distributed.TCPStore(host_name="localhost",
                                           port=12355,
                                           world_size=sim_world_size,
                                           is_master=True,
                                           timeout=timedelta(seconds=300),
                                           wait_for_workers=False)

        # set the simulated world size
        SimProcessGroup.WORLD_SIZE = sim_world_size
        SimProcessGroup.STORE = store

        # register sim backend
        # create_fn = partial(SimProcessGroup.create, backend=default_backend)
        create_fn = partial(SimProcessGroup.create, backend=backend)
        dist.Backend.register_backend(SimProcessGroup.BACKEND, create_fn)

        # emulate all other world devices has joined the newly created group
        SimProcessGroup.store_add_rest_of_world(next_group=True)

        orig_dist_init_process_group(backend=SimProcessGroup.BACKEND,
                                     timeout=timeout,
                                     world_size=sim_world_size,
                                     rank=sim_rank,
                                     store=store,
                                     group_name=group_name,
                                     pg_options=pg_options)

        SimProcessGroup.default_pg().post_create_sim_group()

    def wrapped_dist_new_group(ranks=None, timeout=default_pg_timeout, backend=None, pg_options=None):
        SimProcessGroup.store_add_rest_of_world(next_group=True)
        pg = orig_dist_new_group(ranks=ranks, timeout=timeout, backend=backend, pg_options=pg_options)

        if pg != GroupMember.NON_GROUP_MEMBER:
            if backend is None or backend == SimProcessGroup.BACKEND:
                pg.post_create_sim_group()

        return pg

    def wrapped_dist_broadcast_object_list(object_list, src=0, group=None, device=None):
        rank = SimProcessGroup.default_pg().sim_rank
        if src != sim_rank:
            raise RuntimeError(f'SimProcessGroup does not support dist.broadcast_object_list() '
                               f'for src={src} different than sim_rank={rank}')
        return orig_dist_broadcast_object_list(object_list, src, group, device)

    orig_dist_init_process_group = dist.init_process_group
    dist.init_process_group = wrapped_dist_init_process_group

    orig_dist_new_group = dist.new_group
    dist.new_group = wrapped_dist_new_group

    orig_dist_broadcast_object_list = dist.broadcast_object_list
    dist.broadcast_object_list = wrapped_dist_broadcast_object_list
