# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import unittest
import functools
import torch
import torch.distributed as dist
import pytest

from pg_sim.pg import (install_sim_dist_backend, GroupMember)


class TestBaseWrapper:
    """
    BaseTestWrapper class ensures that the test cases encapsulated
    in ProcessGroupSimTestBase will only be executed by subclasses.
    """

    class ProcessGroupSimTestBase(unittest.TestCase):

        def setUp(self) -> None:
            self.world_size = 8
            self.rank = 0
            self.backend = self.get_backend()
            self.device = self.get_device()

            self.assertIsNotNone(self.backend)
            self.assertIsNotNone(self.device)

            install_sim_dist_backend(sim_world_size=self.world_size, sim_rank=self.rank)

            dist.init_process_group(backend=self.backend,
                                    init_method=None,
                                    store=None,
                                    rank=self.rank,
                                    world_size=self.world_size)

        def get_backend(self):
            self.assertTrue(False, msg='get_backend must be implemented by derived test')

        def get_device(self):
            self.assertTrue(False, msg='get_device must be implemented by derived test')

        def _get_row_first_rank(self):
            row_ranks = list(set(range(self.world_size)) - {self.rank})
            return row_ranks[0] if row_ranks else None

        @staticmethod
        def _get_torch_version():
            return int(torch.__version__.split('.')[1])

        @pytest.mark.forked
        def test_world(self):
            res_rank = dist.get_rank()
            res_ws = dist.get_world_size()
            self.assertEqual(res_rank, self.rank)
            self.assertEqual(res_ws, self.world_size)

        @pytest.mark.forked
        def test_new_group(self):
            t = torch.tensor([1, 2]).to(self.device)
            t_in_out = t.clone()

            pg_1 = dist.new_group(ranks=[self.rank])
            dist.all_reduce(t_in_out, op=dist.ReduceOp.SUM, group=pg_1)
            self.assertTrue(t.eq(t_in_out).all())

            row_rank = self._get_row_first_rank()
            if row_rank:
                pg_2 = dist.new_group(ranks=[row_rank])
                self.assertEqual(pg_2, GroupMember.NON_GROUP_MEMBER)
            else:
                self.skipTest(f'Skipping {self._testMethodName}')

        def _test_broadcast_impl(self, src):
            t = torch.tensor([1, 2]).to(self.device)
            handle = dist.broadcast(t, src=src, async_op=False)
            self.assertIsNone(handle)

            t = torch.tensor([1, 2]).to(self.device)
            handle = dist.broadcast(t, src=src, async_op=True)
            self.assertIsNotNone(handle)
            handle.wait()

        @pytest.mark.forked
        def test_broadcast_src(self):
            self._test_broadcast_impl(src=self.rank)

        @pytest.mark.forked
        def test_broadcast_dst(self):
            row_rank = self._get_row_first_rank()
            if row_rank:
                self._test_broadcast_impl(src=row_rank)
            else:
                self.skipTest(f'Skipping {self._testMethodName}')

        def _test_broadcast_object_type_impl(self, src):
            if dist.get_rank() == src:
                objects = ["foo", 12, {1: 2}]
            else:
                objects = [None, None, None]

            dev = torch.device(self.device)
            dist.broadcast_object_list(objects, src=src, device=dev)

        @pytest.mark.forked
        def test_broadcast_object_type_src(self):
            self._test_broadcast_object_type_impl(src=self.rank)

        @pytest.mark.forked
        def test_broadcast_object_type_dst(self):
            row_rank = self._get_row_first_rank()
            if row_rank:
                with pytest.raises(RuntimeError):
                    self._test_broadcast_object_type_impl(src=row_rank)
            else:
                self.skipTest(f'Skipping {self._testMethodName}')

        @pytest.mark.forked
        def test_all_reduce(self):
            t = torch.tensor([1, 2]).to(self.device)
            t_in_out = t.clone()
            dist.all_reduce(t_in_out, op=dist.ReduceOp.SUM)
            self.assertTrue(t.eq(t_in_out).all())

        def _test_reduce_impl(self, dst):
            t = torch.tensor([1.0, 2.0]).to(self.device)
            t_in_out = t.clone()

            handle = dist.reduce(t_in_out, dst=dst, op=dist.ReduceOp.SUM, async_op=False)
            self.assertIsNone(handle)
            self.assertTrue(t.eq(t_in_out).all())

            handle = dist.reduce(t_in_out, dst=dst, op=dist.ReduceOp.SUM, async_op=True)
            self.assertIsNotNone(handle)
            handle.wait()
            self.assertTrue(t.eq(t_in_out).all())

        @pytest.mark.forked
        def test_reduce_src(self):
            self._test_reduce_impl(dst=self.rank)

        @pytest.mark.forked
        def test_reduce_dst(self):
            row_rank = self._get_row_first_rank()
            if row_rank:
                self._test_reduce_impl(dst=row_rank)
            else:
                self.skipTest(f'Skipping {self._testMethodName}')

        @pytest.mark.forked
        def test_all_gather(self):
            tensor_list = [torch.zeros(2).to(self.device) for _ in range(self.world_size)]
            tensor = torch.ones(2).to(self.device)

            handle = dist.all_gather(tensor_list, tensor, async_op=False)
            self.assertIsNone(handle)
            self.assertTrue(tensor_list[0].eq(tensor).all())

            handle = dist.all_gather(tensor_list, tensor, async_op=True)
            self.assertIsNotNone(handle)
            handle.wait()
            self.assertTrue(tensor_list[0].eq(tensor).all())

        def _test_gather_impl(self, dst, local_dst):
            torch_version = self._get_torch_version()
            if (self.backend == 'nccl') and (torch_version <= 10):
                self.skipTest(f'Skipping {self._testMethodName} for nccl '
                              f'for torch.version={torch_version}')

            tensor = torch.ones(2).to(self.device)
            gather_list = [torch.zeros(2).to(self.device) for _ in range(self.world_size)] if local_dst else None

            handle = dist.gather(tensor, gather_list, dst=dst, async_op=False)
            self.assertIsNone(handle)
            if local_dst:
                self.assertTrue(gather_list[dst].eq(tensor).all())

            handle = dist.gather(tensor, gather_list, dst=dst, async_op=True)
            self.assertIsNotNone(handle)
            handle.wait()
            if local_dst:
                self.assertTrue(gather_list[dst].eq(tensor).all())

        @pytest.mark.forked
        def test_gather_src(self):
            self._test_gather_impl(dst=self.rank, local_dst=True)

        @pytest.mark.forked
        def test_gather_not_src(self):
            row_rank = self._get_row_first_rank()
            if row_rank:
                self._test_gather_impl(dst=row_rank, local_dst=False)
            else:
                self.skipTest(f'Skipping {self._testMethodName}')

        def _test_scatter_impl(self, src, local_src):
            if self.backend not in ('gloo', 'mpi'):
                self.skipTest(f'Skipping {self._testMethodName} for {self.backend}')

            tensor = torch.ones(2).to(self.device)
            scatter_list = [torch.zeros(2).to(self.device) for _ in range(self.world_size)] if local_src else None

            handle = dist.scatter(tensor, scatter_list, src=src, async_op=False)
            self.assertIsNone(handle)
            if local_src:
                self.assertTrue(scatter_list[src].eq(tensor).all())

            handle = dist.scatter(tensor, scatter_list, src=src, async_op=True)
            self.assertIsNotNone(handle)
            handle.wait()
            if local_src:
                self.assertTrue(scatter_list[src].eq(tensor).all())

        @pytest.mark.forked
        def test_scatter_src(self):
            self._test_scatter_impl(src=self.rank, local_src=True)

        @pytest.mark.forked
        def test_scatter_not_src(self):
            row_rank = self._get_row_first_rank()
            if row_rank:
                self._test_scatter_impl(src=row_rank, local_src=False)
            else:
                self.skipTest(f'Skipping {self._testMethodName}')

        @pytest.mark.forked
        def test_reduce_scatter(self):
            if self.backend not in ('nccl', 'hccl'):
                self.skipTest(f'Skipping {self._testMethodName} for {self.backend}')

            output = torch.ones(2).to(self.device)
            input_list = [torch.zeros(2).to(self.device) for _ in range(self.world_size)]

            handle = dist.reduce_scatter(output, input_list, async_op=False)
            self.assertIsNone(handle)
            self.assertTrue(input_list[self.rank].eq(output).all())

            handle = dist.reduce_scatter(output, input_list, async_op=True)
            self.assertIsNotNone(handle)
            handle.wait()
            self.assertTrue(input_list[self.rank].eq(output).all())

        @pytest.mark.forked
        def test_all_to_all(self):
            if self.backend not in ('nccl', 'hccl', 'mpi'):
                self.skipTest(f'Skipping {self._testMethodName} for {self.backend}')

            output_list = [torch.zeros(1).to(self.device) for _ in range(self.world_size)]
            input_list = list(
                torch.arange(self.world_size, dtype=torch.float32).add(1.).to(self.device).chunk(self.world_size))

            expected_res = [
                torch.zeros(1).to(self.device) if i != self.rank else torch.ones(1).to(self.device)
                for i in range(self.world_size)
            ]

            handle = dist.all_to_all(output_list, input_list, async_op=False)
            self.assertIsNone(handle)
            self.assertTrue(
                functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, expected_res, output_list), True))

            handle = dist.all_to_all(output_list, input_list, async_op=True)
            self.assertIsNotNone(handle)
            handle.wait()
            self.assertTrue(
                functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, expected_res, output_list), True))

        @pytest.mark.forked
        def test_barrier(self):
            handle = dist.barrier(async_op=False)
            self.assertIsNone(handle)

            handle = dist.barrier(async_op=True)
            self.assertIsNotNone(handle)
            handle.wait()

        @pytest.mark.forked
        def test_p2p_send(self):
            tensor = torch.ones(2).to(self.device)
            dist.send(tensor, dst=self.rank, group=None, tag=0)

            row_rank = self._get_row_first_rank()
            dist.send(tensor, dst=row_rank, group=None, tag=0) if row_rank else None

            handle = dist.isend(tensor, dst=self.rank, group=None, tag=0)
            self.assertIsNotNone(handle)
            handle.wait()

            handle = dist.isend(tensor, dst=row_rank, group=None, tag=0)
            self.assertIsNotNone(handle)
            handle.wait()

        @pytest.mark.forked
        def test_p2p_recv(self):
            tensor = torch.zeros(2).to(self.device)
            dist.recv(tensor, src=self.rank, group=None, tag=0)

            row_rank = self._get_row_first_rank()
            dist.recv(tensor, src=row_rank, group=None, tag=0) if row_rank else None

            handle = dist.irecv(tensor, src=self.rank, group=None, tag=0)
            self.assertIsNotNone(handle)
            handle.wait()

            handle = dist.irecv(tensor, src=row_rank, group=None, tag=0)
            self.assertIsNotNone(handle)
            handle.wait()
