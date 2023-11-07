# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import unittest
from pg_sim.ut.base import TestBaseWrapper


class ProcessGroupSimTestNccl(TestBaseWrapper.ProcessGroupSimTestBase):

    def setUp(self) -> None:
        super(ProcessGroupSimTestNccl, self).setUp()

    def get_backend(self):
        return 'nccl'

    def get_device(self):
        return 'cuda'


if __name__ == '__main__':
    unittest.main()
