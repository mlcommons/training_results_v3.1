# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import unittest
from pg_sim.ut.base import TestBaseWrapper


class ProcessGroupSimTestHccl(TestBaseWrapper.ProcessGroupSimTestBase):

    def setUp(self) -> None:
        super(ProcessGroupSimTestHccl, self).setUp()

    def get_backend(self):
        return 'hccl'

    def get_device(self):
        return 'hpu'


if __name__ == '__main__':
    unittest.main()
