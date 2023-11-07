# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import unittest
from pg_sim.ut.base import TestBaseWrapper


class ProcessGroupSimTestGloo(TestBaseWrapper.ProcessGroupSimTestBase):

    def setUp(self) -> None:
        super(ProcessGroupSimTestGloo, self).setUp()

    def get_backend(self):
        return 'gloo'

    def get_device(self):
        return 'cpu'


if __name__ == '__main__':
    unittest.main()
