# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.accelerator import get_accelerator


def get_use_hpu():
    return get_accelerator().device_name() == "hpu"
