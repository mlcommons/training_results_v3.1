# Copyright (c) 2023 Habana Labs, Ltd. an Intel Company
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import os
import multiprocessing


def enable_hpu(flag):
    with open('unit/enable_hpu.py', 'w') as fd:
        fd.write("import pytest\n")
        fd.write(f"pytest.use_hpu='{flag}'\n")


def worker(proc_id, return_dict):
    #TODO SW-114787: move to new api outside experimental
    import habana_frameworks.torch.utils.experimental as htexp
    deviceType = htexp._get_device_type()
    if deviceType == htexp.synDeviceType.synDeviceGaudi:
        return_dict['devicetype'] = "Gaudi"
    elif deviceType == htexp.synDeviceType.synDeviceGaudi2:
        return_dict['devicetype'] = "Gaudi2"
    elif deviceType == htexp.synDeviceType.synDeviceGaudi3:
        return_dict['devicetype'] = "Gaudi3"
    else:
        return_dict['devicetype'] = None
        assert False, f'Unexpected hpu device Type: {deviceType}'


def get_hpu_dev_version():
    if not hasattr(pytest, 'hpu_dev'):
        pytest.hpu_dev = None
    if os.getenv("GAUDI3_SIM", default=None):
        pytest.hpu_dev = "Gaudi3"
    if pytest.hpu_dev not in ["Gaudi", "Gaudi2", "Gaudi3"]:
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        proc_id = 0
        multiprocessing.set_start_method("spawn", force=True)
        p = multiprocessing.Process(target=worker, args=(proc_id, return_dict))
        p.start()
        p.join()
        try:
            dev_type = return_dict['devicetype']
        except:
            assert False, 'Unexpected hpu device Type: {}'.format(return_dict['devicetype'])
        p.terminate()
        exit_code = p.exitcode
        if exit_code:
            assert False, 'HPU dev type process exit with: {}'.format(exit_code)
        if dev_type in ["Gaudi", "Gaudi2"]:
            pytest.hpu_dev = dev_type
            return dev_type
        else:
            assert False, 'Unexpected hpu device Type: {}'.format(return_dict['devicetype'])
    else:
        return pytest.hpu_dev


def is_hpu_supported(config):
    # FP16 is not supported in Gaudi1.
    if config.get('fp16'):
        if config.get('fp16', None).get('enabled', None) == True:
            if get_hpu_dev_version() == 'Gaudi':
                return False, "FP16 datatype is not supported by HPU"
    # Fused ADAM is not supported
    if config.get('optimizer'):
        if config.get('optimizer', None).get('params', None):
            if config.get('optimizer', None).get('params', None).get('torch_adam', None) == False:
                return False, "Fused ADAM optimizer is not supported by HPU"
            if config.get('optimizer', None).get('type', None) == "Lamb":
                return False, "LAMB optimizer is not supported by HPU"
            if config.get('optimizer', None).get('type', None) == "OneBitAdam":
                return False, "OneBitAdam optimizer is not supported by HPU"
    # sparse gradients is not supported by HPU.
    if 'sparse_gradients' in config:
        if config['sparse_gradients'] == True:
            return False, "sparse_gradients is not supported by HPU"

    return True, ''
