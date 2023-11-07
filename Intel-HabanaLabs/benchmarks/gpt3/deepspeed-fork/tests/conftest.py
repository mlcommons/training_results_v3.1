# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# tests directory-specific settings - this file is run automatically by pytest before any tests are run

import sys
import pytest
import os
import re
import subprocess
from os.path import abspath, dirname, join
import torch
import warnings
from unit.ci_promote_marker import *
from unit.xfail_marker import *
from unit.skip_marker import *
from unit.hpu import *

# Set this environment variable for the T5 inference unittest(s) (e.g. google/t5-v1_1-small)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# allow having multiple repository checkouts and not needing to remember to rerun
# 'pip install -e .[dev]' when switching between checkouts and running tests.
git_repo_path = abspath(join(dirname(dirname(__file__)), "src"))
sys.path.insert(1, git_repo_path)


def is_hpu_enabled(pytestconfig):
    hpu_enabled = pytestconfig.getoption("use_hpu")
    if hpu_enabled in ['True', 'true', 'TRUE', '1', 'On', 'on', 'ON']:
        return True
    else:
        return False


def update_wa_env_var(key, value):
    if key not in os.environ.keys():
        os.environ[key] = value


def set_habana_visible_modules():
    # Check if hl-smi command is available
    hl_smi_command = "hl-smi"
    try:
        subprocess.run([hl_smi_command], capture_output=True, check=True)
    except:
        print(f"{hl_smi_command} command not found or running in a simulator environment.")
        return False

    # Run the hl-smi command and capture the output
    command_output = subprocess.check_output([hl_smi_command, "-L"])

    # Extract module IDs using regular expressions
    module_ids = re.findall(r"Module ID\s+:\s+(\d+)", command_output.decode())

    # Sort the module IDs in ascending order
    sorted_module_ids = sorted(module_ids, key=int)

    # Concatenate the sorted module IDs with commas
    module_ids_string = ",".join(sorted_module_ids)

    # Set the environment variable
    os.environ["HABANA_VISIBLE_MODULES"] = module_ids_string

    print(f"The environment variable HABANA_VISIBLE_MODULES has been set to: {module_ids_string}")
    return True


def pytest_configure(config):
    config.option.color = "yes"
    config.option.durations = 0
    config.option.durations_min = 1
    config.option.verbose = True
    pytest.use_hpu = bool(is_hpu_enabled(config))
    pytest.hpu_dev = None
    if pytest.use_hpu:
        set_habana_visible_modules()
        pytest.hpu_dev = get_hpu_dev_version()
        # TODO: SW-113485 need to remove the below WA once SW-113485 is unblocked
        update_wa_env_var("PT_HPU_LAZY_ACC_PAR_MODE", "0")
        # todo SW-125782: remove DYNAMIC SHAPE disable WA
        update_wa_env_var("PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES", "0")
        # TODO SW-136999: need to remove the below WA
        update_wa_env_var("TEST_WAIT_AFTER_BARRIER", "1")


def pytest_addoption(parser):
    parser.addoption("--torch_ver", default=None, type=str)
    parser.addoption("--cuda_ver", default=None, type=str)
    parser.addoption("--use_hpu", default=None, type=str)


def validate_version(expected, found):
    version_depth = expected.count('.') + 1
    found = '.'.join(found.split('.')[:version_depth])
    return found == expected


@pytest.fixture(scope="session", autouse=True)
def check_environment(pytestconfig):
    expected_torch_version = pytestconfig.getoption("torch_ver")
    expected_cuda_version = pytestconfig.getoption("cuda_ver")
    if expected_torch_version is None:
        warnings.warn(
            "Running test without verifying torch version, please provide an expected torch version with --torch_ver")
    elif not validate_version(expected_torch_version, torch.__version__):
        pytest.exit(
            f"expected torch version {expected_torch_version} did not match found torch version {torch.__version__}",
            returncode=2)
    if expected_cuda_version is None:
        warnings.warn(
            "Running test without verifying cuda version, please provide an expected cuda version with --cuda_ver")
    elif not validate_version(expected_cuda_version, torch.version.cuda):
        pytest.exit(
            f"expected cuda version {expected_cuda_version} did not match found cuda version {torch.version.cuda}",
            returncode=2)


# Override of pytest "runtest" for DistributedTest class
# This hook is run before the default pytest_runtest_call
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    # We want to use our own launching function for distributed tests
    if getattr(item.cls, "is_dist_test", False):
        dist_test_class = item.cls()
        dist_test_class(item._request)
        item.runtest = lambda: True  # Dummy function so test is not run twice
    else:
        if bool(pytest.use_hpu) == True:
            import habana_frameworks.torch.hpu  # noqa: F401


def pytest_collection_modifyitems(items, config):
    if bool(pytest.use_hpu) == True:
        gaudi_dev = get_hpu_dev_version()
    for item in items:
        if item._nodeid in hpu_ci_tests:
            item._pyfuncitem.add_marker(pytest.mark.hpu_ci)
        if item._nodeid in gpu_ci_tests:
            item._pyfuncitem.add_marker(pytest.mark.gpu_ci)
        if item._nodeid in hpu_promote_tests:
            item._pyfuncitem.add_marker(pytest.mark.hpu_promote)
        if item._nodeid in gpu_promote_tests:
            item._pyfuncitem.add_marker(pytest.mark.gpu_promote)
    for item in items:
        item.user_properties.append(("module_name", item.module.__name__))
        if item._nodeid in hpu_xfail_tests.keys():
            item._pyfuncitem.add_marker(
                pytest.mark.xfail(bool(pytest.use_hpu) == True, reason=hpu_xfail_tests[item._nodeid]))
        if item._nodeid in gpu_xfail_tests.keys():
            if bool(pytest.use_hpu) != True:
                item._pyfuncitem.add_marker(
                    pytest.mark.xfail(bool(pytest.use_hpu) != True, reason=gpu_xfail_tests[item._nodeid]))
        if item._nodeid in hpu_skip_tests.keys():
            item._pyfuncitem.add_marker(
                pytest.mark.skipif(bool(pytest.use_hpu) == True, reason=hpu_skip_tests[item._nodeid]))
        if item._nodeid in gpu_skip_tests.keys():
            if bool(pytest.use_hpu) != True:
                item._pyfuncitem.add_marker(
                    pytest.mark.skipif(bool(pytest.use_hpu) != True, reason=gpu_skip_tests[item._nodeid]))
        if bool(pytest.use_hpu) == True:
            if gaudi_dev == "Gaudi":
                if item._nodeid in g1_xfail_tests.keys():
                    item._pyfuncitem.add_marker(pytest.mark.xfail(reason=g1_xfail_tests[item._nodeid]))
                if item._nodeid in g1_skip_tests.keys():
                    item._pyfuncitem.add_marker(pytest.mark.skip(reason=g1_skip_tests[item._nodeid]))
            if gaudi_dev == "Gaudi2":
                if item._nodeid in g2_xfail_tests.keys():
                    item._pyfuncitem.add_marker(pytest.mark.xfail(reason=g2_xfail_tests[item._nodeid]))
                if item._nodeid in g2_skip_tests.keys():
                    item._pyfuncitem.add_marker(pytest.mark.skip(reason=g2_skip_tests[item._nodeid]))
        for marker in item.own_markers:
            if marker.name in ['skip', 'xfail']:
                if 'reason' in marker.kwargs:
                    item.user_properties.append(("message", marker.kwargs['reason']))


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    if getattr(fixturedef.func, "is_dist_fixture", False):
        #for val in dir(request):
        #    print(val.upper(), getattr(request, val), "\n")
        dist_fixture_class = fixturedef.func()
        dist_fixture_class(request)


def pytest_runtest_makereport(item, call):
    if call.when == 'call':
        if call.excinfo:
            if not (any('message' in prop for prop in item.user_properties)):
                if call.excinfo.value:
                    item.user_properties.append(("message", call.excinfo.value))
