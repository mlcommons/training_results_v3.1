# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2019 NVIDIA CORPORATION. All rights reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import functools
import logging
import random
import datetime
import time
import gc
import numpy as np

import torch
import apex_C, amp_C
from apex.multi_tensor_apply import multi_tensor_applier
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.data.datasets.coco import HybridDataLoader3
from maskrcnn_benchmark.solver.fused_sgd import FusedSGD
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.engine.tester import test
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.modeling.backbone.resnet import _HALO_EXCHANGERS
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process, get_world_size, is_main_evaluation_process
from maskrcnn_benchmark.utils.batch_size import per_gpu_batch_size
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.mlperf_logger import generate_seeds, broadcast_seeds, mllogger
from maskrcnn_benchmark.utils.async_evaluator import init, get_evaluator, set_epoch_tag, get_tag
from maskrcnn_benchmark.utils.timed_section import TimedSection
from maskrcnn_benchmark.structures.image_list import to_image_list, backbone_image
from maskrcnn_benchmark.layers.nhwc import nchw_to_nhwc_transform
from maskrcnn_benchmark.engine.trainer import SyntheticDataLoader
from scaleoutbridge import EmptyObject, ScaleoutBridge as SBridge
from fp16_optimizer import FP16_Optimizer

from mlperf_logging.mllog import constants
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as DDP
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

torch.backends.cudnn.deterministic = True
# Loop over all finished async results, return a dict of { tag : (bbox_map, segm_map) }
finished_prep_work = None

# use NVFuser instead of NNC to enable fusing apex bottleneck's backward ops
torch._C._jit_set_nvfuser_enabled(True)
torch._C._jit_set_texpr_fuser_enabled(False)
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)


def unify_model_names(model_parameters):
    def remove_prefix(prefix, string):
        if string.startswith(prefix):
            return string[len(prefix):]
        else:
            return string
    new_model_parameters = {}
    for name, param in model_parameters.items():
        old_name = name
        name = remove_prefix("graphable.model_segment.", name)
        name = remove_prefix("combined_rpn_roi.", name)
        name = name.replace("roi_mask_only", "roi_heads")
        new_model_parameters[name] = param
    assert( len(model_parameters) == len(new_model_parameters) )
    return new_model_parameters

# Permute some parameters in channels_last format to explicit nhwc
def explicitize_model(parameters):
    p = {}
    for name, param in parameters.items():
        if "roi_mask_head_fcn" in name and "weight" in name and param.is_contiguous(memory_format=torch.channels_last):
            p[name] = param.permute([0,2,3,1])
        else:
            p[name] = param
    return p

def copy_model(from_model, to_model):
    with torch.no_grad():
        from_parameters = explicitize_model(unify_model_names(dict(from_model.named_parameters())))
        to_parameters = unify_model_names(dict(to_model.named_parameters()))
        #if get_rank() == 0:
        #    print("from_parameters = %s, to_parameters = %s" % (str([k for k,v in from_parameters.items()]), str([k for k,v in to_parameters.items()])))
        for name, param in from_parameters.items():
            to_parameters[name].copy_(param)
        from_buffers = unify_model_names(dict(from_model.named_buffers()))
        to_buffers = unify_model_names(dict(to_model.named_buffers()))
        #if get_rank() == 0:
        #    print("from_buffers = %s, to_buffers = %s" % (str([k for k,v in from_buffers.items()]), str([k for k,v in to_buffers.items()])))
        for name, buf in from_buffers.items():
            to_buffers[name].copy_(buf)


# Get complete test model (all parameters + all buffers) as a list that can be flattened
def get_test_model(model_test):
    p = []
    test_parameters = unify_model_names(dict(model_test.named_parameters()))
    for name, param in test_parameters.items():
        p.append(param)
    test_buffers = unify_model_names(dict(model_test.named_buffers()))
    for name, buf in test_buffers.items():
        p.append(buf)
    return p

# Get complete training model compatible with model returned by get_test_model
def get_training_model(model_train, model_test):
    p = []
    training_parameters = explicitize_model(unify_model_names(dict(model_train.named_parameters())))
    test_parameters = unify_model_names(dict(model_test.named_parameters()))
    for name in test_parameters.keys():
        p.append(training_parameters[name])
    training_buffers = unify_model_names(dict(model_train.named_buffers()))
    test_buffers = unify_model_names(dict(model_test.named_buffers()))
    for name in test_buffers.keys():
        p.append(training_buffers[name])
    return p

def flatten_model(params_and_buffers):
    r = []
    for pb in params_and_buffers:
        if pb.is_contiguous(memory_format=torch.channels_last):
            r.append( pb.to(memory_format=torch.contiguous_format) )
        else:
            r.append( pb )
    return apex_C.flatten(r)

def unflatten_model(flat_params_and_buffers, params_and_buffers):
    r = []
    for upb, pb in zip(apex_C.unflatten(flat_params_and_buffers, params_and_buffers), params_and_buffers):
        if pb.is_contiguous(memory_format=torch.channels_last):
            r.append( upb.to(memory_format=torch.channels_last) )
        else:
            r.append( upb )
    return r

#def copy_flat_model(from_model, to_model):
#    flat_params_and_buffers = flatten_model(get_training_model(from_model, to_model))
#    params_and_buffers = get_test_model(to_model)
#    overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
#    multi_tensor_applier(
#            amp_C.multi_tensor_scale,
#            overflow_buf,
#            [unflatten_model(flat_params_and_buffers, params_and_buffers), params_and_buffers],
#            1.0)

def check_completed_tags(iteration, world_size, dedicated_evalution_ranks=0, eval_ranks_comm=None):
    # Check for completeness is fairly expensive, so we only do it once per N iterations
    # Only applies when not using dedicated evaluation ranks
    if dedicated_evalution_ranks == 0 and iteration % 10 != 9:
        return {}

    num_evaluation_ranks = world_size if dedicated_evalution_ranks == 0 else dedicated_evalution_ranks

    global finished_prep_work
    from maskrcnn_benchmark.data.datasets.evaluation.coco.coco_eval import COCOResults, all_gather_prep_work, evaluate_coco
    if num_evaluation_ranks > 1:
        num_finished = torch.zeros([1], dtype=torch.int32, device='cuda') if finished_prep_work is None else torch.ones([1], dtype=torch.int32, device='cuda')
        torch.distributed.all_reduce(num_finished, group=eval_ranks_comm)
        ready_to_submit_evaluation_task = True if num_finished == num_evaluation_ranks else False
    else:
        ready_to_submit_evaluation_task = False if finished_prep_work is None else True
    evaluator = get_evaluator()
    if ready_to_submit_evaluation_task:
        with TimedSection("EXPOSED: Launching evaluation task took %.3fs"):
            coco_results, iou_types, coco, output_folder = finished_prep_work
            finished_prep_work = None
            coco_results = all_gather_prep_work(coco_results, dedicated_evalution_ranks, eval_ranks_comm)
            if is_main_evaluation_process(dedicated_evalution_ranks):
                evaluator.submit_task(get_tag(),
                                      evaluate_coco,
                                      coco,
                                      coco_results,
                                      iou_types,
                                      output_folder)
    else:
        # loop over all all epoch, result pairs that have finished
        all_results = {}
        for t, r in evaluator.finished_tasks().items():
            # Note: one indirection due to possibility of multiple test datasets
            # we only care about the first
            map_results = r# [0]
            if isinstance(map_results, COCOResults):
                bbox_map = map_results.results["bbox"]['AP']
                segm_map = map_results.results["segm"]['AP']
                all_results.update({ t : (bbox_map, segm_map) })
            else:
                finished_prep_work = map_results

        return all_results

    return {}


def mlperf_test_early_exit(iteration, iters_per_epoch, num_iteration_to_run_eval, tester, model_train, model_test, dont_recompute_scale_bias, distributed, min_bbox_map, min_segm_map, world_size, data_loader, sbridge=EmptyObject(), H_split=True):
    # Note: let iters / epoch == 10k, at iter 9999 we've finished epoch 0 and need to test
    if iteration > 0 and (iteration + 1)% iters_per_epoch == 0:
        synchronize(comm=None)
        epoch = iteration // iters_per_epoch + 1
        sbridge.stop_epoch_prof()
        mllogger.end(key=mllogger.constants.EPOCH_STOP, metadata={"epoch_num": epoch})
        mllogger.end(key=mllogger.constants.BLOCK_STOP, metadata={"first_epoch_num": epoch})
        sbridge.start_eval_prof()
        mllogger.start(key=mllogger.constants.EVAL_START, metadata={"epoch_num":epoch})
        # set the async evaluator's tag correctly
        set_epoch_tag(epoch)

        # Note: No longer returns anything, underlying future is in another castle
        copy_model(model_train, model_test)
        if dont_recompute_scale_bias:
            model_test.compute_scale_bias()
        tester(model=model_test, distributed=distributed, H_split=H_split)
        # necessary for correctness
        # TODO: Probably not necessary when model_train and model_test are two distinct models?
        model_train.enable_train()
        if hasattr(data_loader.dataset, "global_transforms") and data_loader.dataset.global_transforms:
            data_loader.dataset.shuffle()
    elif iteration > 0 and num_iteration_to_run_eval > 0 and (iteration + 1)% num_iteration_to_run_eval == 0:
        synchronize(comm=None)
        sbridge.start_eval_prof()
        # TODO: Should this be done before start_eval_prof?
        copy_model(model_train, model_test)
        if dont_recompute_scale_bias:
            model_test.compute_scale_bias()
        # End TODO comment
        tester(model=model_test, distributed=distributed)
        model_train.enable_train()
    elif iteration % 10 == 9: # do finished check after every 10 iterations
        # Otherwise, check for finished async results
        results = check_completed_tags(iteration, world_size)

        # on master process, check each result for terminating condition
        # sentinel for run finishing
        finished = 0
        if is_main_process():
            for result_epoch, (bbox_map, segm_map) in results.items():
                logger = logging.getLogger('maskrcnn_benchmark.trainer')
                logger.info('bbox mAP: {}, segm mAP: {}'.format(bbox_map, segm_map))

                mllogger.event(key=mllogger.constants.EVAL_ACCURACY, value={"BBOX" : bbox_map, "SEGM" : segm_map}, metadata={"epoch_num" : result_epoch} )
                sbridge.stop_eval_prof()
                mllogger.end(key=mllogger.constants.EVAL_STOP, metadata={"epoch_num": result_epoch})
                # terminating condition
                if bbox_map >= min_bbox_map and segm_map >= min_segm_map:
                    logger.info("Target mAP reached, exiting...")
                    finished = 1
                    #return True

        # We now know on rank 0 whether or not we should terminate
        # Bcast this flag on multi-GPU
        if world_size > 1:
            with torch.no_grad():
                finish_tensor = torch.tensor([finished], dtype=torch.int32, device = torch.device('cuda'))
                torch.distributed.broadcast(finish_tensor, 0)

                # If notified, end.
                if finish_tensor.item() == 1:
                    return True, sbridge
        else:
            # Single GPU, don't need to create tensor to bcast, just use value directly
            if finished == 1:
                return True, sbridge

    # Otherwise, default case, continue
    return False, sbridge

__eval_start_time = 0

def mlperf_evaluation_test_loop(tester, model_test, distributed, eval_ranks_comm, dedicated_evaluation_ranks, num_training_ranks, spatial_group_size_train, min_bbox_map, min_segm_map, world_size, H_split=True):
    finished = 0
    params_and_buffers = get_test_model(model_test)
    flat_params_and_buffers = flatten_model(params_and_buffers)
    while finished == 0:
        torch.distributed.barrier() # block process until training ranks have work for us

        # wait for parameter broadcast from training master rank
        torch.distributed.broadcast(flat_params_and_buffers, 0)

        # wait for epoch from training master rank
        epoch_t = torch.zeros([1], dtype=torch.int32, device='cuda')
        torch.distributed.broadcast(epoch_t, 0)
        epoch = epoch_t.item()
        dryrun = True if epoch == 0 else False

        # decide if we are done
        finished = 1 if epoch_t.item() < 0 else 0

        if finished == 0:
            global __eval_start_time
            __eval_start_time = time.time()

            # update evaluation model
            overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
            multi_tensor_applier(
                    amp_C.multi_tensor_scale,
                    overflow_buf,
                    [unflatten_model(flat_params_and_buffers, params_and_buffers), params_and_buffers],
                    1.0)

            # set the async evaluator's tag correctly
            set_epoch_tag(epoch)

            # do evaluation
            tester(model=model_test, distributed=distributed, dryrun=dryrun, H_split=H_split)
            #model.enable_train()

            if not dryrun:
                # busy wait until evaluation is done
                got_results = False
                while not got_results:
                    time.sleep(0.05)
                    results = check_completed_tags(0, world_size, dedicated_evaluation_ranks, eval_ranks_comm) # iteration is ignored when using dedicated evaluation ranks

                    # on master process, check each result for terminating condition
                    # sentinel for run finishing
                    if is_main_evaluation_process(dedicated_evaluation_ranks):
                        for result_epoch, (bbox_map, segm_map) in results.items():
                            # terminating condition
                            if bbox_map >= min_bbox_map and segm_map >= min_segm_map:
                                finished = 1
                            with torch.no_grad():
                                results_t = torch.tensor([finished, result_epoch, bbox_map, segm_map], dtype=torch.float64, device='cuda')
                            got_results = True
                        if got_results:
                            elapsed_evaluation_time = time.time() - __eval_start_time
                            logger = logging.getLogger('maskrcnn_benchmark.evaluation')
                            logger.info("Evaluation took %.3f seconds" % (elapsed_evaluation_time))

                    # signal to other evaluation ranks whether we got results or not
                    got_results_t = torch.tensor([1 if got_results else 0], dtype=torch.int32, device='cuda')
                    torch.distributed.broadcast(got_results_t, num_training_ranks*spatial_group_size_train, group=eval_ranks_comm)
                    got_results = True if got_results_t.item() == 1 else False

                # broadcast result
                torch.distributed.barrier() # block process until training ranks are ready to accept results
                if not is_main_evaluation_process(dedicated_evaluation_ranks):
                    with torch.no_grad():
                        results_t = torch.zeros([4], dtype=torch.float64, device='cuda')
                torch.distributed.broadcast(results_t, num_training_ranks*spatial_group_size_train)
                finished, result_epoch, bbox_map, segm_map = results_t.tolist()
                finished = int(finished)
                result_epoch = int(result_epoch)

    # TODO: Find out why this barrier call is necessary. Code hangs without it
    if torch.distributed.is_initialized():
        torch.distributed.barrier() # prevent evaluation ranks from terminating before training ranks
    else:
        torch.cuda.synchronize()

__eval_start_iteration = -1

def launch_eval_on_dedicated_ranks(model_train, model_test, iteration, epoch):
    global __eval_start_iteration

    torch.distributed.barrier() # release evaluation ranks so they can wait for work broadcast
    
    # broadcast model so evaluation ranks can start evaluation
    params_and_buffers = get_training_model(model_train, model_test)
    flat_params_and_buffers = flatten_model(params_and_buffers)
    torch.distributed.broadcast(flat_params_and_buffers, 0)

    # broadcast epoch so master evaluation rank can set async evaluator's tag correctly
    epoch_t = torch.tensor([epoch], dtype=torch.int32, device='cuda')
    torch.distributed.broadcast(epoch_t, 0)

    dryrun = True if epoch == 0 else False
    if not dryrun:
        __eval_start_iteration = iteration

# TODO: Make sure protocol allows evaluation ranks to finish when training ranks reach max_iter
def mlperf_training_test_early_exit(iteration, iters_per_epoch, training_ranks_comm, num_training_ranks, spatial_group_size_train, model_train, model_test, data_loader, wait_this_many_iterations_before_checking_result, sbridge=EmptyObject()):
    global __eval_start_iteration
    if __eval_start_iteration >= 0:
        epoch = iteration // iters_per_epoch
        early_wait, early_epochs, late_wait = wait_this_many_iterations_before_checking_result
        lapsed_iterations = iteration - __eval_start_iteration
        if (epoch <= early_epochs and lapsed_iterations >= early_wait) or (epoch > early_epochs and lapsed_iterations >= late_wait):
            __eval_start_iteration = -1
            # wait for result
            start = time.time()
            torch.distributed.barrier() # signal to evaluation ranks that we are ready for results
            with torch.no_grad():
                results_t = torch.zeros([4], dtype=torch.float64, device='cuda')
            torch.distributed.broadcast(results_t, num_training_ranks*spatial_group_size_train)
            finished, result_epoch, bbox_map, segm_map = results_t.tolist()
            finished = int(finished)
            result_epoch = int(result_epoch)
                
            if is_main_process() and result_epoch > 0:
                end = time.time()
                logger = logging.getLogger('maskrcnn_benchmark.trainer')
                logger.info("Waited for %.3f seconds for results for epoch %d" % (end-start, epoch))
                logger.info('bbox mAP: {}, segm mAP: {}'.format(bbox_map, segm_map))

                mllogger.event(key=mllogger.constants.EVAL_ACCURACY, value={"BBOX" : bbox_map, "SEGM" : segm_map}, metadata={"epoch_num" : result_epoch} )
                mllogger.end(key=mllogger.constants.EVAL_STOP, metadata={"epoch_num": result_epoch})
                if finished == 1:
                    logger.info("Target mAP reached, exiting...")

            if finished == 1:
                return True, sbridge

    elif iteration > 0 and (iteration + 1)% iters_per_epoch == 0:
        synchronize(training_ranks_comm)
        epoch = iteration // iters_per_epoch + 1

        mllogger.end(key=mllogger.constants.EPOCH_STOP, metadata={"epoch_num": epoch})
        mllogger.end(key=mllogger.constants.BLOCK_STOP, metadata={"first_epoch_num": epoch})
        mllogger.start(key=mllogger.constants.EVAL_START, metadata={"epoch_num":epoch})

        launch_eval_on_dedicated_ranks(model_train, model_test, iteration, epoch)
        if hasattr(data_loader.dataset, "global_transforms") and data_loader.dataset.global_transforms:
            data_loader.dataset.shuffle()

    return False, sbridge

def terminate_evaluation_ranks(iters_per_epoch, training_ranks_comm, num_training_ranks, spatial_group_size_train, model_train, model_test, data_loader, wait_this_many_iterations_before_checking_result):
    # collect last pending results (if any)
    global __eval_start_iteration
    if __eval_start_iteration >= 0:
        early_wait, early_epochs, late_wait = wait_this_many_iterations_before_checking_result
        iteration = __eval_start_iteration + max(early_wait, late_wait)
        success, _ = mlperf_training_test_early_exit(iteration, iters_per_epoch, training_ranks_comm, num_training_ranks, spatial_group_size_train, model_train, model_test, data_loader, wait_this_many_iterations_before_checking_result)
        __eval_start_iteration = -1
    else:
        success = False
   
    torch.distributed.barrier() # release evaluation ranks so they can wait for work broadcast

    # signal to evaluation ranks that they're finished
    params_and_buffers = get_training_model(model_train, model_test)
    flat_params_and_buffers = flatten_model(params_and_buffers)
    torch.distributed.broadcast(flat_params_and_buffers, 0)

    # negative value for epoch signals that we are done
    epoch = -1
    epoch_t = torch.tensor([epoch], dtype=torch.int32, device='cuda')
    torch.distributed.broadcast(epoch_t, 0)

    return success

def mlperf_log_epoch_start(iteration, iters_per_epoch):
    # First iteration:
    #     Note we've started training & tag first epoch start
    if iteration == 0:
        log_start(key=constants.BLOCK_START, metadata={"first_epoch_num":1, "epoch_count":1})
        log_start(key=constants.EPOCH_START, metadata={"epoch_num":1})
        return
    if iteration % iters_per_epoch == 0:
        epoch = iteration // iters_per_epoch + 1
        log_start(key=constants.BLOCK_START, metadata={"first_epoch_num": epoch, "epoch_count": 1})
        log_start(key=constants.EPOCH_START, metadata={"epoch_num": epoch})

from maskrcnn_benchmark.layers.batch_norm import FrozenBatchNorm2d
from maskrcnn_benchmark.layers.nhwc.batch_norm import FrozenBatchNorm2d_NHWC
from maskrcnn_benchmark.modeling.backbone.resnet import Bottleneck
def cast_frozen_bn_to_half(module):
    if isinstance(module, FrozenBatchNorm2d) or isinstance(module, FrozenBatchNorm2d_NHWC):
        module.half()
    for child in module.children():
        cast_frozen_bn_to_half(child)
    return module

def mlperf_log_epoch_start(iteration, iters_per_epoch):
    # First iteration:
    #     Note we've started training & tag first epoch start
    if iteration == 0:
        mllogger.start(key=mllogger.constants.BLOCK_START, metadata={"first_epoch_num":1, "epoch_count":1})
        mllogger.start(key=mllogger.constants.EPOCH_START, metadata={"epoch_num":1})
        return
    if iteration % iters_per_epoch == 0:
        epoch = iteration // iters_per_epoch + 1
        mllogger.start(key=mllogger.constants.BLOCK_START, metadata={"first_epoch_num": epoch, "epoch_count": 1})
        mllogger.start(key=mllogger.constants.EPOCH_START, metadata={"epoch_num": epoch})

from maskrcnn_benchmark.layers.batch_norm import FrozenBatchNorm2d
from maskrcnn_benchmark.layers.nhwc.batch_norm import FrozenBatchNorm2d_NHWC
from maskrcnn_benchmark.modeling.backbone.resnet import Bottleneck
def cast_frozen_bn_to_half(module):
    if isinstance(module, FrozenBatchNorm2d) or isinstance(module, FrozenBatchNorm2d_NHWC):
        module.half()
    for child in module.children():
        cast_frozen_bn_to_half(child)
    return module


def initialize_nccl_comms(cfg):
    dedicated_evaluation_ranks, num_training_ranks, images_per_batch_train, images_per_gpu_train, rank_train, rank_in_group_train, spatial_group_size_train, num_evaluation_ranks, images_per_batch_test, images_per_gpu_test, rank_test, rank_in_group_test, spatial_group_size_test = per_gpu_batch_size(cfg, log_info=True)
    is_training_rank = True if rank_train >= 0 else False
    is_evaluation_rank = True if rank_test >= 0 else False

    if dedicated_evaluation_ranks > 0:
        # create nccl comm for training ranks
        training_ranks = [i for i in range(num_training_ranks*spatial_group_size_train)]
        training_comm = torch.distributed.new_group(ranks=training_ranks)
        if is_training_rank:
            dummy = torch.ones([1], device='cuda')
            torch.distributed.all_reduce(dummy, group=training_comm) # wake up new comm

        # create nccl comm for evaluation ranks
        evaluation_ranks = [i+num_training_ranks*spatial_group_size_train for i in range(num_evaluation_ranks*spatial_group_size_test)]
        print("%d :: evaluation_ranks = %s" % (get_rank(), str(evaluation_ranks)))
        evaluation_comm = torch.distributed.new_group(ranks=evaluation_ranks)
        if is_evaluation_rank:
            dummy = torch.ones([1], device='cuda')
            torch.distributed.all_reduce(dummy, group=evaluation_comm) # wake up new comm

        return training_comm, evaluation_comm
    else:
        # initialize global communicator
        dummy = torch.ones([1], device='cuda')
        torch.distributed.all_reduce(dummy)
        return None, None


def train(cfg, rank, world_size, distributed, training_comm, evaluation_comm, random_number_generator=None, seed=None):
    logger = logging.getLogger('maskrcnn_benchmark.trainer')

    # Model logging
    mllogger.event(key=mllogger.constants.GLOBAL_BATCH_SIZE, value=cfg.SOLVER.IMS_PER_BATCH)
    mllogger.event(key=mllogger.constants.NUM_IMAGE_CANDIDATES, value=cfg.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN)
    mllogger.event(key=mllogger.constants.GRADIENT_ACCUMULATION_STEPS, value=1)

    H_split = cfg.MODEL.BACKBONE.SPATIAL_H_SPLIT
    assert(H_split), "MODEL.BACKBONE.SPATIAL_H_SPLIT must be True for now"

    from maskrcnn_benchmark.modeling.detector.generalized_rcnn import create_spatial_parallel_args
    spatial_parallel_args_train, peer_pool_train, spatial_parallel_args_test, peer_pool_test = create_spatial_parallel_args(cfg)

    dedicated_evaluation_ranks, num_training_ranks, images_per_batch_train, images_per_gpu_train, rank_train, rank_in_group_train, spatial_group_size_train, num_evaluation_ranks, images_per_batch_test, images_per_gpu_test, rank_test, rank_in_group_test, spatial_group_size_test = per_gpu_batch_size(cfg, log_info=False)
    is_training_rank = True if rank_train >= 0 else False
    is_evaluation_rank = True if rank_test >= 0 else False
    print("%d :: dedicated_evaluation_ranks = %d, num_training_ranks = %d, images_per_batch_train = %d, images_per_gpu_train = %d, rank_train = %d, rank_in_group_train = %d, spatial_group_size_train = %d, num_evaluation_ranks = %d, images_per_batch_test = %d, images_per_gpu_test = %d, rank_test = %d, rank_in_group_test = %d, spatial_group_size_test = %d" % (get_rank(), dedicated_evaluation_ranks, num_training_ranks, images_per_batch_train, images_per_gpu_train, rank_train, rank_in_group_train, spatial_group_size_train, num_evaluation_ranks, images_per_batch_test, images_per_gpu_test, rank_test, rank_in_group_test, spatial_group_size_test))
    print("%d :: is_training_rank = %s, is_evaluation_rank = %s" % (get_rank(), "True" if is_training_rank else "False", "True" if is_evaluation_rank else "False"))

    annotations_per_image = cfg.DATALOADER.MAX_ANNOTATIONS_PER_IMAGE
    # TODO: Only training utilizes batch-size-1 graphing for the time being
    batch_size_one = cfg.ALLOW_BATCH_SIZE_ONE_GRAPHING and images_per_gpu_train == 1 and annotations_per_image > 0
    mask_loss_from_global = cfg.DATALOADER.MASK_LOSS_FROM_GLOBAL_TENSORS
    if get_rank() == 0:
        print("%d :: batch_size_one=%s, images_per_gpu_train=%d, images_per_gpu_test=%d" % (get_rank(), str(batch_size_one), images_per_gpu_train, images_per_gpu_test))
    if cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR_FUSION:
        assert(annotations_per_image > 0 and images_per_gpu_train == 1), "Error! mask head fusion requires cfg.DATALOADER.MAX_ANNOTATIONS_PER_IMAGE to be > 0 and batch-size == 1"
    elif annotations_per_image > 0 and images_per_gpu_train == 1:
        logger.info("Setting MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR_FUSION to True may increase performance")
    model_train = build_detection_model(cfg, (spatial_parallel_args_train, peer_pool_train, spatial_parallel_args_test, peer_pool_test,), batch_size_one)
    cfg.defrost()
    # Always disable compliance logs for evaluation model
    cfg.ENABLE_COMPLIANCE_LOG = False
    # mask head only supports fusion during training
    cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR_FUSION = False
    cfg.freeze()
    model_test = build_detection_model(cfg, (spatial_parallel_args_train, peer_pool_train, spatial_parallel_args_test, peer_pool_test,), False)
    device = torch.device(cfg.MODEL.DEVICE)
    model_train.to(device)
    model_test.to(device)

    # Initialize mixed-precision training
    is_fp16 = (cfg.DTYPE == "float16")
    if is_fp16:
        # convert model to FP16
        model_train.half()
        model_test.half()

    # - CUDA graph ------
    from function import graph

    if cfg.DATALOADER.ALWAYS_PAD_TO_MAX or cfg.USE_CUDA_GRAPH:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN[0] if isinstance(cfg.INPUT.MIN_SIZE_TRAIN, tuple) else cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN[0] if isinstance(cfg.INPUT.MAX_SIZE_TRAIN, tuple) else cfg.INPUT.MAX_SIZE_TRAIN
        divisibility = max(1, cfg.DATALOADER.SIZE_DIVISIBILITY)

        if is_training_rank:
            # training shapes
            divisibility_train_H = divisibility * (model_train.spatial_group_size_train if model_train.spatial_H_split else 1)
            divisibility_train_W = divisibility * (1 if model_train.spatial_H_split else model_train.spatial_group_size_train)
            shapes_per_orientation_train_H = cfg.CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION
            shapes_per_orientation_train_W = cfg.CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION
            min_size_train_H = ((min_size + divisibility_train_H - 1) // divisibility_train_H) * divisibility_train_H
            max_size_train_H = ((max_size + divisibility_train_H - 1) // divisibility_train_H) * divisibility_train_H
            min_size_train_W = ((min_size + divisibility_train_W - 1) // divisibility_train_W) * divisibility_train_W
            max_size_train_W = ((max_size + divisibility_train_W - 1) // divisibility_train_W) * divisibility_train_W
            size_range_train_H = (max_size_train_H - min_size_train_H) // divisibility_train_H
            size_range_train_W = (max_size_train_W - min_size_train_W) // divisibility_train_W
            max_shapes_per_orientation_train_H = size_range_train_H + 1
            max_shapes_per_orientation_train_W = size_range_train_W + 1
            if shapes_per_orientation_train_H > max_shapes_per_orientation_train_H:
                logger.info("Reduced number of training shapes for H from %d to %d to satisfy divisibility of %d" % (shapes_per_orientation_train_H, max_shapes_per_orientation_train_H, divisibility_train_H))
                shapes_per_orientation_train_H = max_shapes_per_orientation_train_H
            if shapes_per_orientation_train_W > max_shapes_per_orientation_train_W:
                logger.info("Reduced number of training shapes for W from %d to %d to satisfy divisibility of %d" % (shapes_per_orientation_train_W, max_shapes_per_orientation_train_W, divisibility_train_W))
                shapes_per_orientation_train_W = max_shapes_per_orientation_train_W
            shapes_train = []
            for i in range(0,shapes_per_orientation_train_H):
                size_H = min_size_train_H + ((i+1) * size_range_train_H // shapes_per_orientation_train_H) * divisibility_train_H
                shapes_train.append( (size_H, min_size_train_W) )
            for i in range(0,shapes_per_orientation_train_W):
                size_W = min_size_train_W + ((i+1) * size_range_train_W // shapes_per_orientation_train_W) * divisibility_train_W
                shapes_train.append( (min_size_train_H, size_W) )
            logger.info("Training shapes are %s" % (str(shapes_train))) 

        if is_evaluation_rank:
            # evaluation shapes
            divisibility_test_H = divisibility * (model_test.spatial_group_size_eval if model_test.spatial_H_split else 1)
            divisibility_test_W = divisibility * (1 if model_test.spatial_H_split else model_test.spatial_group_size_eval)
            shapes_per_orientation_test_H = cfg.CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION_TEST
            shapes_per_orientation_test_W = cfg.CUDA_GRAPH_NUM_SHAPES_PER_ORIENTATION_TEST
            min_size_test_H = ((min_size + divisibility_test_H - 1) // divisibility_test_H) * divisibility_test_H
            max_size_test_H = ((max_size + divisibility_test_H - 1) // divisibility_test_H) * divisibility_test_H
            min_size_test_W = ((min_size + divisibility_test_W - 1) // divisibility_test_W) * divisibility_test_W
            max_size_test_W = ((max_size + divisibility_test_W - 1) // divisibility_test_W) * divisibility_test_W
            size_range_test_H = (max_size_test_H - min_size_test_H) // divisibility_test_H
            size_range_test_W = (max_size_test_W - min_size_test_W) // divisibility_test_W
            max_shapes_per_orientation_test_H = size_range_test_H + 1
            max_shapes_per_orientation_test_W = size_range_test_W + 1
            if shapes_per_orientation_test_H > max_shapes_per_orientation_test_H:
                logger.info("Reduced number of evaluation shapes for H from %d to %d to satisfy divisibility of %d" % (shapes_per_orientation_test_H, max_shapes_per_orientation_test_H, divisibility_test_H))
                shapes_per_orientation_test_H = max_shapes_per_orientation_test_H
            if shapes_per_orientation_test_W > max_shapes_per_orientation_test_W:
                logger.info("Reduced number of evaluation shapes for W from %d to %d to satisfy divisibility of %d" % (shapes_per_orientation_test_W, max_shapes_per_orientation_test_W, divisibility_test_W))
                shapes_per_orientation_test_W = max_shapes_per_orientation_test_W
            shapes_test = []
            for i in range(0,shapes_per_orientation_test_H):
                size_H = min_size_test_H + ((i+1) * size_range_test_H // shapes_per_orientation_test_H) * divisibility_test_H
                shapes_test.append( (size_H, min_size_test_W) )
            for i in range(0,shapes_per_orientation_test_W):
                size_W = min_size_test_W + ((i+1) * size_range_test_W // shapes_per_orientation_test_W) * divisibility_test_W
                shapes_test.append( (min_size_test_H, size_W) )
            logger.info("testing shapes are %s" % (str(shapes_test))) 
    else:
        shapes_train = None
        shapes_test = None

    if cfg.USE_CUDA_GRAPH:
        if cfg.MODEL.BACKBONE.DONT_RECOMPUTE_SCALE_AND_BIAS:
            model_train.compute_scale_bias() # Enable caching of scale and bias for frozen batchnorms
            model_test.compute_scale_bias()

        if is_training_rank and is_evaluation_rank:
            per_gpu_batch_sizes = [(True, images_per_gpu_train), (False, images_per_gpu_test)]
        elif is_training_rank:
            per_gpu_batch_sizes = [(True, images_per_gpu_train)]
        elif is_evaluation_rank:
            per_gpu_batch_sizes = [(False, images_per_gpu_test)]
        else:
            assert(False), "%d :: Weird state - rank is neither training nor evaluation" % get_rank()
        print("USE_CUDA_GRAPH :: per_gpu_batch_sizes = %s" % (str(per_gpu_batch_sizes)))

        graphed_forwards_train, graphed_forwards_test = {}, {}
        graph_stream = torch.cuda.Stream()
        # Peer pool needs to be reset for each batch during training and evaluation,
        # but this is not required if peer pool is only used inside graphed sections.
        # In the latter case, it is only necessary to keep peer pool alive for the duration
        # of the graphed section, to prevent memory from being deallocated.
        #
        # Potential problem: During graph capture, we do multiple warm-up iterations,
        # without possibility of resetting peer pool in between. This means each step
        # will allocate a fresh set of tensors in CUDA memory. Once graph is captured,
        # peer memory consumption stops because all peer memory buffers are frozen.
        # Workaround for now is to make peer pool large enough to handle this.
        # 
        # This dict keeps peer pool(s) alive until training has finished.
        for (is_training, images_per_gpu) in per_gpu_batch_sizes:
            if is_training:
                model = model_train
                model.enable_train()
                shapes = shapes_train
                spatial_group_size = spatial_group_size_train
                spatial_group_rank = rank_in_group_train
            else:
                model = model_test
                model.enable_eval()
                shapes = shapes_test
                spatial_group_size = spatial_group_size_test
                spatial_group_rank = rank_in_group_test
            for i, shape in enumerate(shapes):
                torch.cuda.empty_cache()
                h, w = shape
                dl = SyntheticDataLoader(device, images_per_gpu, h, w, annotations_per_image if annotations_per_image > 0 else 10, 10,
                        cfg.DATALOADER.MASK_LOSS_FROM_GLOBAL_TENSORS and is_training, cfg.DATALOADER.MASK_LOSS_FROM_GLOBAL_NUMELS)
                dummy_batch, (dummy_target_bboxes, dummy_target_objectness, dummy_target_labels, dummy_targets) = next(dl)
                if cfg.NHWC:
                    dummy_batch.tensors = nchw_to_nhwc_transform(dummy_batch.tensors)
                dummy_shape = tuple(list(dummy_batch.tensors.shape))
                if batch_size_one and is_training:
                    if mask_loss_from_global:
                        dummy_target = dummy_targets[0]
                        sample_args = (dummy_batch.tensors.clone(),dummy_batch.image_sizes_tensor.clone(),dummy_target_bboxes,dummy_target_objectness,dummy_target_labels,
                                dummy_target.get_field('target_index'),
                                dummy_target.get_field('after_transforms_img_infos'),
                                dummy_target.get_field('after_transforms_indexes'),
                                dummy_target.get_field('after_transforms_dense_xy'),
                                )
                    else:
                        sample_args = (dummy_batch.tensors.clone(),dummy_batch.image_sizes_tensor.clone(),dummy_target_bboxes,dummy_target_objectness,dummy_target_labels,)
                else:
                    sample_args = (dummy_batch.tensors.clone(),dummy_batch.image_sizes_tensor.clone(),)

                forward_fn = "graph_forward_%s_%d_%d" % ("train" if is_training else "test", images_per_gpu, i+1)
                #print("%d :: %s" % (get_rank(), forward_fn))
                if i == 0:
                    model.graphable = graph(model.graphable,
                                           sample_args,
                                           graph_stream=graph_stream,
                                           warmup_only=True,
                                           overwrite_fn='eager_forward')
                    model.graphable, pool_id = graph(model.graphable,
                                                    sample_args,
                                                    graph_stream=graph_stream,
                                                    warmup_only=False,
                                                    overwrite_fn=forward_fn,
                                                    return_pool_id=True)
                else:
                    model.graphable = graph(model.graphable,
                                           sample_args,
                                           graph_stream=graph_stream,
                                           warmup_only=False,
                                           overwrite_fn=forward_fn,
                                           use_pool_id=pool_id)
                if is_training:
                    graphed_forwards_train[dummy_shape] = getattr(model.graphable, forward_fn)
                else:
                    graphed_forwards_test[dummy_shape] = getattr(model.graphable, forward_fn)
        del shapes # make sure we don't accidentally use this local variable instead of shapes_train / shapes_test
        model = None
        torch.cuda.empty_cache()

        class GraphedWrapper(torch.nn.Module):
            def __init__(self, model_segment, expected_batch_size, graphed_forwards):
                super().__init__()
                self.model_segment = model_segment
                self.expected_batch_size = expected_batch_size
                self.graphed_forwards = graphed_forwards

            def pad_incomplete_batch(self, graphed_forwards, shape, expected_batch_size, tensor, sizes_tensor, *argv):
                if shape in graphed_forwards:
                    #print("%d :: (wtf) tensor=%s, sizes_tensor=%s, argv=%s" % (get_rank(), str(list(tensor.shape)), str(list(sizes_tensor.shape)), str([list(x.shape) for x in argv])))
                    return graphed_forwards[shape](tensor, sizes_tensor, *argv)
                elif tensor.shape[0] < expected_batch_size:
                    # pad
                    before_pad = tensor.shape[0]
                    tensor = torch.nn.functional.pad(tensor, (0,0,0,0,0,0,0,expected_batch_size-before_pad))
                    sizes_tensor = torch.nn.functional.pad(sizes_tensor, (0,0,0,expected_batch_size-before_pad))
                    # run with graph
                    shape = tuple(list(tensor.shape))
                    if shape in graphed_forwards:
                        out = graphed_forwards[shape](tensor, sizes_tensor, *argv)
                    else:
                        out = self.model_segment.eager_forward(tensor, sizes_tensor, *argv)
                    # unpad
                    out = [o[0:before_pad] for o in out]
                    return out
                else:
                    return self.model_segment.eager_forward(tensor, sizes_tensor, *argv)

            def forward(self, images_tensor, image_sizes_tensor, *argv):
                shape = tuple(list(images_tensor.shape))
                return self.pad_incomplete_batch(self.graphed_forwards, shape, self.expected_batch_size, images_tensor, image_sizes_tensor, *argv)

        model_train.graphable = GraphedWrapper(model_train.graphable, images_per_gpu_train, graphed_forwards_train)
        model_test.graphable = GraphedWrapper(model_test.graphable, images_per_gpu_test, graphed_forwards_test)
    # ------------------

    optimizer = FusedSGD(model_train, cfg, 1.0/(num_training_ranks*spatial_group_size_train), distributed, None if dedicated_evaluation_ranks == 0 else training_comm)
    # Optimizer logging
    mllogger.event(key=mllogger.constants.OPT_NAME, value="sgd_with_momentum")
    mllogger.event(key=mllogger.constants.OPT_BASE_LR, value=cfg.SOLVER.BASE_LR)
    mllogger.event(key=mllogger.constants.OPT_LR_WARMUP_STEPS, value=cfg.SOLVER.WARMUP_ITERS)
    mllogger.event(key=mllogger.constants.OPT_LR_WARMUP_FACTOR, value=cfg.SOLVER.WARMUP_FACTOR)
    mllogger.event(key=mllogger.constants.OPT_LR_DECAY_FACTOR, value=cfg.SOLVER.GAMMA)
    mllogger.event(key=mllogger.constants.OPT_LR_DECAY_STEPS, value=cfg.SOLVER.STEPS)
    mllogger.event(key=mllogger.constants.MIN_IMAGE_SIZE, value=cfg.INPUT.MIN_SIZE_TRAIN[0])
    mllogger.event(key=mllogger.constants.MAX_IMAGE_SIZE, value=cfg.INPUT.MAX_SIZE_TRAIN)

    # disable the garbage collection
    gc.disable()

    if distributed:
        # master rank broadcasts parameters
        params = list(model_train.parameters())
        flat_params = flatten_model(params)
        torch.distributed.broadcast(flat_params, 0)
        overflow_buf = torch.zeros([1], dtype=torch.int32, device='cuda')
        multi_tensor_applier(
                amp_C.multi_tensor_scale,
                overflow_buf,
                [unflatten_model(flat_params, params), params],
                1.0)

    arguments = {}
    arguments["iteration"] = 0
    arguments["nhwc"] = cfg.NHWC
    arguments['ims_per_batch'] = cfg.SOLVER.IMS_PER_BATCH
    arguments["distributed"] = distributed
    arguments["max_annotations_per_image"] = cfg.DATALOADER.MAX_ANNOTATIONS_PER_IMAGE
    arguments["dedicated_evaluation_ranks"] = dedicated_evaluation_ranks
    arguments["num_training_ranks"] = num_training_ranks
    arguments["training_comm"] = None if dedicated_evaluation_ranks == 0 else training_comm
    arguments["images_per_gpu_train"] = images_per_gpu_train
    arguments["additional_meters"] = cfg.ADDITIONAL_METERS
    arguments["use_synthetic_input"] = cfg.DATALOADER.USE_SYNTHETIC_INPUT
    assert not (cfg.DATALOADER.USE_SYNTHETIC_INPUT and cfg.DATALOADER.HYBRID), "USE_SYNTHETIC_INPUT and HYBRID can't both be used together"
    arguments["enable_nsys_profiling"] = cfg.ENABLE_NSYS_PROFILING
    # Pass training peer pool to training loop so it can be reset every iteration.
    # This is not necessary for maskrcnn since peer pool is only used inside graphed sections.
    #arguments["peer_pool"] = peer_pools[True]
    arguments["spatial_group_size"] = spatial_group_size_train
    arguments["cuda_profiler_api_profiling"] = cfg.CUDA_PROFILER_API_PROFILING
    arguments["save_gradients"] = cfg.DEBUG_SAVE_GRADIENTS
    arguments["mask_loss_from_global"] = cfg.DATALOADER.MASK_LOSS_FROM_GLOBAL_TENSORS
    arguments["global_target_tensor_nels"] = cfg.DATALOADER.MASK_LOSS_FROM_GLOBAL_NUMELS
    arguments["sustained_training_time"] = cfg.SUSTAINED_TRAINING_TIME
    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model_train, optimizer, None, output_dir, save_to_disk
    )
    arguments["save_checkpoints"] = cfg.SAVE_CHECKPOINTS
    
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, cfg.NHWC)
    arguments.update(extra_checkpoint_data)
    optimizer.update_master_params()

    if cfg.DEBUG_DETERMINISTIC:
        fname = "/workspace/current_dir/reference_model.pt"
        with open(fname, "rb") as f:
            with torch.no_grad():
                reference_model = dict(torch.load(f))
                for k,p in dict(model_train.named_parameters()).items():
                    k = k.replace("_base_stem_hsplit", "_base_stem")
                    k = k.replace("_base_stem_wsplit", "_base_stem")
                    p.copy_(reference_model[k])
        print("%d :: Loaded reference model from %s" % (get_rank(), fname))

    copy_model(model_train, model_test)
    if cfg.MODEL.BACKBONE.DONT_RECOMPUTE_SCALE_AND_BIAS:
        model_train.compute_scale_bias() # recompute scale and bias for frozen batchnorms after loading checkpoint
        # TODO: Do this for model_test before every eval
        model_test.compute_scale_bias()

    # allocate shared pinned memory image transfer buffers. It is an expensive operation, and no knowledge of the dataset is required,
    # hence we do it here to save some time
    if rank < num_training_ranks * spatial_group_size_train:
        hybrid_dataloader = HybridDataLoader3(cfg, images_per_gpu_train, cfg.DATALOADER.SIZE_DIVISIBILITY, shapes_train, spatial_group_size_train, rank_in_group_train, H_split) if cfg.DATALOADER.HYBRID else None

    # run dummy data through model to trigger cudnn rt fusion compilation
    if rank < num_training_ranks * spatial_group_size_train:
        if shapes_train is not None:
            logger.info("Running dummy shapes through finished training model to trigger cudnn rt fusion")
            for (h, w) in shapes_train:
                #def __init__(self, device, bs, img_h, img_w, annotations_per_image, max_iter):
                dl = SyntheticDataLoader(device, images_per_gpu_train, h, w, annotations_per_image if annotations_per_image > 0 else 10, 10,
                        cfg.DATALOADER.MASK_LOSS_FROM_GLOBAL_TENSORS, cfg.DATALOADER.MASK_LOSS_FROM_GLOBAL_NUMELS)
                for iteration, (images, targets) in enumerate(dl, 1):
                    if mask_loss_from_global and len(targets) == 1:
                        target = targets[0]
                    images = to_image_list(images)
                    images.tensors = nchw_to_nhwc_transform(images.tensors)
                    loss_dict = model_train(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    for p in model_train.parameters():
                        if p.requires_grad:
                            p.grad = None

    torch.cuda.empty_cache()

    # will sync loggers between each rank
    mllogger.log_init_stop_run_start()

    if rank < num_training_ranks * spatial_group_size_train:
        if dedicated_evaluation_ranks > 0:
            # launch dummy eval of epoch 0 to initialize buffers for evaluation pipeline
            launch_eval_on_dedicated_ranks(model_train, model_test, 0, 0)

        data_loader, iters_per_epoch = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=arguments["iteration"],
            random_number_generator=random_number_generator,
            seed=seed,
            shapes=shapes_train,
            hybrid_dataloader=hybrid_dataloader,
            comm=training_comm,
            master_rank=0
        )
        mllogger.event(key=mllogger.constants.TRAIN_SAMPLES, value=len(data_loader))
        num_iteration_to_run_eval = cfg.NUM_ITERATION_TO_RUN_EVAL
        checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

        # set the callback function to evaluate and potentially
        # early exit each epoch
        if cfg.PER_EPOCH_EVAL:
            if dedicated_evaluation_ranks == 0:
                per_iter_callback_fn = functools.partial(
                        mlperf_test_early_exit,
                        iters_per_epoch=iters_per_epoch,
                        num_iteration_to_run_eval=num_iteration_to_run_eval,
                        tester=functools.partial(test, cfg=cfg, shapes=shapes_test, H_split=H_split),
                        model_train=model_train,
                        model_test=model_test,
                        dont_recompute_scale_bias=cfg.MODEL.BACKBONE.DONT_RECOMPUTE_SCALE_AND_BIAS,
                        distributed=distributed,
                        min_bbox_map=cfg.MLPERF.MIN_BBOX_MAP,
                        min_segm_map=cfg.MLPERF.MIN_SEGM_MAP,
                        world_size=world_size,
                        data_loader=data_loader)
                final_callback_fn = None
            else:
                # make sure DEDICATED_EVALUATION_WAIT_FOR_RESULT_ITERATIONS is valid
                early_wait, early_epoch, late_wait = cfg.DEDICATED_EVALUATION_WAIT_FOR_RESULT_ITERATIONS
                if early_wait <= 0 or early_wait >= iters_per_epoch:
                    early_wait = iters_per_epoch - 1
                if late_wait <= 0 or late_wait >= iters_per_epoch:
                    late_wait = iters_per_epoch - 1
                early_epoch = max(0,early_epoch)
                dedicated_evaluation_wait_for_result = (early_wait, early_epoch, late_wait,)
                if is_main_process():
                    logger = logging.getLogger('maskrcnn_benchmark.trainer')
                    logger.info("Using %d dedicated evaluation ranks. Polling will happen after %d steps for first %d epochs and then every %d steps." % (dedicated_evaluation_ranks, early_wait, early_epoch, late_wait))

                # per_iter_callback_fn does two things
                # broadcast parameters with grads from rank 0
                # after N training iterations:
                #   wait for broadcast of evaluation result from evaluation master rank
                per_iter_callback_fn = functools.partial(
                        mlperf_training_test_early_exit,
                        iters_per_epoch=iters_per_epoch,
                        training_ranks_comm=training_comm,
                        num_training_ranks=num_training_ranks,
                        spatial_group_size_train=spatial_group_size_train,
                        model_train=model_train,
                        model_test=model_test,
                        data_loader=data_loader,
                        wait_this_many_iterations_before_checking_result=dedicated_evaluation_wait_for_result)
                final_callback_fn = functools.partial(
                        terminate_evaluation_ranks,
                        iters_per_epoch=iters_per_epoch,
                        training_ranks_comm=training_comm,
                        num_training_ranks=num_training_ranks,
                        spatial_group_size_train=spatial_group_size_train,
                        model_train=model_train,
                        model_test=model_test,
                        data_loader=data_loader,
                        wait_this_many_iterations_before_checking_result=dedicated_evaluation_wait_for_result)
        else:
            per_iter_callback_fn = None
            final_callback_fn = None

        start_train_time = time.time()

        success = do_train(
            model_train,
            data_loader,
            optimizer,
            checkpointer,
            device,
            checkpoint_period,
            arguments,
            cfg.DISABLE_REDUCED_LOGGING,
            cfg.DISABLE_LOSS_LOGGING,
            per_iter_start_callback_fn=functools.partial(mlperf_log_epoch_start, iters_per_epoch=iters_per_epoch),
            per_iter_end_callback_fn=per_iter_callback_fn,
            final_callback_fn=final_callback_fn,
            rank=rank
        )

        end_train_time = time.time()
        total_training_time = end_train_time - start_train_time

        throughput = "{:.4f}".format((arguments["iteration"] * cfg.SOLVER.IMS_PER_BATCH) / total_training_time )
        print(
                f"&&&& MLPERF METRIC THROUGHPUT={throughput} samples / s"
        )
        return model_train, success, throughput
    else:
        # evaluation rank enters loop where it:
        # waits for model broadcast
        # evaluates
        # broadcast result from evaluation master rank
        mlperf_evaluation_test_loop(
                tester=functools.partial(test, cfg=cfg, shapes=shapes_test, eval_ranks_comm=evaluation_comm, H_split=H_split),
                model_test=model_test, 
                distributed=distributed,
                eval_ranks_comm=evaluation_comm, 
                dedicated_evaluation_ranks=dedicated_evaluation_ranks, 
                num_training_ranks=num_training_ranks, 
                spatial_group_size_train=spatial_group_size_train,
                min_bbox_map=cfg.MLPERF.MIN_BBOX_MAP, 
                min_segm_map=cfg.MLPERF.MIN_SEGM_MAP,
                world_size=world_size)

        #print(
        #        "Evaluation rank %d/%d shutting down" % (rank-num_training_ranks, dedicated_evaluation_ranks)
        #)
        return model_test, False, -1


def main():

    mllogger.start(key=mllogger.constants.INIT_START)

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )


    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    # if is_main_process:
    #     # Setting logging file parameters for compliance logging
    #     os.environ["COMPLIANCE_FILE"] = '/MASKRCNN_complVv0.5.0_' + str(datetime.datetime.now())
    #     constants.LOG_FILE = os.getenv("COMPLIANCE_FILE")
    #     constants._FILE_HANDLER = logging.FileHandler(mllogger.constants.LOG_FILE)
    #     constants._FILE_HANDLER.setLevel(logging.DEBUG)
    #     constants.LOGGER.addHandler(constants._FILE_HANDLER)

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        world_size = get_world_size()
        rank = get_rank()
        # TMJ - It is of critical importance for SHARP that the training communicator all_reduce is the first collective called.
        #       That happens in this function. DO NOT MOVE IT.
        training_comm, evaluation_comm = initialize_nccl_comms(cfg)
        # setting seeds - needs to be timed, so after RUN_START
        if is_main_process():
            master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
            seed_tensor = torch.tensor(master_seed, dtype=torch.float32, device=torch.device("cuda"))
        else:
            seed_tensor = torch.tensor(0, dtype=torch.float32, device=torch.device("cuda"))

        torch.distributed.broadcast(seed_tensor, 0)
        master_seed = int(seed_tensor.item())
    else:
        world_size = 1
        rank = 0
        training_comm, evaluation_comm = None, None
        # random master seed, random.SystemRandom() uses /dev/urandom on Unix
        master_seed = random.SystemRandom().randint(0, 2 ** 32 - 1)
    # override master_seed for reproducibility
    if cfg.DEBUG_DETERMINISTIC:
        master_seed = 54637

    # actually use the random seed
    args.seed = master_seed
    # random number generator with seed set to master_seed
    random_number_generator = random.Random(master_seed)
    mllogger.event(key=mllogger.constants.SEED, value=master_seed)

    dedicated_evaluation_ranks = cfg.DEDICATED_EVALUATION_RANKS
    num_training_ranks = world_size - dedicated_evaluation_ranks

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank(), dedicated_evaluation_ranks=dedicated_evaluation_ranks)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # generate worker seeds, one seed for every distributed worker
    worker_seeds = generate_seeds(random_number_generator, world_size)

    # todo sharath what if CPU
    # broadcast seeds from rank=0 to other workers
    if world_size > 1:
        worker_seeds = broadcast_seeds(worker_seeds, device='cuda')
    
    # get spatial parallel rank
    _, _, _, _, rank_train, _, _, _, _, _, _, _, _ = per_gpu_batch_size(cfg)

    # Setting worker seeds
    logger.info("Worker {}: Setting seed {}".format(rank, worker_seeds[rank]))
    seed_rank = rank_train if rank_train >= 0 else rank
    torch.manual_seed(worker_seeds[seed_rank])
    random.seed(worker_seeds[seed_rank])
    np.random.seed(worker_seeds[seed_rank])

    if is_main_process():
        # collect_env_info() crashes when run on multiple ranks
        logger.info("Collecting env info (might take some time)")
        logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # Initialise async eval
    init()

    mllogger.event(key='d_batch_size', value=cfg.SOLVER.IMS_PER_BATCH/num_training_ranks)

    model, success, throughput = train(cfg, rank, world_size, args.distributed, training_comm, evaluation_comm, random_number_generator, seed=worker_seeds[seed_rank])

    if rank < num_training_ranks and success is not None:
        if not success:
            mllogger.log_run_stop(status=mllogger.constants.ABORTED)

if __name__ == "__main__":
    start = time.time()
    torch.set_num_threads(1)
    main()
    if torch.distributed.is_initialized():
        torch.distributed.barrier() # prevent evaluation ranks from terminating before training ranks
    else:
        torch.cuda.synchronize()
    print("&&&& MLPERF METRIC TIME=", time.time() - start)
