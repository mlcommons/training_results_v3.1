# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2023 NVIDIA CORPORATION. All rights reserved.
import datetime
import logging
import time
import math
import os

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

import apex_C, amp_C
from apex.multi_tensor_apply import multi_tensor_applier

from maskrcnn_benchmark.utils.comm import get_rank, get_world_size, is_main_process, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.mlperf_logger import mllogger, barrier
from maskrcnn_benchmark.structures.image_list import ImageList, to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from mlperf_common.frameworks.pyt import PyTProfilerHandler, PyTCommunicationHandler
from mlperf_common.scaleoutbridge import init_bridge, ScaleoutBridgeBase as SBridge

from apex import amp


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


# Instead of zeroing, set parameter grads to None
# Prevents extraneous copy as we're not accumulating
def set_grads_to_none(model):
    for param in model.parameters():
        param.grad = None


class SyntheticDataLoader:
    """
    Generate ramdom data for profiling purpose
    """
    def __init__(self, device, bs, img_h, img_w, annotations_per_image, max_iter, mask_loss_from_global, global_target_tensor_nels):
        self.device = device
        self.data_shape = (bs, 3, img_h, img_w)
        self.batch_size, self.c, self.h, self.w = self.data_shape
        self.annotations_per_image = annotations_per_image
        self.cur_iter = 0
        self.max_iter = max_iter
        self.images = None
        self.targets = None
        self.target_bboxes = None
        self.target_objectness = None
        self.target_labels = None
        self.bbox_min_w = 20  # min bbox width
        self.bbox_min_h = 20  # min bbox height
        self.num_labels = 80  # coco 2017
        self.data = []
        self.mask_loss_from_global = mask_loss_from_global
        self.global_target_tensor_nels = global_target_tensor_nels

    def _gen_random_image_list(self):
        return to_image_list(torch.empty(self.data_shape).uniform_(-1, 1).half()).to(
            self.device
        )

    def _gen_random_bbox(self):
        while True:
            try:
                x_tl = torch.empty(1).uniform_(0, self.w)
                y_tl = torch.empty(1).uniform_(0, self.h)
                x_br = torch.empty(1).uniform_(x_tl.item() + self.bbox_min_w, self.w)
                y_br = torch.empty(1).uniform_(y_tl.item() + self.bbox_min_h, self.h)
                if x_br.item() < self.w and y_br.item() < self.w:
                    break
            except:
                continue
        return torch.tensor([x_tl, y_tl, x_br, y_br]).flatten().to(self.device)

    def _gen_polygon_from_bbox(self, bbox):
        x_tl, y_tl, x_br, y_br = (
            bbox[0].item(),
            bbox[1].item(),
            bbox[2].item(),
            bbox[3].item(),
        )
        w = x_br - x_tl
        h = y_br - y_tl
        return torch.tensor(
            [
                [
                    x_tl + w / 4,
                    y_tl + h / 4,
                    x_br - w / 4,
                    y_tl + h / 4,
                    x_br - w / 4,
                    y_br - w / 4,
                    x_tl + w / 4,
                    y_br - w / 4,
                ]
            ]
        ).to(self.device)

    def _gen_random_images(self):
        images = self._gen_random_image_list()
        return images

    def _pad_targets(self, targets):
        target_bboxes = torch.stack([target.bbox for target in targets]).to(self.device)
        target_objectness = torch.stack(
            [
                torch.ones(target.bbox.shape[0], device=target.bbox.device)
                for target in targets
            ]
        ).to(self.device)
        target_labels = (
            torch.stack([target.get_field("labels") for target in targets]).to(self.device)
        )
        return target_bboxes, target_objectness, target_labels, targets

    def _gen_random_targets(self):
        targets = []
        for img_idx in range(self.batch_size):
            bboxes = []
            masks = []
            for box_idx in range(self.annotations_per_image):
                bboxes.append(self._gen_random_bbox())
            for bbox in bboxes:
                masks.append(self._gen_polygon_from_bbox(bbox))
            labels = (
                torch.randint(1, self.num_labels + 1, (len(bboxes),))
                .type(torch.float32)
                .to(self.device)
            )

            # assign to target
            target = BoxList(
                torch.stack(bboxes).to(self.device), (self.w, self.h), mode="xyxy"
            )

            target.add_field("labels", labels)

            masks = SegmentationMask(masks, (self.w, self.h))
            target.add_field("masks", masks)
            targets.append(target)
        return targets

    def _add_global_target_tensors(self, targets):
        # We do this for every target in the synthetic loader, in the real loader we do it once for all targets.
        # We pad global tensors so they are the same size as the truly global ones generated by real loader.
        index = 0
        for target in targets:
            header_size = 32
            header = [0]*header_size
            header[0] = header_size
            header[2] = 1 # flag indicating that annotations are included in targets
            header[3] = 1 # total number of targets (in global tensors)

            per_image_mask_idx = [] # index of first mask for image
            per_mask_poly_idx = [] # index of first polygon for mask
            per_poly_sample_idx = [] # index of first sample for polygon
            dense_xy = [] # polygon samples, densely packed

            per_image_mask_idx.append(len(per_mask_poly_idx))
            #print("i ==%d :: per_image_mask_idx[-1] = %d" % (i, per_image_mask_idx[-1]))
            for mask in target.get_field('masks'):
                #print("mask=",mask)
                per_mask_poly_idx.append(len(per_poly_sample_idx))
                #print("i == %d :: per_mask_poly_idx[-1] = %d" % (i, per_mask_poly_idx[-1]))
                for poly in mask.polygons:
                    #print("poly=",poly)
                    per_poly_sample_idx.append(len(dense_xy))
                    #print("i == %d :: per_poly_sample_idx[-1] = %d" % (i, per_poly_sample_idx[-1]))
                    dense_xy.extend(poly.tolist())

            per_image_mask_idx.append(len(per_mask_poly_idx))
            per_mask_poly_idx.append(len(per_poly_sample_idx))
            per_poly_sample_idx.append(len(dense_xy))
            nn = len(header) + len(per_image_mask_idx)
            per_image_mask_idx = [i+nn for i in per_image_mask_idx]
            nn = nn + len(per_mask_poly_idx)
            per_mask_poly_idx = [i+nn for i in per_mask_poly_idx]

            # Cast dense_xy to fp32 by passing it through FloatTensor
            # Done to prevent small round-off errors, since values will be stored as FloatTensor
            dense_xy = torch.FloatTensor(dense_xy)

            indexes = torch.IntTensor(header+per_image_mask_idx+per_mask_poly_idx+per_poly_sample_idx)

            img_infos, bboxes = [], []
            img_infos.extend( [self.h, self.w, index, len(bboxes)] )
            bboxes.extend( target.bbox.flatten().tolist() )
            bboxes.extend( target.get_field('labels').tolist() )
            # add dummy img_info tuple so reader can determine length of bbox tensor for last index
            img_infos.extend( [0,0,0,len(bboxes)] )
            img_infos = torch.IntTensor(img_infos)
            bboxes = torch.FloatTensor(bboxes)

            # pad
            _, after_transforms_img_infos_numel, indexes_numel,  after_transforms_dense_xy_numel, max_num_polygons = self.global_target_tensor_nels
            img_infos = pad(img_infos, [0,after_transforms_img_infos_numel-img_infos.numel()], value=0).cuda().reshape([-1,4])
            indexes = pad(indexes, [0,indexes_numel-indexes.numel()], value=0).cuda()
            dense_xy = pad(dense_xy, [0,after_transforms_dense_xy_numel-dense_xy.numel()], value=0).cuda()

            target.add_field('target_index', torch.tensor([index], dtype=torch.int32, pin_memory=True).to(device='cuda', non_blocking=True).detach())
            target.add_field('after_transforms_img_infos', img_infos.detach())
            target.add_field('after_transforms_indexes', indexes.detach())
            target.add_field('after_transforms_dense_xy', dense_xy.detach())
            target.add_field('max_num_polygons', max_num_polygons)
        return targets

    def __iter__(self):
        return self

    def __next__(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            targets = self._gen_random_targets()
            if self.mask_loss_from_global:
                targets = self._add_global_target_tensors(targets)
            return self._gen_random_images(), self._pad_targets(targets)
        else:
            (
                self.images,
                self.targets,
                self.target_bboxes,
                self.target_objectness,
                self.target_labels,
            ) = (None, None, None, None, None)
            self.cur_iter = 0
            raise StopIteration()

    def prefetch_GPU(self):
        return
    def prefetch_CPU(self):
        return


class Prefetcher:
    def __init__(self, data_loader, device, max_annotations_per_image):
        self.data_loader = iter(data_loader)
        self.device = device
        self.max_annotations_per_image = max_annotations_per_image
        self.images = None
        self.targets = None
        self.target_bboxes = None
        self.target_objectness = None
        self.target_labels = None
        self.loader_stream = torch.cuda.Stream()
        self.done = False

    def create_padded_target_tensors(self, targets):
        if targets is None:
            target_objectness, target_bboxes, target_labels = None, None, None
        else:
            num_images = len(targets)
            target_objectness = [
                torch.ones(target.bbox.shape[0], device=target.bbox.device)
                for target in targets
            ]
            target_labels = [target.get_field("labels") for target in targets]
            if num_images > 1 or self.max_annotations_per_image <= 0:
                target_bboxes = pad_sequence(
                    [target.bbox for target in targets],
                    batch_first=True,
                    padding_value=-1,
                )
                target_objectness = pad_sequence(
                    target_objectness, batch_first=True, padding_value=-1
                )
                target_labels = pad_sequence(
                    target_labels, batch_first=True, padding_value=-1
                )
            else:
                target_bboxes = targets[0].bbox
                target_objectness = target_objectness[0]
                target_labels = target_labels[0]
                if self.max_annotations_per_image > 0:
                    # shapes_before = str([list(target_bboxes.shape), list(target_objectness.shape), list(target_labels.shape)])
                    num_anno = target_objectness.shape[0]
                    target_bboxes = pad(
                        target_bboxes,
                        [0, 0, 0, self.max_annotations_per_image - num_anno],
                        value=-1,
                    )
                    target_objectness = pad(
                        target_objectness,
                        [0, self.max_annotations_per_image - num_anno],
                        value=-1,
                    )
                    target_labels = pad(
                        target_labels,
                        [0, self.max_annotations_per_image - num_anno],
                        value=-1,
                    )
                    # shapes_after = str([list(target_bboxes.shape), list(target_objectness.shape), list(target_labels.shape)])
                    # print("%s -> %s" % (shapes_before, shapes_after))
                target_bboxes = target_bboxes.unsqueeze(0)
                target_objectness = target_objectness.unsqueeze(0)
                target_labels = target_labels.unsqueeze(0)
            target_bboxes.requires_grad = False
            target_objectness.requires_grad = False
            target_labels.requires_grad = False
            # print("self.max_annotations_per_image = %d, target_bboxes = %s,%s, target_objectness = %s,%s, target_labels = %s,%s" % (self.max_annotations_per_image, str(list(target_bboxes.shape)), str(target_bboxes.dtype), str(list(target_objectness.shape)), str(target_objectness.dtype), str(list(target_labels.shape)), str(target_labels.dtype)))
        return target_bboxes, target_objectness, target_labels

    def __iter__(self):
        return self

    def prefetch_CPU(self):
        try:
            with torch.no_grad():
                with torch.cuda.stream(self.loader_stream):
                    self.images, self.targets, _ = next(self.data_loader)
                    self.target_bboxes, self.target_objectness, self.target_labels = (
                        None,
                        None,
                        None,
                    )
        except StopIteration:
            (
                self.images,
                self.targets,
                self.target_bboxes,
                self.target_objectness,
                self.target_labels,
            ) = (None, None, None, None, None)
            self.done = True

    def prefetch_GPU(self):
        if self.images is not None:
            with torch.no_grad():
                with torch.cuda.stream(self.loader_stream):
                    self.images = self.images.to(self.device)
                    self.targets = [
                        target.to(self.device, non_blocking=True)
                        for target in self.targets
                    ]
                    (
                        self.target_bboxes,
                        self.target_objectness,
                        self.target_labels,
                    ) = self.create_padded_target_tensors(self.targets)

    def __next__(self):
        if self.images is None and not self.done:
            self.prefetch_CPU()
            self.prefetch_GPU()
        torch.cuda.current_stream().wait_stream(self.loader_stream)
        if self.done:
            raise StopIteration()
        else:
            targets = self.targets
            images, target_bboxes, target_objectness, target_labels = (
                self.images,
                self.target_bboxes,
                self.target_objectness,
                self.target_labels,
            )
            (
                self.images,
                self.targets,
                self.target_bboxes,
                self.target_objectness,
                self.target_labels,
            ) = (None, None, None, None, None)
            return images, (target_bboxes, target_objectness, target_labels, targets)


def do_train(
    model,
    data_loader,
    optimizer,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    disable_allreduce_for_logging,
    disable_loss_logging,
    per_iter_start_callback_fn=None,
    per_iter_end_callback_fn=None,
    final_callback_fn=None,
    rank=0,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    distributed = arguments["distributed"]
    num_training_ranks = arguments["num_training_ranks"]
    training_comm = arguments["training_comm"]
    images_per_gpu_train = arguments["images_per_gpu_train"]
    spatial_group_size = arguments["spatial_group_size"]
    additional_meters = arguments["additional_meters"]
    cuda_profiler_api_profiling = arguments["cuda_profiler_api_profiling"]
    save_gradients = arguments["save_gradients"]
    model.enable_train()
    start_training_time = time.time()
    end = time.time()
    if additional_meters:
        prev_time, prev_iteration = end, start_iter - 1

    sbridge = init_bridge(PyTProfilerHandler(), PyTCommunicationHandler(), mllogger)

    synchronize(training_comm)
    optimizer.zero_grad()
    prefetcher = Prefetcher(data_loader, device, arguments["max_annotations_per_image"]) if not arguments["use_synthetic_input"] else SyntheticDataLoader(device, bs=arguments["images_per_gpu_train"], img_h=800, img_w = 1344, annotations_per_image = 10, max_iter = 65535, mask_loss_from_global=arguments["mask_loss_from_global"], global_target_tensor_nels=arguments["global_target_tensor_nels"])

    sbridge.start_epoch_prof()

    vss = []
    if arguments["enable_nsys_profiling"]:
        torch.cuda.cudart().cudaProfilerStart()
    for iteration, (images, targets) in enumerate(prefetcher, start_iter):
        sbridge.start_prof(SBridge.ITER_TIME)
        if cuda_profiler_api_profiling[0] > 0 and cuda_profiler_api_profiling[0] < cuda_profiler_api_profiling[1]:
            if iteration == cuda_profiler_api_profiling[0]:
                torch.cuda.cudart().cudaProfilerStart()
            elif iteration == cuda_profiler_api_profiling[1]:
                torch.cuda.cudart().cudaProfilerStop()

        if per_iter_start_callback_fn is not None:
            per_iter_start_callback_fn(iteration=iteration)

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        sbridge.start_prof(SBridge.FWD_TIME)
        # TMJ: This was a WAR that is no longer needed, so commenting it out
        #if images_per_gpu_train == 1:
        #    if distributed:
        #        torch.distributed.barrier(
        #            group=training_comm
        #        )  # Sync all processes before training starts to prevent CPUs from getting too far ahead of GPUs
        #    else:
        #        torch.cuda.synchronize()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        sbridge.stop_start_prof(SBridge.FWD_TIME, SBridge.BWD_TIME)
        # optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # with optimizer.scale_loss(losses) as scaled_losses:
        optimizer.backward(losses)
        optimizer.copy_gradients()

        # At this point we are waiting for kernels launched by cuda graph to finish, so CPU is idle.
        # Take advantage of this by loading next input batch before calling step
        prefetcher.prefetch_CPU()

        sbridge.stop_start_prof(SBridge.BWD_TIME, SBridge.OPT_TIME)

        prefetcher.prefetch_GPU()
        optimizer.step()

        sbridge.stop_prof(SBridge.OPT_TIME)
        will_report_this_iteration = iteration % 50 == 0 or iteration == max_iter

        # reduce losses over all GPUs for logging purposes
        if not disable_loss_logging and not disable_allreduce_for_logging:
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if math.isfinite(losses_reduced):
                meters.update(loss=losses_reduced, **loss_dict_reduced)
        else:
            # Optimization:
            # Cat all meter updates + finite check if they are all single item tensors
            # This reduces number of D2H transfers to 1.
            ks, vs = zip(*[(k, v.unsqueeze(dim=0)) for (k, v) in loss_dict.items()])
            if disable_loss_logging:
                vs = torch.zeros([len(vs)], dtype=torch.float32)
            else:
                vs = list(vs)
                vs.append(losses.unsqueeze(dim=0))
                vs = torch.cat(vs)
            vss.append(vs)
            if will_report_this_iteration:
                vss = torch.stack(vss).cpu()  # will sync
                for vs in vss:
                    vs = [v.item() for v in list(vs.split(split_size=1))]
                    losses_host = vs.pop(-1)
                    if math.isfinite(losses_host):
                        loss_dict = {k: v for (k, v) in zip(ks, vs)}
                        meters.update(loss=losses_host, **loss_dict)
                vss = []

        # set_grads_to_none(model)
        optimizer.zero_grad()

        now = time.time()
        batch_time = now - end
        end = now
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        sbridge.stop_prof(SBridge.ITER_TIME)

        if will_report_this_iteration:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "loss_scaler: {loss_scaler:.1f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    loss_scaler=optimizer.optimizer_state[2],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

            # for mlperf throughput - DLFW CI/CD

            meter_str = {}
            for name, meter in meters.meters.items():
                meter_str[name] = meter.global_avg
            throughput = arguments["ims_per_batch"] / float(meter_str["time"])
            meter_str["throughput"] = throughput
            if additional_meters:
                average_step_time = (now - prev_time) / (iteration - prev_iteration)
                prev_time, prev_iteration = now, iteration
                instantaneous_throughput = arguments["ims_per_batch"] / average_step_time
                meter_str["instantaneous_throughput"] = instantaneous_throughput
                meter_str["average_step_time"] = average_step_time * 1000
            mllogger.event(
                key="tracked_stats", value=meter_str, metadata={"step": (iteration)}
            )
            mllogger.event(key="throughput", value=throughput)

        if iteration % checkpoint_period == 0 and arguments["save_checkpoints"]:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter and arguments["save_checkpoints"]:
            checkpointer.save("model_final", **arguments)

        # per-epoch work (testing)
        if per_iter_end_callback_fn is not None:
            # Note: iteration has been incremented previously for
            # human-readable checkpoint names (i.e. 60000 instead of 59999)
            # so need to adjust again here
            early_exit, sbridge = per_iter_end_callback_fn(
                iteration=iteration - 1, sbridge=sbridge
            )
            if early_exit:
                break

    if final_callback_fn is not None and not early_exit:
        if final_callback_fn():
            early_exit = True

    sbridge.stop_epoch_prof()
    torch.cuda.cudart().cudaProfilerStop()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (arguments["iteration"])
        )
    )
    if per_iter_end_callback_fn is not None:
        if early_exit:
            return True
        else:
            return False
    else:
        return None
