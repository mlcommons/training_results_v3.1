# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

import amp_C
from apex.multi_tensor_apply import multi_tensor_applier
from apex.contrib.bottleneck import HaloExchangerSendRecv, HaloExchangerPeer, HaloExchangerNoComm
from apex.contrib.peer_memory import PeerMemoryPool

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.layers.nhwc import nchw_to_nhwc_transform, nhwc_to_nchw_transform
from apex.contrib.bottleneck.bottleneck import SpatialBottleneck
from maskrcnn_benchmark.modeling.backbone.resnet import GatherTensor, GatherTensors, BaseStem, _HALO_EXCHANGERS
from maskrcnn_benchmark.utils.batch_size import per_gpu_batch_size
from maskrcnn_benchmark.utils.save import save
from maskrcnn_benchmark.utils.comm import get_rank

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..roi_heads.mask_head.mask_head import keep_only_positive_boxes


class Graphable(nn.Module):
    def __init__(self, cfg):
        super(Graphable, self).__init__()

        self.backbone = build_backbone(cfg)
        from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn_head
        self.anchor_generator, self.head = build_rpn_head(cfg)
        self.nhwc = cfg.NHWC
        self.stream1 = torch.cuda.Stream()

    # Return all modules that are not spatial parallel, i.e. they are doing redundant calculations when spatial_group_size > 1
    def get_redundant_modules(self):
        redm = []
        for i in range(len(self.backbone)):
            if i > 0:
                redm.append(self.backbone[i])
        redm.append(self.anchor_generator)
        redm.append(self.head)
        return redm

    def forward(self, images_tensor, image_sizes_tensor):
        current_stream = torch.cuda.current_stream()
        features = self.backbone(images_tensor)
        features_nchw = [feature.permute([0,3,1,2]) for feature in features] if self.nhwc else features
        self.stream1.wait_stream(current_stream)
        with torch.cuda.stream(self.stream1):
            objectness, rpn_box_regression = self.head(features_nchw)
        with torch.no_grad():
            anchor_boxes, anchor_visibility = self.anchor_generator(image_sizes_tensor.int(), features_nchw)
        current_stream.wait_stream(self.stream1)
        return features + tuple(objectness) + tuple(rpn_box_regression) + (anchor_boxes, anchor_visibility)


class GraphableTrainBS1(nn.Module):
    def __init__(self, cfg):
        super(GraphableTrainBS1, self).__init__()

        self.backbone = build_backbone(cfg)
        from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn_head
        self.anchor_generator, self.head = build_rpn_head(cfg)
        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg, True, False, True)
        self.nhwc = cfg.NHWC
        self.stream1 = torch.cuda.Stream()

    # Return all modules that are not spatial parallel, i.e. they are doing redundant calculations when spatial_group_size > 1
    def get_redundant_modules(self):
        redm = []
        for i in range(len(self.backbone)):
            if i > 0:
                redm.append(self.backbone[i])
        redm.append(self.anchor_generator)
        redm.append(self.head)
        return redm

    def forward(self, images_tensor, image_sizes_tensor, target_bboxes, target_objectness, target_labels):
        current_stream = torch.cuda.current_stream()
        features = self.backbone(images_tensor)
        features_nchw = [feature.permute([0,3,1,2]) for feature in features] if self.nhwc else features
        self.stream1.wait_stream(current_stream)
        with torch.cuda.stream(self.stream1):
            objectness, rpn_box_regression = self.head(features_nchw)
        with torch.no_grad():
            anchor_boxes, anchor_visibility = self.anchor_generator(image_sizes_tensor.int(), features_nchw)
        current_stream.wait_stream(self.stream1)
        batched_anchor_data = [anchor_boxes, anchor_visibility, [(1344,800) for _ in range(image_sizes_tensor.shape[0])]]
        stream1 = torch.cuda.Stream()
        stream1.wait_stream(current_stream)
        targets = [target_bboxes, target_objectness, target_labels, None]
        with torch.cuda.stream(stream1):
            loss_objectness, loss_rpn_box_reg = self.rpn.loss_evaluator(
                batched_anchor_data, objectness, rpn_box_regression, targets
            )
        with torch.no_grad():
            proposals = self.rpn.box_selector_train(
                batched_anchor_data, objectness, rpn_box_regression, image_sizes_tensor, targets
            )
            detections = self.roi_heads.box.loss_evaluator.subsample(proposals, targets)
        x = self.roi_heads.box.feature_extractor(features, detections)
        class_logits, box_regression = self.roi_heads.box.predictor(x)

        loss_classifier, loss_box_reg = self.roi_heads.box.loss_evaluator(
            [class_logits.float()], [box_regression.float()]
        )
        current_stream.wait_stream(stream1)

        assert( len(detections) == 1 )
        detections = detections[0]
        return features + (detections.bbox, detections.extra_fields["matched_idxs"], detections.extra_fields["regression_targets"], detections.extra_fields["labels"], loss_objectness, loss_rpn_box_reg, loss_classifier, loss_box_reg,)


class Combined_RPN_ROI(nn.Module):
    def __init__(self, cfg):
        super(Combined_RPN_ROI, self).__init__()

        self.rpn = build_rpn(cfg)
        self.roi_heads = build_roi_heads(cfg, True, True, True)
        self.take_shortcut = True if not cfg.MODEL.RPN_ONLY and not cfg.MODEL.KEYPOINT_ON and cfg.MODEL.MASK_ON and not cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR else False
        if self.take_shortcut:
            self.stream1 = torch.cuda.Stream()
            self.stream2 = torch.cuda.Stream()
            self.stream3 = torch.cuda.Stream()

    def forward(self, images, anchor_boxes, anchor_visibility, objectness, rpn_box_regression, targets, features):
        if self.take_shortcut:
            if self.training:
                current_stream = torch.cuda.current_stream()

                #print("%d :: anchor_boxes = %s, anchor_visibility = %s, objectness = %s, rpn_box_regression = %s" % (
                #    get_rank(), str(anchor_boxes), str(anchor_visibility), str(objectness.shape), str(rpn_box_regression.shape)))

                #
                # RPN inference, roi subsample
                #

                batched_anchor_data = [anchor_boxes, anchor_visibility, [tuple(image_size_wh) for image_size_wh in images.image_sizes_wh]]
                self.stream1.wait_stream(current_stream)
                with torch.no_grad():
                    proposals = self.rpn.box_selector_train(
                        batched_anchor_data, objectness, rpn_box_regression, images.image_sizes_tensor, targets
                    )

                #
                # loss calculations
                #

                # rpn losses
                with torch.cuda.stream(self.stream1):
                    loss_objectness, loss_rpn_box_reg = self.rpn.loss_evaluator(
                        batched_anchor_data, objectness, rpn_box_regression, targets
                        )

                with torch.no_grad():
                    detections = self.roi_heads.box.loss_evaluator.subsample(proposals, targets)
                self.stream2.wait_stream(current_stream)
                self.stream3.wait_stream(current_stream)

                # box losses
                with torch.cuda.stream(self.stream2):
                    x = self.roi_heads.box.feature_extractor(features, detections)
                    class_logits, box_regression = self.roi_heads.box.predictor(x)

                    loss_classifier, loss_box_reg = self.roi_heads.box.loss_evaluator(
                        [class_logits.float()], [box_regression.float()]
                    )
                    loss_box = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)

                # mask losses
                with torch.cuda.stream(self.stream3):
                    _, _, loss_mask = self.roi_heads.mask(features, detections, targets, syncfree=True)

                current_stream.wait_stream(self.stream1)
                current_stream.wait_stream(self.stream2)
                current_stream.wait_stream(self.stream3)

                losses = {}
                losses.update(loss_box)
                losses.update(loss_mask)
                proposal_losses = {
                    "loss_objectness": loss_objectness,
                    "loss_rpn_box_reg": loss_rpn_box_reg,
                    }
                losses.update(proposal_losses)
    
                return losses
            else:
                batched_anchor_data = [anchor_boxes, anchor_visibility, [tuple(image_size_wh) for image_size_wh in images.image_sizes_wh]]
                proposals = self.rpn.box_selector_test(batched_anchor_data, objectness, rpn_box_regression, images.image_sizes_tensor)

                x = self.roi_heads.box.feature_extractor(features, proposals)
                class_logits, box_regression = self.roi_heads.box.predictor(x)
                detections = self.roi_heads.box.post_processor((class_logits, box_regression), proposals)

                x = self.roi_heads.mask.feature_extractor(features, detections, None)
                mask_logits = self.roi_heads.mask.predictor(x)
                detections = self.roi_heads.mask.post_processor(mask_logits, detections)
                    
                return detections
        else:
            proposals, proposal_losses = self.rpn(images, anchor_boxes, anchor_visibility, objectness, rpn_box_regression, targets)
            if self.roi_heads:
                x, result, detector_losses = self.roi_heads(features, proposals, targets, syncfree=True)
            ## for NHWC layout case, features[0] are NHWC features, and [1] NCHW
            ## when syncfree argument is True, x == None
            else:
                # RPN-only models don't have roi_heads
                ## TODO: take care of NHWC/NCHW cases for RPN-only case 
                x = features
                result = proposals
                detector_losses = {}

            if self.training:
                losses = {}
                losses.update(detector_losses)
                losses.update(proposal_losses)
                return losses

            return result



        flat_res = self.graphable(images.tensors, images.image_sizes_tensor)
        features, objectness, rpn_box_regression, anchor_boxes, anchor_visibility = flat_res[0:5], list(flat_res[5:10]), list(flat_res[10:15]), flat_res[15], flat_res[16]
        return self.combined_rpn_roi(images, anchor_boxes, anchor_visibility, objectness, rpn_box_regression, targets, features)

def single_create_spatial_parallel_args(rank, rank_in_group, num_ranks, spatial_rank_offset, spatial_group_size, halo_ex, static_size, dynamic_size, numSM, spatial_method, use_delay_kernel):
    if spatial_group_size > 1:
        my_comm = None
        for spatial_group in range(num_ranks):
            peer_group_ranks = [spatial_rank_offset+spatial_group*spatial_group_size+i for i in range(spatial_group_size)]
            comm = torch.distributed.new_group(ranks=peer_group_ranks)
            if rank in peer_group_ranks:
                my_comm = comm
            print("%d :: Created spatial communicator for ranks %s" % (rank, str(peer_group_ranks)))
            torch.cuda.synchronize()
            torch.distributed.barrier()
        print("%d :: Line 1" % (rank))
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if rank_in_group < 0:
            # This rank does not participate in training
            peer_ranks = [rank]
        else:
            peer_group = (rank-spatial_rank_offset) // spatial_group_size
            peer_ranks = [spatial_rank_offset+peer_group*spatial_group_size+i for i in range(spatial_group_size)]
        print("%d :: Line 2" % (rank))
        if halo_ex == "HaloExchangerPeer":
            peer_pool = PeerMemoryPool(static_size, dynamic_size, peer_ranks=peer_ranks)
            halo_ex = HaloExchangerPeer(peer_ranks, rank_in_group, peer_pool, numSM)
        elif halo_ex == "HaloExchangerSendRecv":
            peer_pool = None
            print("%d :: Creating HaloExchangerSendRecv for ranks %s" % (rank, str(peer_ranks)))
            halo_ex = HaloExchangerSendRecv(peer_ranks, rank_in_group)
            print("%d :: Created HaloExchangerSendRecv for ranks %s" % (rank, str(peer_ranks)))
        elif halo_ex == "HaloExchangerNoComm":
            peer_pool = None
            print("%d :: Creating haloExchangerNoComm for ranks %s" % (rank, str(peer_ranks)))
            halo_ex = HaloExchangerNoComm(peer_ranks, rank_in_group)
            print("%d :: Created haloExchangerNoComm for ranks %s" % (rank, str(peer_ranks)))
        else:
            assert(False), "Unknown halo exchanger type %s" % (halo_ex)
        print("%d :: Line 3" % (rank))
        spatial_parallel_args = (spatial_group_size, rank_in_group, my_comm, halo_ex, spatial_method, use_delay_kernel,)
        return spatial_parallel_args, peer_pool
    else:
        my_comm = None
        halo_ex = None
        peer_pool = None
    spatial_parallel_args = (spatial_group_size, rank_in_group, my_comm, halo_ex, spatial_method, use_delay_kernel,)
    return spatial_parallel_args, peer_pool


def create_spatial_parallel_args(cfg):
    # TODO: Consider if more properties should be part of cfg?
    numSM = 1
    static_memory_bytes = 64*1024*1024
    dynamic_memory_bytes = 1024*1024*1024
    halo_ex, spatial_method, use_delay_kernel = cfg.MODEL.BACKBONE.HALO_EXCHANGER, cfg.MODEL.BACKBONE.SPATIAL_METHOD, cfg.MODEL.BACKBONE.USE_DELAY_KERNEL
    # end of properties
    rank = get_rank()
    dedi_eval_ranks, num_training_ranks, _, _, _, rank_in_group_train, spatial_group_size_train, num_evaluation_ranks, _, _, _, rank_in_group_test, spatial_group_size_test = per_gpu_batch_size(cfg)
    spatial_parallel_args_train, peer_pool_train = single_create_spatial_parallel_args(
            rank, rank_in_group_train, num_training_ranks, 0, spatial_group_size_train, 
            halo_ex, static_memory_bytes, dynamic_memory_bytes, numSM, spatial_method, use_delay_kernel)
    rank_offset_test = num_training_ranks if dedi_eval_ranks > 0 else 0
    spatial_parallel_args_test, peer_pool_test = single_create_spatial_parallel_args(
            rank, rank_in_group_test, num_evaluation_ranks, rank_offset_test, spatial_group_size_test, 
            halo_ex, static_memory_bytes, dynamic_memory_bytes, numSM, spatial_method, use_delay_kernel)
    return spatial_parallel_args_train, peer_pool_train, spatial_parallel_args_test, peer_pool_test


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg, spatial_blob, batch_size_one):
        super(GeneralizedRCNN, self).__init__()

        self.batch_size_one = batch_size_one
        self.precompute_rpn_constant_tensors = cfg.PRECOMPUTE_RPN_CONSTANT_TENSORS
        if self.batch_size_one:
            self.graphable = GraphableTrainBS1(cfg)
            self.roi_mask_only = build_roi_heads(cfg, False, True, False)
        else:
            self.graphable = Graphable(cfg)
            self.combined_rpn_roi = Combined_RPN_ROI(cfg)
        self.nhwc = cfg.NHWC
        self.dali = cfg.DATALOADER.DALI
        self.hybrid_loader = cfg.DATALOADER.HYBRID
        self.scale_bias_callables = None
        self.mta_scale = amp_C.multi_tensor_scale
        self.spatial_H_split = cfg.MODEL.BACKBONE.SPATIAL_H_SPLIT
        self.spatial_parallel_args_train, self.peer_pool_train, self.spatial_parallel_args_eval, self.peer_pool_eval = spatial_blob
        self.spatial_group_size_train, _, _, _, _, _ = self.spatial_parallel_args_train
        self.spatial_group_size_eval, _, _, _, _, _ = self.spatial_parallel_args_eval
        self.change_spatial_parallel_args(self.spatial_parallel_args_train)

    #
    # Change spatial parallel args for all SpatialBottleneck blocks.
    # This is an acceptable hack to allow spatial parameters to change
    # mid-flight, for instance using a different set of spatial parameters
    # for training and evaluation.
    #
    # Note that spatial parallel args will be frozen in stone when
    # SpatialBottleneck block is part of module that is CUDA graphed.
    # In this context, changing spatial parameters can only be done
    # before graphing.
    #
    def change_spatial_parallel_args(self, spatial_parallel_args):
        self.spatial_group_size, self.spatial_parallel_rank, self.spatial_communicator, _, _, _ = spatial_parallel_args
        self.spatial_parallel_args = spatial_parallel_args
        logger = logging.getLogger('maskrcnn_benchmark.trainer')
        num_spatial, num_gathers, num_base_stem = 0, 0, 0
        base_stem_conv = None
        if getattr(self.graphable, "backbone", None):
            # if self.graphable has backbone attribute, it means self.graphable has not been graphed.
            for m in self.graphable.backbone.modules():
                if isinstance(m, SpatialBottleneck):
                    num_spatial = num_spatial + 1
                    m.spatial_parallel_args = spatial_parallel_args
                elif isinstance(m, GatherTensors) or isinstance(m, GatherTensor):
                    num_gathers = num_gathers + 1
                    m.reconfigure(*spatial_parallel_args)
                elif isinstance(m, BaseStem):
                    num_base_stem = num_base_stem + 1
                    m.reconfigure(self.spatial_group_size, self.spatial_parallel_rank, self.spatial_H_split)
        if num_spatial > 0: logger.info("Changed spatial parallel args for %d SpatialBottleneck modules." % (num_spatial))
        if num_gathers > 0: logger.info("Reconfigured %d GatherTensors modules." % (num_gathers))
        if num_base_stem > 0: logger.info("Reconfigured %d BaseStem modules." % (num_base_stem))

    def enable_train(self):
        self.change_spatial_parallel_args(self.spatial_parallel_args_train)
        self.train()

    def enable_eval(self):
        self.change_spatial_parallel_args(self.spatial_parallel_args_eval)
        self.eval()

    def compute_scale_bias(self):
        if self.scale_bias_callables is None:
            self.scale_bias_callables = []
            for module in self.graphable.modules():
                if getattr(module, "get_scale_bias_callable", None):
                    #print(module)
                    c = module.get_scale_bias_callable()
                    self.scale_bias_callables.append(c)
        for c in self.scale_bias_callables:
            c()

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.batch_size_one and not self.training:
            raise ValueError("batch_size_one is only allowed in training mode")
        if not self.hybrid_loader:
            images = to_image_list(images)
            if self.nhwc and not self.dali:
                # data-loader outputs nchw images
                images.tensors = nchw_to_nhwc_transform(images.tensors)
            elif self.dali and not self.nhwc:
                # dali pipeline outputs nhwc images
                images.tensors = nhwc_to_nchw_transform(images.tensors)
        if self.batch_size_one:
            flat_res = self.graphable(images.tensors, images.image_sizes_tensor, targets[0], targets[1], targets[2])
            features, detections_bbox, detections_matched_idxs, detections_regression_targets, detections_labels, loss_classifier, loss_box_reg, loss_objectness, loss_rpn_box_reg = flat_res[0:5], flat_res[5], flat_res[6], flat_res[7], flat_res[8], flat_res[9], flat_res[10], flat_res[11], flat_res[12]
            detections = BoxList(detections_bbox, image_size = images.image_sizes_wh[0])
            detections.add_field("matched_idxs", detections_matched_idxs)
            detections.add_field("regression_targets", detections_regression_targets)
            detections.add_field("labels", detections_labels)
            detections = [detections]
            _, _, loss_mask = self.roi_mask_only.mask(features, detections, targets, syncfree=True)
            loss_box = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
            losses = {}
            losses.update(loss_box)
            losses.update(loss_mask)
            proposal_losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                }
            losses.update(proposal_losses)
            return losses
        else:
            flat_res = self.graphable(images.tensors, images.image_sizes_tensor)
            features, objectness, rpn_box_regression, anchor_boxes, anchor_visibility = flat_res[0:5], list(flat_res[5:10]), list(flat_res[10:15]), flat_res[15], flat_res[16]
            return self.combined_rpn_roi(images, anchor_boxes, anchor_visibility, objectness, rpn_box_regression, targets, features)

