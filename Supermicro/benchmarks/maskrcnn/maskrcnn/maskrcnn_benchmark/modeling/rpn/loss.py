# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file contains specific functions for computing losses on the RPN
file
"""
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from .utils import concat_box_prediction_layers

from ..balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from ..utils import cat

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou, boxlist_iou_batched
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist


class RPNLossComputation(object):
    """
    This class computes the RPN loss.
    """

    def __init__(self, proposal_matcher, fg_bg_sampler, box_coder,
                 generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        # self.target_preparator = target_preparator
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']
        self.syncfree = True

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds

        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def match_targets_to_anchors_batched(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou_batched(target, anchor)
        matched_idxs = self.proposal_matcher(match_quality_matrix, batched=1)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        # target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        return matched_idxs

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image.masked_fill_(bg_indices, 0)

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image.masked_fill_(~anchors_per_image.get_field("visibility"), -1)

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image.masked_fill_(inds_to_discard, -1)

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def prepare_targets_batched(self, anchors, targets, anchors_visibility):

        matched_idxs = self.match_targets_to_anchors_batched(anchors, targets)
        labels = generate_rpn_labels2(matched_idxs)        
        labels = labels.to(dtype=torch.float32)

        bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
        labels.masked_fill_(bg_indices, 0)

        # discard anchors that go out of the boundaries of the image
        if "not_visibility" in self.discard_cases:
            labels.masked_fill_(anchors_visibility==0, -1)

        # discard indices that are between thresholds
        if "between_thresholds" in self.discard_cases:
            inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels.masked_fill_(inds_to_discard, -1)

        img_idx = torch.arange(anchors.size(0), device = anchors.device)[:, None]
        matched_targets = targets[img_idx, matched_idxs.clamp(min=0)]

        # compute regression targets
        regression_targets = self.box_coder.encode(
            matched_targets.view(-1,4), anchors.view(-1,4)
        )
        return labels.view(-1), regression_targets

    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor
        """
        anchors_cat = anchors[0]
        N = anchors_cat.size(0)
        anchors_visibility = anchors[1]
        device = anchors_cat.device
        targets_cat, _, _, _ = targets
        labels, regression_targets = self.prepare_targets_batched(anchors_cat, targets_cat, anchors_visibility)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels.view(N,-1), is_rpn=1)

        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)

        objectness = objectness.squeeze()

        if self.syncfree and N == 1:
            import maskrcnn_benchmark.Syncfree
            sampled_inds, counts = maskrcnn_benchmark.Syncfree.balanced_pos_neg_sampler_repeat(
                    sampled_pos_inds[0], torch.empty([0], device=device, dtype=torch.int64),
                    sampled_neg_inds[0], torch.empty([0], device=device, dtype=torch.int64),
                    self.fg_bg_sampler.batch_size_per_image,
                    self.fg_bg_sampler.batch_size_per_image,
                    True)
            sampled_pos_inds_mask = (torch.arange(0,self.fg_bg_sampler.batch_size_per_image,1, device=device) < counts[0]).unsqueeze(1)

            box_loss = smooth_l1_loss(
                box_regression.index_select(0, sampled_inds) * sampled_pos_inds_mask,
                regression_targets.index_select(0, sampled_inds) * sampled_pos_inds_mask,
                beta=1.0 / 9,
                size_average=False,
            ) / counts[1]
        else:
            if N == 1:
                # sampled pos inds only has 1 element if num_images is 1, so avoid torch.cat
                with torch.cuda.nvtx.range("NZ1"):
                    sampled_pos_inds = torch.nonzero(sampled_pos_inds[0]).squeeze(1)
                    sampled_neg_inds = torch.nonzero(sampled_neg_inds[0]).squeeze(1)
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

            box_loss = smooth_l1_loss(
                box_regression.index_select(0, sampled_pos_inds),
                regression_targets.index_select(0, sampled_pos_inds),
                beta=1.0 / 9,
                size_average=False,
            ) / (sampled_inds.numel())

        objectness_loss = F.binary_cross_entropy_with_logits(
            objectness.index_select(0, sampled_inds), labels.index_select(0, sampled_inds)
        )
        return objectness_loss, box_loss

# This function should be overwritten in RetinaNet
def generate_rpn_labels(matched_targets):
    matched_idxs = matched_targets.get_field("matched_idxs")
    labels_per_image = matched_idxs >= 0
    return labels_per_image

def generate_rpn_labels2(matched_idxs):
   # matched_idxs = matched_targets.get_field("matched_idxs")
    labels = matched_idxs >= 0
    return labels

def make_rpn_loss_evaluator(cfg, box_coder):
    matcher = Matcher(
        cfg.MODEL.RPN.FG_IOU_THRESHOLD,
        cfg.MODEL.RPN.BG_IOU_THRESHOLD,
        allow_low_quality_matches=True,
    )

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
    )

    loss_evaluator = RPNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        generate_rpn_labels
    )
    return loss_evaluator
