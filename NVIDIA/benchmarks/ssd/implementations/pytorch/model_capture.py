# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import utils
from engine import preprocessing, init_scratchpad, loss_preprocessing, compute_loss, compute_matched_idxs
import copy


def whole_model_capture(model, optimizer, scaler, dataset, lr_scheduler, args):
    print('CUDA graph capture')

    # Back up initial model parameters and optimizer state.
    # Using model.state_dict seems to cause some inconsistency in the model
    # where the forward pass outputs differ between different runs,
    # so we now use a WAR where the model parameters are individually restored in-place.
    # model_bak = copy.deepcopy(model.state_dict())
    model_bak = [p.data.clone().detach() for p in model.parameters()]
    optimizer_bak = copy.deepcopy(optimizer.state_dict())
    if args.master_weights:
        master_weights_bak = copy.deepcopy(optimizer.param_groups_master)

    model.train()

    # direct pointer to the model
    model_ptr = model.module if args.distributed else model

    # extracting the device name from some layer
    device = model_ptr.backbone.body.conv1.weight.device

    if args.cuda_graphs_syn:
        assert (dataset is None)

        images, targets = [], {'boxes': [], 'labels': []}
        for b in range(args.batch_size):
            # These are just arbitrary sizes for model capture
            images.append(torch.randint(low=0, high=256, size=[3, 1000, 1000], device=device).float() / 255)
            targets['boxes'].append(torch.tensor([[10, 20, 30, 40]], device=device))
            targets['labels'].append(torch.tensor([1], device=device))
        images, targets = preprocessing(images, targets, model_ptr, args.data_layout)
    else:
        images, targets = [], []

        # taking the first batch
        for images_, targets_ in dataset:
            images = images_
            targets = targets_
            break

        # if not DALI, then we should preprocess the data
        if not args.dali:
            images = list(image.to(device, non_blocking=True) for image in images)
            targets = {k: [dic[k].to(device, non_blocking=True) for dic in targets] for k in targets[0]}

            # --- preprocessing
            images, targets = preprocessing(images, targets, model_ptr, args.data_layout)

    # DALI can compute matched_idxs and put it in targets, but if it doesn't do so, do it here
    if 'matched_idxs' not in targets:
        with torch.cuda.amp.autocast(enabled=args.amp):
            targets['matched_idxs'] = compute_matched_idxs(targets['boxes'], model_ptr)

    with torch.cuda.amp.autocast(enabled=args.amp):
        init_scratchpad(images, targets, args.batch_size, args.num_classes, args.amp,
                        args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad,
                        args.cuda_graphs)

        if args.not_graphed_prologues:
            gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask = \
                loss_preprocessing(utils.ScratchPad.target_boxes_padded if args.reg_head_pad else targets['boxes'],
                                   utils.ScratchPad.target_labels_padded if args.cls_head_pad else targets['labels'],
                                   utils.ScratchPad.target_matched_idxs, model_ptr,
                                   args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad)

    static_matched_idxs = torch.zeros_like(targets['matched_idxs'])
    static_matched_idxs.copy_(targets['matched_idxs'])
    print('CUDA graphs: data preprocessing complete')

    # --- warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for j in range(11):
            if args.apex_adam:
                # set_to_none is True by default
                optimizer.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)

            lr_scheduler.step()

            with torch.cuda.amp.autocast(enabled=args.amp):
                if not args.not_graphed_prologues:
                    # preprocess everything that does not require model forward and backward
                    gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask = \
                        loss_preprocessing(utils.ScratchPad.target_boxes_padded if args.reg_head_pad else targets['boxes'],
                                           utils.ScratchPad.target_labels_padded if args.cls_head_pad else targets['labels'],
                                           utils.ScratchPad.target_matched_idxs, model_ptr,
                                           args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad)

                # forward
                model_output = model(images)
                # features = model_output[0:5]
                # head_outputs = {'cls_logits': model_output[5], 'bbox_regression': model_output[6]}

                cls_loss, reg_loss = compute_loss(model_ptr, model_output[5], model_output[6], valid_idxs,
                                                  gt_classes_target, num_foreground, target_regression,
                                                  foreground_idxs_mask, args.apex_focal_loss, args.reg_head_pad)

                losses = cls_loss + reg_loss
                #assert(not torch.isnan(losses)) # Seems to cause hangs

            # backward
            scaler.scale(losses).backward()

            # optimizer
            scaler.step(optimizer)
            scaler.update()
    torch.cuda.current_stream().wait_stream(s)
    print('CUDA graphs: warmup iterations complete')

    # --- capture
    g = torch.cuda.CUDAGraph()

    if args.apex_adam:
        # set_to_none is True by default
        optimizer.zero_grad()
    else:
        optimizer.zero_grad(set_to_none=True)

    with torch.cuda.graph(g):
        # LR was already copied during warmup
        if args.warmup_epochs > 0:
            lr_scheduler.step()

        with torch.cuda.amp.autocast(enabled=args.amp):
            if not args.not_graphed_prologues:
                # loss_preprocessing is now part of the graph
                gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask = \
                    loss_preprocessing(utils.ScratchPad.target_boxes_padded if args.reg_head_pad else targets['boxes'],
                                       utils.ScratchPad.target_labels_padded if args.cls_head_pad else targets['labels'],
                                       utils.ScratchPad.target_matched_idxs, model_ptr,
                                       args.apex_focal_loss, args.max_boxes, args.cls_head_pad, args.reg_head_pad)

            # forward
            static_model_output = model(images)

            # loss
            static_cls_loss, static_reg_loss = compute_loss(model_ptr, static_model_output[5], static_model_output[6],
                                                            valid_idxs, gt_classes_target, num_foreground,
                                                            target_regression, foreground_idxs_mask,
                                                            args.apex_focal_loss, args.reg_head_pad)

            static_loss = static_cls_loss + static_reg_loss

        # backward
        scaler.scale(static_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print('CUDA graphs: capture complete')

    scaler.step(optimizer)
    # Restore gradient scaler, model and optimizer back to their initial states
    scaler.update(65536.0)
    # Model parameters are individually restored instead of using state_dict
    # model.load_state_dict(model_bak)
    with torch.no_grad():
        for pi, p in enumerate(model.parameters()):
            p.data.copy_(model_bak[pi])
    optimizer.load_state_dict(optimizer_bak)
    if args.master_weights:
        for gi, pg in enumerate(optimizer.param_groups_master):
            param_list = pg['params']
            param_list_bak = master_weights_bak[gi]['params']
            for pi, p in enumerate(param_list):
                p.data.copy_(param_list_bak[pi].data)

    if args.not_graphed_prologues:
        static_prologues_out = [gt_classes_target, target_regression, num_foreground, valid_idxs, foreground_idxs_mask]
    else:
        static_prologues_out = None
    print('CUDA graph capture for training complete')

    return g, images, static_loss, static_prologues_out


def whole_model_capture_eval(model, dataset, args):
    # save original params for later
    model_bak = copy.deepcopy(model.state_dict())

    # direct pointer to the model
    model_ptr = model.module if args.distributed else model

    # extracting the device name from some layer
    device = model_ptr.backbone.body.conv1.weight.device

    # Convert epochs to iterations
    # we want to control warmup at the epoch level, but update lr every iteration

    if args.cuda_graphs_syn:
        assert (dataset is None)

        images, targets = [], {'boxes': [], 'labels': []}
        for b in range(args.eval_batch_size):
            # These are just arbitrary sizes for model capture
            images.append(torch.rand([3, 1000, 1000], device=device))
            targets['boxes'].append(torch.tensor([[10, 20, 30, 40]], device=device))
            targets['labels'].append(torch.tensor([1], device=device))
        images, targets = preprocessing(images, targets, model_ptr, args.data_layout)
    else:
        images, targets = [], []

        # taking the first batch
        for images_, targets_ in dataset:
            images = images_
            targets = targets_
            break

        # if not DALI, then we should preprocess the data
        if not args.dali:
            images = list(image.to(device, non_blocking=True) for image in images)
            targets = {k: [dic[k].to(device, non_blocking=True) for dic in targets] for k in targets[0]}

        # --- preprocessing
        images, targets = preprocessing(images, targets, model_ptr, args.data_layout)

    # --- warmup
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for j in range(11):
            with torch.cuda.amp.autocast(enabled=args.amp):
                # forward
                model_output = model(images)
                # features = model_output[0:5]
                # head_outputs = {'cls_logits': model_output[5], 'bbox_regression': model_output[6]}
    torch.cuda.current_stream().wait_stream(s)

    # --- capture
    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        with torch.cuda.amp.autocast(enabled=args.amp):
            # forward
            static_model_output = model(images)

    return g, images, static_model_output


def model_eval_warmup(model, batch_size, iters, args):
    model.eval()

    # direct pointer to the model
    model_ptr = model.module if args.distributed else model
    # extracting the device name from some layer
    device = model_ptr.backbone.body.conv1.weight.device

    for i in range(iters):
        with torch.cuda.amp.autocast(enabled=args.amp):
            x = torch.rand([batch_size, 3, args.image_size[0], args.image_size[1]], device=device)
            model(x)
