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

import torch
from torch.optim.optimizer import Optimizer, required
from apex.multi_tensor_apply import multi_tensor_applier
from maskrcnn_benchmark.utils.comm import get_rank

class FusedSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Currently GPU-only.  Requires Apex to be installed via
    ``pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./``.

    This version of fused SGD implements 2 fusions.

      * Fusion of the SGD update's elementwise operations
      * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.

    :class:`apex.optimizers.FusedSGD` may be used as a drop-in replacement for ``torch.optim.SGD``::

        opt = apex.optimizers.FusedSGD(model.parameters(), lr = ....)
        ...
        opt.step()

    :class:`apex.optimizers.FusedSGD` may be used with or without Amp.  If you wish to use :class:`FusedSGD` with Amp,
    you may choose any ``opt_level``::

        opt = apex.optimizers.FusedSGD(model.parameters(), lr = ....)
        model, opt = amp.initialize(model, opt, opt_level="O0" or "O1 or "O2")
        ...
        opt.step()

    In general, ``opt_level="O1"`` is recommended.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """
    def __init__(self, model, cfg, gradient_scaler, distributed, training_comm):
        assert(cfg.DTYPE == "float16"), "float32 not suported by mlperf maskrcnn"
        # TODO: Don't think this is necessary, optimizer should work for float32 as well.

        params, bias_params = [], []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            if "bias" in key:
                bias_params.append(value)
            else:
                params.append(value)
        params = [
                {"params": params, "lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY},
                {"params": bias_params, "lr": cfg.SOLVER.BASE_LR*cfg.SOLVER.BIAS_LR_FACTOR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY*cfg.SOLVER.WEIGHT_DECAY_BIAS}
                ]

        defaults = dict(lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, dampening=0,
                        weight_decay=0, nesterov=False)
        super(FusedSGD, self).__init__(params, defaults)
        base_lrs = [group['lr'] for group in self.param_groups]
        self.base_lrs = torch.cuda.FloatTensor(base_lrs)

        warmup_methods = {"constant": 1, "linear": 2, "mlperf_linear": 3}
        self.warmup_method = cfg.SOLVER.WARMUP_METHOD
        if self.warmup_method not in warmup_methods:
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(self.warmup_method)
            )
        self.warmup_method_index = warmup_methods[self.warmup_method]
        self.milestones = cfg.SOLVER.STEPS
        self.gamma = cfg.SOLVER.GAMMA
        self.warmup_factor = cfg.SOLVER.WARMUP_FACTOR
        self.warmup_iters = cfg.SOLVER.WARMUP_ITERS
        self.dynamic_loss_scale = True
        self.scale_window = cfg.DYNAMIC_LOSS_SCALE_WINDOW

        if multi_tensor_applier.available:
            import amp_C, apex_C
            import maskrcnn_benchmark.Syncfree
            # Skip buffer
            self.overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_scale = amp_C.multi_tensor_scale
            self.multi_tensor_l2norm = amp_C.multi_tensor_l2norm
            self.flatten = apex_C.flatten
            self.unflatten = apex_C.unflatten
            self.step_optimizer_state = maskrcnn_benchmark.Syncfree.step_optimizer_state_cuda
            self.multi_tensor_sgd = maskrcnn_benchmark.Syncfree.multi_tensor_sgd_cuda
        else:
            raise RuntimeError('FusedSGD requires cuda extensions')

        optimizer_state = [0,0,65536] # cur_iter, last_overflow_iter, cur_scale
        self.fp32_params, self.fp32_m = [], []
        self.fp16_params, self.fp32_from_fp16_params, self.fp32_from_fp16_m = [], [], []
        group_index_fp16, group_index_fp32 = [], []
        for i, group in enumerate(self.param_groups):
            optimizer_state = optimizer_state + [
                    group['weight_decay'],
                    group['momentum'],
                    group['dampening'],
                    1.0 if group['nesterov'] else 0.0,
                    0.0 # lr
                    ]
            for p in group['params']:
                if p.dtype == torch.float16:
                    self.fp16_params.append(p)
                    self.fp32_from_fp16_params.append(p.clone().float().detach())
                    self.fp32_from_fp16_m.append(p.clone().float().detach())
                    group_index_fp16.append(i)
                elif p.dtype == torch.float32:
                    self.fp32_params.append(p)
                    self.fp32_m.append(p.clone().detach())
                    group_index_fp32.append(i)
        self.optimizer_state = torch.cuda.FloatTensor(optimizer_state)
        for m in self.fp32_m + self.fp32_from_fp16_m:
            m.zero_()
        self.has_fp16_params = True if len(self.fp16_params) > 0 else False
        self.has_fp32_params = True if len(self.fp32_params) > 0 else False
        if self.has_fp16_params:
            self.flat_fp16_grads = self.flatten(self.fp16_params)
            self.fp16_grads = self.unflatten(self.flat_fp16_grads, self.fp16_params)
            self.fp16_group_index = torch.cuda.IntTensor(group_index_fp16)
        if self.has_fp32_params:
            self.flat_fp32_grads = self.flatten(self.fp32_params)
            self.fp32_grads = self.unflatten(self.flat_fp32_grads, self.fp32_params)
            self.fp32_group_index = torch.cuda.IntTensor(group_index_fp32)
        self.gradient_scaler = gradient_scaler # should be 1.0 / (num_training_ranks * spatial_group_size) for most cases
        self.distributed = distributed
        self.training_comm = training_comm
        if get_rank() == 0:
            print("%d :: self.fp16_group_index=%s" % (get_rank(), str(self.fp16_group_index)))

        self.wd_after_momentum = False

    def __setstate__(self, state):
        super(FusedSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def update_master_params(self):
        if self.has_fp16_params:
            self.overflow_buf.zero_()
            multi_tensor_applier(
                self.multi_tensor_scale,
                self.overflow_buf,
                [self.fp16_params, self.fp32_from_fp16_params],
                1.0
                )
            for m in self.fp32_from_fp16_m:
                m.zero_()
        if self.has_fp32_params:
            for m in self.fp32_m:
                m.zero_()

    def backward(self, loss):
        scaled_loss = (loss.float()) * self.optimizer_state[2]
        scaled_loss.backward()

    def copy_gradients(self):
        self.overflow_buf.zero_()
        if self.has_fp16_params:
            multi_tensor_applier(
                self.multi_tensor_scale,
                self.overflow_buf,
                [[p.grad for p in self.fp16_params], self.fp16_grads],
                self.gradient_scaler
                )
        if self.has_fp32_params:
            multi_tensor_applier(
                self.multi_tensor_scale,
                self.overflow_buf,
                [[p.grad for p in self.fp32_params], self.fp32_grads],
                self.gradient_scaler
                )

    def zero_grad(self, set_grads_to_None=True):
        if set_grads_to_None:
            if self.has_fp16_params:
                for p in self.fp16_params:
                    p.grad = None
            if self.has_fp32_params:
                for p in self.fp32_params:
                    p.grad = None
        else:
            if self.has_fp16_params:
                for p in self.fp16_params:
                    p.grad.zero_()
            if self.has_fp32_params:
                for p in self.fp32_params:
                    p.grad.zero_()

    def step(self):
        if self.distributed:
            if self.has_fp16_params:
                torch.distributed.all_reduce(self.flat_fp16_grads, group=self.training_comm)
                norm, norm_per_tensor = multi_tensor_applier(
                        self.multi_tensor_l2norm,
                        self.overflow_buf,
                        [self.fp16_grads],
                        True)
            if self.has_fp32_params:
                torch.distributed.all_reduce(self.flat_fp32_grads, group=self.training_comm)
                norm, norm_per_tensor = multi_tensor_applier(
                        self.multi_tensor_l2norm,
                        self.overflow_buf,
                        [self.fp32_grads],
                        True)
        if self.has_fp16_params:
            multi_tensor_applier(
                self.multi_tensor_sgd,
                self.overflow_buf,
                [self.fp16_grads, self.fp32_from_fp16_params, self.fp32_from_fp16_m, self.fp16_params],
                self.fp16_group_index,
                self.optimizer_state,
                self.wd_after_momentum)
        if self.has_fp32_params:
            multi_tensor_applier(
                self.multi_tensor_sgd,
                self.overflow_buf,
                [self.fp32_grads, self.fp32_params, self.fp32_m],
                self.fp32_group_index,
                self.optimizer_state,
                self.wd_after_momentum)
        self.step_optimizer_state(
                self.overflow_buf,
                self.optimizer_state,
                self.base_lrs,
                self.warmup_factor,
                self.warmup_method_index,
                self.warmup_iters,
                self.gamma,
                self.milestones[0],
                self.milestones[1],
                1 if self.dynamic_loss_scale else 0,
                2.0,
                self.scale_window)

