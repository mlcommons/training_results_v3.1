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

""" example train fit utility """
import logging
import os
import time
import re
import math
import mxnet as mx
import horovod.mxnet as hvd
import numpy as np

#### imports needed for fit monkeypatch
from mxnet.initializer import Uniform
from mxnet.context import cpu
from mxnet.monitor import Monitor
from mxnet.model import BatchEndParam
from mxnet.initializer import Uniform
from mxnet.io import DataDesc, DataIter, DataBatch
from mxnet.base import _as_list
from mxnet import cuda_utils as cu
import copy
##### imports needed for custom optimizer
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import (NDArray, zeros, clip, sqrt, cast, maximum, abs as NDabs, array, multiply,
                           multi_sum_sq, multi_lars)
from mxnet.ndarray import (sgd_update, sgd_mom_update, adam_update, rmsprop_update, rmspropalex_update,
                           mp_sgd_update, mp_sgd_mom_update, square, ftrl_update, ftml_update,
                           signsgd_update, signum_update,
                           multi_sgd_update, multi_sgd_mom_update, multi_mp_sgd_update,
                           multi_mp_sgd_mom_update,
                           lars_multi_sgd_update, lars_multi_sgd_mom_update,
                           lars_multi_mp_sgd_update, lars_multi_mp_sgd_mom_update)
from mxnet.ndarray import sparse
#####

from mlperf_log_utils import mllogger, mpiwrapper
from mxnet import cuda_utils as cu
from common.optimizer import SGDwFASTLARSV2
import cuda_graphs.graph_wrapper as graph_wrapper

from common.data import SyntheticDataIter

# from scaleoutbridge import init_bridge, ScaleoutBridge as SBridge
from mlperf_common.scaleoutbridge import init_bridge, ScaleoutBridgeBase as SBridge
from mlperf_common.frameworks.mxnet import MXNetProfilerHandler, MPICommunicationHandler

TRAIN_CUDA_GRAPH_ID = 0

def _flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

@register
class SGDwLARS(Optimizer):
    """The SGD optimizer with momentum and weight decay.

    If the storage types of grad is ``row_sparse`` and ``lazy_update`` is True, \
    **lazy updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = lr * (rescale_grad * clip(grad[row], clip_gradient) + wd * weight[row])
            state[row] = momentum[row] * state[row] + rescaled_grad[row]
            weight[row] = weight[row] - state[row]

    The sparse update only updates the momentum for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    Otherwise, **standard updates** are applied by::

        rescaled_grad = lr * (rescale_grad * clip(grad, clip_gradient) + wd * weight)
        state = momentum * state + rescaled_grad
        weight = weight - state

    For details of the update algorithm see
    :class:`~mxnet.ndarray.sgd_update` and :class:`~mxnet.ndarray.sgd_mom_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
        The momentum value.
    lazy_update : bool, optional
        Default is True. If True, lazy updates are applied \
        if the storage types of weight and grad are both ``row_sparse``.
    multi_precision: bool, optional
        Flag to control the internal precision of the optimizer.::

            False: results in using the same precision as the weights (default),
            True: makes internal 32-bit copy of the weights and applies gradients
            in 32-bit precision even if actual weights used in the model have lower precision.
            Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, momentum=0.0, lazy_update=True, lars=True, lars_eta=0.001, lars_eps=0, **kwargs):
        super(SGDwLARS, self).__init__(**kwargs)
        self.momentum = momentum
        self.lazy_update = lazy_update
        self.aggregate_num = int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE', "4"))
        self.lars = lars
        self.lars_eta = lars_eta
        self.lars_eps = lars_eps
        self.skip = 0
        self.last_lr = None
        self.cur_lr = None


    def _get_lrs(self, indices):
        """Gets the learning rates given the indices of the weights.

        Parameters
        ----------
        indices : list of int
            Indices corresponding to weights.

        Returns
        -------
        lrs : list of float
            Learning rates for those indices.
        """
        if self.cur_lr is not None:
            self.last_lr = self.cur_lr

        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.num_update)
        else:
            lr = self.lr

        if self.cur_lr is None:
            self.last_lr = lr
        self.cur_lr = lr

        lrs = [lr for _ in indices]
        for i, index in enumerate(indices):
            if index in self.param_dict:
                lrs[i] *= self.param_dict[index].lr_mult
            elif index in self.lr_mult:
                lrs[i] *= self.lr_mult[index]
            elif index in self.idx2name:
               lrs[i] *= self.lr_mult.get(self.idx2name[index], 1.0)
        return lrs

    def set_wd_mult(self, args_wd_mult):
        self.wd_mult = {}
        for n in self.idx2name.values():
            is_weight = n.endswith('_weight')
            is_fc_bias = 'fc' in n and 'bias' in n
            if not (is_weight or is_fc_bias):
                self.wd_mult[n] = 0.0

        if self.sym_info:
            attr, arg_names = self.sym_info
            for name in arg_names:
                if name in attr and '__wd_mult__' in attr[name]:
                    self.wd_mult[name] = float(attr[name]['__wd_mult__'])
        self.wd_mult.update(args_wd_mult)

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == np.float16:
            weight_master_copy = weight.astype(np.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == np.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "SGD optimizer")
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            stype = weight.stype if self.lazy_update else 'default'
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    def _l2norm(self, v, rescale=False):
        "L2 Norm implementation"
        v = v.astype('float32')
        if rescale:
            v *= self.rescale_grad
        norm = mx.nd.norm(v).asnumpy()[0]
        return norm

    def _get_lars(self, i, weight, g, lr, wd):
        "Returns a scaling factor for the learning rate for this layer"
        name = self.idx2name[i] if i in self.idx2name else str(i)
        if name.endswith('gamma') or name.endswith('beta') or name.endswith('bias'):
            return lr

        w_norm = self._l2norm(weight)
        g_norm = self._l2norm(g, rescale=True)

        if w_norm > 0.0 and g_norm > 0.0:
            lars = self.lars_eta * w_norm/(g_norm + wd * w_norm + self.lars_eps)
        else:
            lars = 1.0

        return lars * lr

    def _update_impl(self, indices, weights, grads, states, multi_precision=False):
        aggregate = True
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
            weights = [weights]
            grads = [grads]
            states = [states]
        for weight, grad in zip(weights, grads):
            assert(isinstance(weight, NDArray))
            assert(isinstance(grad, NDArray))
            aggregate = (aggregate and
                         weight.stype == 'default' and
                         grad.stype == 'default')
        self._update_count(indices)
        lrs = self._get_lrs(indices)

        wds = self._get_wds(indices)

        if self.lars:
            lrs = [self._get_lars(i, w, g, lr, wd) for (i, w, g, lr, wd) in zip(indices, weights, grads, lrs, wds)]

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum * (self.cur_lr / self.last_lr)

        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if aggregate:
            current_index = 0
            while current_index < len(indices):
                sidx = current_index
                eidx = current_index + self.aggregate_num
                if not multi_precision:
                    if self.momentum > 0:
                        multi_sgd_mom_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                grads[sidx:eidx],
                                                                states[sidx:eidx])),
                                             out=weights[sidx:eidx],
                                             num_weights=len(weights[sidx:eidx]),
                                             lrs=lrs[sidx:eidx],
                                             wds=wds[sidx:eidx],
                                             **kwargs)
                    else:
                        multi_sgd_update(*_flatten_list(zip(weights[sidx:eidx],
                                                            grads[sidx:eidx])),
                                         out=weights[sidx:eidx],
                                         num_weights=len(weights[sidx:eidx]),
                                         lrs=lrs[sidx:eidx],
                                         wds=wds[sidx:eidx],
                                         **kwargs)
                else:
                    if self.momentum > 0:
                        multi_mp_sgd_mom_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                   grads[sidx:eidx],
                                                                   *zip(*states[sidx:eidx]))),
                                                out=weights[sidx:eidx],
                                                num_weights=len(weights[sidx:eidx]),
                                                lrs=lrs[sidx:eidx],
                                                wds=wds[sidx:eidx],
                                                **kwargs)
                    else:
                        multi_mp_sgd_update(*_flatten_list(zip(weights[sidx:eidx],
                                                               grads[sidx:eidx],
                                                               list(zip(*states[sidx:eidx]))[1])),
                                            out=weights[sidx:eidx],
                                            num_weights=len(weights[sidx:eidx]),
                                            lrs=lrs[sidx:eidx],
                                            wds=wds[sidx:eidx],
                                            **kwargs)
                current_index += self.aggregate_num
        else:
            for weight, grad, state, lr, wd in zip(weights, grads, states, lrs, wds):
                if not multi_precision:
                    if state is not None:
                        sgd_mom_update(weight, grad, state, out=weight,
                                       lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)

                    else:
                        sgd_update(weight, grad, out=weight, lazy_update=self.lazy_update,
                                   lr=lr, wd=wd, **kwargs)
                else:
                    if state[0] is not None:
                        mp_sgd_mom_update(weight, grad, state[0], state[1], out=weight,
                                          lr=lr, wd=wd, **kwargs)
                    else:
                        mp_sgd_update(weight, grad, state[1], out=weight,
                                      lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        if not isinstance(index, (tuple, list)):
            use_multi_precision = self.multi_precision and weight.dtype == np.float16
        else:
            use_multi_precision = self.multi_precision and weight[0].dtype == np.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

@register
class SGDwFASTLARS(Optimizer):
    """The SGD optimizer with momentum and weight decay.

    If the storage types of grad is ``row_sparse`` and ``lazy_update`` is True, \
    **lazy updates** are applied by::

        for row in grad.indices:
            rescaled_grad[row] = lr * (rescale_grad * clip(grad[row], clip_gradient) + wd * weight[row])
            state[row] = momentum[row] * state[row] + rescaled_grad[row]
            weight[row] = weight[row] - state[row]

    The sparse update only updates the momentum for the weights whose row_sparse
    gradient indices appear in the current batch, rather than updating it for all
    indices. Compared with the original update, it can provide large
    improvements in model training throughput for some applications. However, it
    provides slightly different semantics than the original update, and
    may lead to different empirical results.

    Otherwise, **standard updates** are applied by::

        rescaled_grad = lr * (rescale_grad * clip(grad, clip_gradient) + wd * weight)
        state = momentum * state + rescaled_grad
        weight = weight - state

    For details of the update algorithm see
    :class:`~mxnet.ndarray.sgd_update` and :class:`~mxnet.ndarray.sgd_mom_update`.

    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.

    Parameters
    ----------
    momentum : float, optional
        The momentum value.
    lazy_update : bool, optional
        Default is True. If True, lazy updates are applied \
        if the storage types of weight and grad are both ``row_sparse``.
    multi_precision: bool, optional
        Flag to control the internal precision of the optimizer.::

            False: results in using the same precision as the weights (default),
            True: makes internal 32-bit copy of the weights and applies gradients
            in 32-bit precision even if actual weights used in the model have lower precision.
            Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self, momentum=0.0, lazy_update=True, lars=True, lars_eta=0.001, lars_eps=0, **kwargs):
        super(SGDwFASTLARS, self).__init__(**kwargs)
        self.momentum = momentum
        self.lazy_update = lazy_update
        self.aggregate_num = int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE', "4"))
        self.lars = lars
        self.lars_eta = lars_eta
        self.lars_eps = lars_eps
        self.skip = 0
        self.last_lr = None
        self.cur_lr = None
        self.use_lars_cached = False 
        self.use_sgd_cached = False 
        self.new_lrs = None
        self.new_wds = None
        self.sgd_wds = None
        self.w_sum_sq = None
        self.g_sum_sq = None

    def _get_lrs(self, indices):
        """Gets the learning rates given the indices of the weights.

        Parameters
        ----------
        indices : list of int
            Indices corresponding to weights.

        Returns
        -------
        lrs : list of float
            Learning rates for those indices.
        """
        if self.cur_lr is not None:
            self.last_lr = self.cur_lr

        if self.lr_scheduler is not None:
            lr = self.lr_scheduler(self.num_update)
        else:
            lr = self.lr

        if self.cur_lr is None:
            self.last_lr = lr
        self.cur_lr = lr

        lrs = [lr for _ in indices]
        for i, index in enumerate(indices):
            if index in self.param_dict:
                lrs[i] *= self.param_dict[index].lr_mult
            elif index in self.lr_mult:
                lrs[i] *= self.lr_mult[index]
            elif index in self.idx2name:
                lrs[i] *= self.lr_mult.get(self.idx2name[index], 1.0)
        return lrs

    def set_wd_mult(self, args_wd_mult):
        self.wd_mult = {}
        for n in self.idx2name.values():
            is_weight = n.endswith('_weight')
            is_fc_bias = 'fc' in n and 'bias' in n
            if not (is_weight or is_fc_bias):
                self.wd_mult[n] = 0.0

        if self.sym_info:
            attr, arg_names = self.sym_info
            for name in arg_names:
                if name in attr and '__wd_mult__' in attr[name]:
                    self.wd_mult[name] = float(attr[name]['__wd_mult__'])
        self.wd_mult.update(args_wd_mult)

    def create_state_multi_precision(self, index, weight):
        weight_master_copy = None
        if self.multi_precision and weight.dtype == np.float16:
            weight_master_copy = weight.astype(np.float32)
            return (self.create_state(index, weight_master_copy), weight_master_copy)
        if weight.dtype == np.float16 and not self.multi_precision:
            warnings.warn("Accumulating with float16 in optimizer can lead to "
                          "poor accuracy or slow convergence. "
                          "Consider using multi_precision=True option of the "
                          "SGD optimizer")
        return self.create_state(index, weight)

    def create_state(self, index, weight):
        momentum = None
        if self.momentum != 0.0:
            stype = weight.stype if self.lazy_update else 'default'
            momentum = zeros(weight.shape, weight.context, dtype=weight.dtype, stype=stype)
        return momentum

    def _l2norm(self, v, rescale=False):
        "L2 Norm implementation"
        v = v.astype('float32')
        if rescale:
            v *= self.rescale_grad
        norm = mx.nd.norm(v).asnumpy()[0]
        return norm

    def _get_lars(self, i, weight, g, lr, wd):
        "Returns a scaling factor for the learning rate for this layer"
        name = self.idx2name[i] if i in self.idx2name else str(i)
        if name.endswith('gamma') or name.endswith('beta') or name.endswith('bias'):
            return lr

        w_norm = self._l2norm(weight)
        g_norm = self._l2norm(g, rescale=True)

        if w_norm > 0.0 and g_norm > 0.0:
            lars = self.lars_eta * w_norm/(g_norm + wd * w_norm + self.lars_eps)
        else:
            lars = 1.0

        return lars * lr

    def _update_impl(self, indices, weights, grads, states, multi_precision=False):
        aggregate = True
        if not isinstance(indices, (tuple, list)):
            indices = [indices]
            weights = [weights]
            grads = [grads]
            states = [states]
        for weight, grad in zip(weights, grads):
            assert(isinstance(weight, NDArray))
            assert(isinstance(grad, NDArray))
            aggregate = (aggregate and
                         weight.stype == 'default' and
                         grad.stype == 'default')
        self._update_count(indices)
        lrs = self._get_lrs(indices)
        wds = self._get_wds(indices)

        kwargs = {'rescale_grad': self.rescale_grad}
        if self.momentum > 0:
            kwargs['momentum'] = self.momentum * (self.cur_lr / self.last_lr)

        if self.clip_gradient:
            kwargs['clip_gradient'] = self.clip_gradient

        if aggregate:
            nb_params = len(indices)
            names = [self.idx2name[i] if i in self.idx2name else str(i) for i in indices]
            lars_idx = [i for i in range(nb_params) if not(names[i].endswith('gamma')
                        or names[i].endswith('beta') or names[i].endswith('bias'))]
            if self.lars and len(lars_idx) > 0:
                nb_lars = len(lars_idx)
                no_lars_idx = [i for i in range(nb_params) if (names[i].endswith('gamma') or
                               names[i].endswith('beta') or names[i].endswith('bias'))]
                cur_ctx = weights[0].context
                full_idx = lars_idx + no_lars_idx
                if not self.use_lars_cached:
                    self.use_lars_cached = True
                    self.new_lrs = array([lrs[i] for i in full_idx], ctx=cur_ctx, dtype='float32')
                    self.new_wds = array([wds[i] for i in full_idx], ctx=cur_ctx, dtype='float32')
                    self.w_sum_sq = array([0.0 for i in lars_idx], ctx=cur_ctx, dtype='float32')
                    self.g_sum_sq = array([0.0 for i in lars_idx], ctx=cur_ctx, dtype='float32')
                else:
                    self.new_lrs[:] = np.array([lrs[i] for i in full_idx],dtype='float32')[:]
                    self.new_wds[:] = np.array([wds[i] for i in full_idx],dtype='float32')[:]
                new_weights = [weights[i] for i in full_idx]
                new_grads = [grads[i] for i in full_idx]
                multi_sum_sq(*new_weights[:nb_lars], num_arrays=nb_lars, out=self.w_sum_sq[:nb_lars])
                multi_sum_sq(*new_grads[:nb_lars], num_arrays=nb_lars, out=self.g_sum_sq[:nb_lars])
                multi_lars(self.new_lrs[:nb_lars], self.w_sum_sq, self.g_sum_sq, self.new_wds[:nb_lars],
                           eta=self.lars_eta, eps=self.lars_eps, rescale_grad=self.rescale_grad,
                           out=self.new_lrs[:nb_lars])
                new_states = [states[i] for i in full_idx]
                # Same than usual using preloaded sgd functions
                sidx = 0
                while sidx < len(indices):
                    eidx = sidx + len(new_weights[sidx:sidx+self.aggregate_num])
                    if not multi_precision:
                        if self.momentum > 0:
                            lars_multi_sgd_mom_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           new_states[sidx:eidx])),
                                        self.new_lrs[sidx:eidx],
                                        self.new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                        else:
                            lars_multi_sgd_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                            new_grads[sidx:eidx])),
                                        self.new_lrs[sidx:eidx],
                                        self.new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                    else:
                        if self.momentum > 0:
                            lars_multi_mp_sgd_mom_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           *zip(*new_states[sidx:eidx]))),
                                        self.new_lrs[sidx:eidx],
                                        self.new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                        else:
                            lars_multi_mp_sgd_update(
                                        *_flatten_list(zip(new_weights[sidx:eidx],
                                                           new_grads[sidx:eidx],
                                                           list(zip(*new_states[sidx:eidx]))[1])),
                                        self.new_lrs[sidx:eidx],
                                        self.new_wds[sidx:eidx],
                                        out=new_weights[sidx:eidx],
                                        num_weights=len(new_weights[sidx:eidx]),
                                        **kwargs)
                    sidx += self.aggregate_num
            else:
                current_index = 0
                while current_index < len(indices):
                    sidx = current_index
                    eidx = current_index + self.aggregate_num
                    if not multi_precision:
                        if self.momentum > 0:
                            multi_sgd_mom_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                    grads[sidx:eidx],
                                                                    states[sidx:eidx])),
                                                 out=weights[sidx:eidx],
                                                 num_weights=len(weights[sidx:eidx]),
                                                 lrs=lrs[sidx:eidx],
                                                 wds=wds[sidx:eidx],
                                                 **kwargs)
                        else:
                            multi_sgd_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                grads[sidx:eidx])),
                                             out=weights[sidx:eidx],
                                             num_weights=len(weights[sidx:eidx]),
                                             lrs=lrs[sidx:eidx],
                                             wds=wds[sidx:eidx],
                                             **kwargs)
                    else:
                        if self.momentum > 0:
                            multi_mp_sgd_mom_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                       grads[sidx:eidx],
                                                                       *zip(*states[sidx:eidx]))),
                                                    out=weights[sidx:eidx],
                                                    num_weights=len(weights[sidx:eidx]),
                                                    lrs=lrs[sidx:eidx],
                                                    wds=wds[sidx:eidx],
                                                    **kwargs)
                        else:
                            multi_mp_sgd_update(*_flatten_list(zip(weights[sidx:eidx],
                                                                   grads[sidx:eidx],
                                                                   list(zip(*states[sidx:eidx]))[1])),
                                                out=weights[sidx:eidx],
                                                num_weights=len(weights[sidx:eidx]),
                                                lrs=lrs[sidx:eidx],
                                                wds=wds[sidx:eidx],
                                                **kwargs)
                    current_index += self.aggregate_num
        else:
            if self.lars:
                lrs = [self._get_lars(i, w, g, lr, wd) for (i, w, g, lr, wd) in
                       zip(indices, weights, grads, lrs, wds)]

            for weight, grad, state, lr, wd in zip(weights, grads, states, lrs, wds):
                if not multi_precision:
                    if state is not None:
                        sgd_mom_update(weight, grad, state, out=weight,
                                       lazy_update=self.lazy_update, lr=lr, wd=wd, **kwargs)
                    else:
                        sgd_update(weight, grad, out=weight, lazy_update=self.lazy_update,
                                   lr=lr, wd=wd, **kwargs)
                else:
                    if state[0] is not None:
                        mp_sgd_mom_update(weight, grad, state[0], state[1], out=weight,
                                          lr=lr, wd=wd, **kwargs)
                    else:
                        mp_sgd_update(weight, grad, state[1], out=weight,
                                      lr=lr, wd=wd, **kwargs)

    def update(self, index, weight, grad, state):
        self._update_impl(index, weight, grad, state, multi_precision=False)

    def update_multi_precision(self, index, weight, grad, state):
        if not isinstance(index, (tuple, list)):
            use_multi_precision = self.multi_precision and weight.dtype == np.float16
        else:
            use_multi_precision = self.multi_precision and weight[0].dtype == np.float16
        self._update_impl(index, weight, grad, state,
                          multi_precision=use_multi_precision)

def get_num_workers(args, kv):
    if 'horovod' in args.kv_store:
        num_workers = hvd.size()
    else:
        num_workers = kv.num_workers if kv else 1
    return num_workers

def get_epoch_size(args, kv):
    num_workers = get_num_workers(args, kv)
    return math.ceil(int(args.num_examples / num_workers) / args.batch_size)

def _get_gpu(gpus):
    idx = hvd.local_rank()
    gpu = gpus.split(",")[idx]
    return gpu

def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = get_epoch_size(args, kv)
    begin_epoch = 0
    if 'pow' in args.lr_step_epochs:
        num_workers = get_num_workers(args, kv)
        epoch_size = math.ceil(int(args.num_examples/num_workers)/args.batch_size)
        warmup_steps = epoch_size * args.warmup_epochs
        total_steps = epoch_size * args.num_epochs
        return (args.lr, PolySchedule(args.lr, total_steps, warmup_steps))

    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    mllogger.event(key=mllogger.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS,
                          value=step_epochs)
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d',
                     lr, begin_epoch)

    steps = [epoch_size * (x - begin_epoch)
             for x in step_epochs if x - begin_epoch > 0]
    if steps:
        num_workers = get_num_workers(args, kv)
        epoch_size = math.ceil(int(args.num_examples/num_workers)/args.batch_size)
        mllogger.event(key=mllogger.constants.OPT_LR_DECAY_BOUNDARY_EPOCHS,
                        value=step_epochs)
        mllogger.event(key=mllogger.constants.OPT_LR_DECAY_BOUNDARY_STEPS,
                        value=[lr * (args.lr_factor ** i) for i in range(len(step_epochs))])
        return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor,
                                                         base_lr=args.lr, warmup_steps=epoch_size * args.warmup_epochs,
                                                         warmup_mode=args.warmup_strategy))
    else:
        return (lr, None)

class PolySchedule():
    def __init__(self, base_lr, iterations, warmup_iterations):
        self.base_lr = base_lr
        self.iterations = iterations
        self.warmup_iterations = warmup_iterations
        self.end_lr = 0.0001
        self.lr_decay_poly_power = 2
        mllogger.event(key='sgd_opt_learning_rate_decay_poly_power', value=self.lr_decay_poly_power)
        mllogger.event(key='sgd_opt_end_learning_rate', value=self.end_lr)
        mllogger.event(key=mllogger.constants.LARS_OPT_LR_DECAY_POLY_POWER, value=self.lr_decay_poly_power)
        mllogger.event(key=mllogger.constants.LARS_OPT_END_LR, value=self.end_lr)

    def __call__(self, iteration):
        if iteration <= self.warmup_iterations:
            return self.base_lr * (iteration / self.warmup_iterations)
        else:
            polyit = iteration - self.warmup_iterations
            polytotal = self.iterations - self.warmup_iterations

            return self.end_lr + ((self.base_lr - self.end_lr) * (1 - (polyit / polytotal))**self.lr_decay_poly_power)

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str, default='resnet-v1b-mainloop-fl',
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int, default=50,
                       help='number of layers in the neural network, \
                             required by some networks such as resnet')
    train.add_argument('--num-classes', type=int, default=1000)
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--num-epochs', type=int, default=37,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=11.0,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str, default="pow2",
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--initializer', type=str, default='default',
                       help='the initializer type')
    train.add_argument('--label-smoothing', type=float, default=0.1)
    train.add_argument('--optimizer', type=str, default='sgdwfastlars',
                       help='the optimizer type')
    train.add_argument('--lars-eps', type=float, default=0,
                       help='lars epsilon param')
    train.add_argument('--lars-eta', type=float, default=0.001,
                       help='lars trust_factor param')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=5.0e-05,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=400,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    train.add_argument('--save-period', type=int, default=1, help='params saving period')
    train.add_argument('--eval-period', type=int, default=4, help='evaluation every N epochs')
    train.add_argument('--eval-offset', type=int, default=2, help='first evaluation on epoch N')

    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--dtype', type=str, default='float16',
                       help='precision: float32 or float16')
    # additional parameters for large batch sgd
    train.add_argument('--warmup-epochs', type=int, default=2,
                       help='the epochs to ramp-up lr to scaled large-batch value')
    train.add_argument('--warmup-strategy', type=str, default='linear',
                       help='the ramping-up strategy for large batch sgd')
    train.add_argument('--logging-dir', type=str, default='logs')
    train.add_argument('--log', type=str, default='')
    train.add_argument('--bn-gamma-init0', action='store_true')
    train.add_argument('--epoch-size',type=int, default=0,
                       help='set number of batches in an epoch. useful for debugging')
    train.add_argument('--profile-worker-suffix', type=str, default='',
                       help='profile workers actions into this file. During distributed training\
                             filename saved will be rank1_ followed by this suffix')
    train.add_argument('--profile-server-suffix', type=str, default='',
                       help='profile server actions into a file with name like rank1_ followed by this suffix \
                             during distributed training')
    train.add_argument('--accuracy-threshold', default=0.759, type=float,
                       help='stop training after top1 reaches this value')
    parser.add_argument('--profile', type=int, default=0,
                        help='nvprof profiling enabled')
    parser.add_argument('--load-checkpoint-path', type=str, default=None)
    parser.add_argument('--save-checkpoint-path', type=str, default=None)
    return train


class CorrectCount(mx.metric.Accuracy):
    def __init__(self, axis=1, name='correct-count',
                 output_names=None, label_names=None):
        super(CorrectCount, self).__init__(
                name=name, axis=axis,
                output_names=output_names, label_names=label_names)
        self.axis = axis

    def get(self):
        return (self.name, self.sum_metric)

    def get_global(self):
        return (self.name, self.global_sum_metric)


class TotalCount(mx.metric.Accuracy):
    def __init__(self, axis=1, name='total-count',
                 output_names=None, label_names=None):
        super(TotalCount, self).__init__(
                name=name, axis=axis,
                output_names=output_names, label_names=label_names)
        self.axis = axis

    def get(self):
        return (self.name, self.num_inst)

    def get_global(self):
        return (self.name, self.global_num_inst)


class TopKCorrectCount(mx.metric.TopKAccuracy):
    def __init__(self, name='top-k-correct-count',
                 output_names=None, label_names=None):
        super(TopKCorrectCount, self).__init__(
                name=name, top_k=5,
                output_names=output_names, label_names=label_names)

    def get(self):
        return (self.name, self.sum_metric)

    def get_global(self):
        return (self.name, self.global_sum_metric)


class CrossEntropyCount(mx.metric.CrossEntropy):
    def __init__(self, name='cross-entropy',
                 output_names=None, label_names=None):
        super(CrossEntropyCount, self).__init__(
                name=name, output_names=output_names, label_names=label_names)

    def get(self):
        return (self.name, self.sum_metric)

    def get_global(self):
        return (self.name, self.global_sum_metric)


def save_checkpoint(model, path):
    for k, v in model.get_params()[0].items():
        np.save(os.path.join(path, f"{k}.npy"), v.asnumpy())
    for k, v in model.get_params()[1].items():
        np.save(os.path.join(path, f"{k}.npy"), v.asnumpy())


def mlperf_fit(self, args, train_data,
               #cuda-graph inputs
               dummy_data, dummy_label , output_arr,
               eval_data=None, eval_metric='acc',
               epoch_end_callback=None, batch_end_callback=None, kvstore='local',
               optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
               eval_end_callback=None,
               eval_batch_end_callback=None, initializer=Uniform(0.01),
               arg_params=None, aux_params=None, allow_missing=False,
               force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
               validation_metric=None, monitor=None, sparse_row_id_fn=None,
               eval_offset=0, eval_period=1,
               accuracy_threshold=1.0,
               multi_gpu_per_process=False,
               run_start_time=0):
    global TRAIN_CUDA_GRAPH_ID
    assert num_epoch is not None, 'please specify number of epochs'

    if monitor is not None:
        self.install_monitor(monitor)

    if validation_metric is None:
        validation_metric = eval_metric
    ###########################################################################
    # Adding Correct and Total Count metrics
    ###########################################################################
    if not isinstance(validation_metric, list):
        validation_metric = [validation_metric]

    validation_metric = mx.metric.create(validation_metric)

    if not isinstance(validation_metric, mx.metric.CompositeEvalMetric):
        vm = mx.metric.CompositeEvalMetric()
        vm.append(validation_metric)
        validation_metric = vm

    for m in [CorrectCount(), TotalCount()]:
        validation_metric.metrics.append(m)
    ###########################################################################

    if not isinstance(eval_metric, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metric)
    block_epoch_start = begin_epoch
    block_epoch_count = eval_offset + 1 - (begin_epoch % eval_period)
    if block_epoch_count < 0:
        block_epoch_count += eval_period

    mllogger.start(
        key=mllogger.constants.BLOCK_START,
        metadata={'first_epoch_num': block_epoch_start + 1, 'epoch_count': block_epoch_count})

    #sbridge = init_bridge(hvd.rank())
    sbridge = init_bridge(MXNetProfilerHandler(), MPICommunicationHandler(), mllogger)

    ################################################################################
    # training loop with dali overlap with fwd
    ################################################################################
    epoch = 0
    converged = False
    num_epoch = int(1e6) if args.sustained_training_time > 0 else num_epoch
    for epoch in range(begin_epoch, num_epoch):
        sbridge .start_epoch_prof()
        mllogger.start(key=mllogger.constants.EPOCH_START, metadata={'epoch_num': epoch + 1})
        tic = time.time()
        eval_metric.reset()
        nbatch = 0
        data_iter = iter(train_data)
        end_of_batch = False
        next_data_batch = next(data_iter)
        next_next_data_batch = None

        while not end_of_batch:
            sbridge.start_prof(SBridge.ITER_TIME)
            if nbatch % 2 == 0:
                data_batch = next_data_batch
            else:
                data_batch = next_next_data_batch

            if monitor is not None:
                monitor.tic()
            
            if not args.e2e_cuda_graphs:
                sbridge.start_prof(SBridge.FWD_TIME)
                self.forward(data_batch)
                sbridge.stop_prof(SBridge.FWD_TIME)
            else:
                if args.use_dali:
                    data_batch[0].data[0].copyto(dummy_data[0])
                    data_batch[0].label[0].copyto(dummy_label[0])
                else:
                    data_batch.data[0].copyto(dummy_data[0])
                    data_batch.label[0].copyto(dummy_label[0])
                graph_wrapper.graph_replay(0, hvd.local_rank(), [dummy_data[0], dummy_label[0]], output_arr)
            
            try:
                if nbatch % 2 == 0:
                    # pre fetch next batch
                    next_next_data_batch = next(data_iter)
                    self.prepare(next_next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
                else:
                    next_data_batch = next(data_iter)
                    self.prepare(next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
            except StopIteration:
                end_of_batch = True

            if not args.e2e_cuda_graphs:
                sbridge.start_prof(SBridge.BWD_TIME)
                self.backward()
                sbridge.stop_start_prof(SBridge.BWD_TIME, SBridge.OPT_TIME)
                self.update()
                sbridge.stop_prof(SBridge.OPT_TIME)
            if monitor is not None:
                monitor.toc_print()

            if batch_end_callback is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch,
                        eval_metric=eval_metric,
                        locals=locals())
                for callback in _as_list(batch_end_callback):
                    callback(batch_end_params)

            nbatch += 1
            sbridge.stop_prof(SBridge.ITER_TIME)

        if not args.use_dali:
            mx.ndarray.waitall() 

        mllogger.end(key=mllogger.constants.EPOCH_STOP, metadata={"epoch_num": epoch + 1})
        # one epoch of training is finished
        toc = time.time()

        if kvstore:
            if kvstore.rank == 0:
                self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))
        elif 'horovod' in args.kv_store:
            if hvd.rank() == 0:
                self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))
        else:
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

        # for DLFW CI/CD
        # args.num_examples overrides len(train_data)
        speed = args.num_examples / (toc-tic)
        mllogger.event(key='tracked_stats',
                              value={'throughput': speed},
                              metadata={'step': (epoch + 1)})

        # sync aux params across devices if there is more than one GPU per process
        if multi_gpu_per_process:
            arg_params, aux_params = self.get_params()
            self.set_params(arg_params, aux_params)

        if epoch_end_callback is not None:
            for callback in _as_list(epoch_end_callback):
                callback(epoch, self.symbol, arg_params, aux_params)

        #----------------------------------------
        # evaluation on validation set
        if eval_data and epoch % eval_period == eval_offset:
            mx.ndarray.waitall() 
            sbridge.start_eval_prof()
            mllogger.start(key=mllogger.constants.EVAL_START, metadata={'epoch_num': epoch + 1})
            reduce_batchnorm_stats(self)
            res = self.score(eval_data, validation_metric,
                         score_end_callback=eval_end_callback,
                         batch_end_callback=eval_batch_end_callback, epoch=epoch)
            if kvstore:
                if kvstore.rank == 0:
                    for name, val in res:
                        self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            elif 'horovod' in args.kv_store:
                if hvd.rank() == 0:
                    for name, val in res:
                        self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            else:
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)
            res = dict(res)

            acc = [res['correct-count'], res['total-count']]
            acc = mpiwrapper.allreduce(acc)
            acc = acc[0]/acc[1]
            mllogger.end(key=mllogger.constants.EVAL_STOP, metadata={'epoch_num': epoch + 1})
            sbridge.stop_eval_prof()

            mllogger.event(key=mllogger.constants.EVAL_ACCURACY, value=acc,
                            metadata={'epoch_num': epoch + 1})

            mllogger.end(key=mllogger.constants.BLOCK_STOP,
                    metadata={'first_epoch_num': block_epoch_start + 1})

            if acc >= accuracy_threshold and not converged:
                mllogger.log_run_stop(status='success', epoch=epoch)
                converged = True
                sbridge.stop_epoch_prof()
                if args.save_checkpoint_path is not None:
                    if hvd.rank() == 0:
                        os.makedirs(args.save_checkpoint_path, exist_ok=True)
                        save_checkpoint(self, args.save_checkpoint_path)

            if converged:
                if args.sustained_training_time < (time.time() - run_start_time) / 60:
                    return epoch
                else:
                    if hvd.rank() == 0:
                        print(f"Training for {args.sustained_training_time} min, "
                              f"{round((time.time() - run_start_time) / 60, 2)} elapsed.")

            if epoch < (num_epoch - 1):
                block_epoch_start = epoch + 1
                block_epoch_count = num_epoch - epoch - 1
                if block_epoch_count > eval_period:
                    block_epoch_count = eval_period
                mllogger.start(
                    key=mllogger.constants.BLOCK_START,
                    metadata={'first_epoch_num': block_epoch_start + 1,
                              'epoch_count': block_epoch_count})
        sbridge.stop_epoch_prof()

    if args.profile > 0:
        mllogger.log_run_stop(status='success', epoch=epoch)
        cu.cuda_profiler_stop()
    else:
        mllogger.log_run_stop(status='aborted', epoch=epoch)


    return num_epoch

def reduce_batchnorm_stats(module, in_place=True):
    '''
        In place all reduce of running_mean and running_var
        module._exec_group.aux_arrays = nested list
    '''
    if (in_place) :
        tensor = []
        for i in range(0,len(module._exec_group.aux_arrays)):
            tensor.extend(module._exec_group.aux_arrays[i])
        hvd.grouped_allreduce_(tensor, name="reduce_bn_stats") # in place reduction
    else :
        arg_params, aux_params = module.get_params()
        param_names = list(aux_params.keys())
        param_names.sort()
        reduced_stats = {}
        stat_list = []
        # Implementation1 : copies the params
        for k in param_names:
            stat_list.append(aux_params[k])
        hvd.grouped_allreduce_(stat_list, name="reduce_bn_stats") # in place reduction
        for i, k in enumerate(param_names):
            reduced_stats[k] = stat_list[i]
        module.set_params(
                arg_params, reduced_stats,
                allow_missing = False, force_init = True,
                allow_extra = False)

def fit(args, kv, model, initializer, data_loader, devs, arg_params, aux_params, **kwargs):
    """
    train a model
    args : argparse returns
    model : loaded model of the neural network
    initializer : weight initializer
    data_loader : function that returns the train and val data iterators
    devs : devices for training
    arg_params : model parameters
    aux_params : model parameters
    """
    if 'horovod' in args.kv_store:
        kv = None
        rank = hvd.rank()
    else:
        rank = kv.rank
    num_workers = get_num_workers(args, kv)
    if args.profile_server_suffix:
        mx.profiler.set_config(filename=args.profile_server_suffix, profile_all=True, profile_process='server')
        mx.profiler.set_state(state='run', profile_process='server')

    if args.profile_worker_suffix:
        if num_workers > 1:
            filename = 'rank' + str(rank) + '_' + args.profile_worker_suffix
        else:
            filename = args.profile_worker_suffix
        mx.profiler.set_config(filename=filename, profile_all=True, profile_process='worker')
        mx.profiler.set_state(state='run', profile_process='worker')

    # logging
    # head = '%(asctime)-15s Node[' + str(rank) + '] %(message)s'
    # logging.basicConfig(level=logging.DEBUG, format=head)
    # logging.info('start with arguments %s', args)

    epoch_size = get_epoch_size(args, kv)


    # save model
    epoch_end_callbacks = []

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    optimizer_params = {
        'learning_rate': lr,
        'wd': args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    if 'horovod' in args.kv_store:
        optimizer_params['rescale_grad'] = 1. / args.batch_size

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd', 'sgdwlars', 'sgdwfastlars','sgdwfastlarsv2'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom

    mllogger.event(key='d_batch_size', value=args.batch_size)
    mllogger.event(key='s_optimizer', value=args.optimizer)
    mllogger.event(key='s_network', value=args.network)
    mllogger.event(key='s_process', value=args.kv_store)
    mllogger.event(key=mllogger.constants.GLOBAL_BATCH_SIZE, value=args.batch_size * num_workers)
    mllogger.event(key='s_optimizer', value=args.optimizer)
    mllogger.event(key='s_network', value=args.network)
    mllogger.event(key='s_process', value=args.kv_store)
    mllogger.event(key=mllogger.constants.GRADIENT_ACCUMULATION_STEPS, value=1)

    if args.optimizer in {'sgdwlars', 'sgdwfastlars','sgdwfastlarsv2'}:
        optimizer_params['lars'] = True
        optimizer_params['lars_eta'] = args.lars_eta
        optimizer_params['lars_eps'] = args.lars_eps
        mllogger.event(key=mllogger.constants.OPT_NAME,
                              value='lars')
        mllogger.event(key=mllogger.constants.LARS_EPSILON,
                              value=args.lars_eps)
        mllogger.event(key=mllogger.constants.LARS_OPT_WEIGHT_DECAY,
                              value=args.wd)
        mllogger.event(key='lars_opt_momentum', value=optimizer_params['momentum'])
        mllogger.event(key='lars_opt_base_learning_rate', value=optimizer_params['learning_rate'])
        mllogger.event(key='lars_opt_learning_rate_warmup_epochs', value=args.warmup_epochs)
        mllogger.event(key=mllogger.constants.LARS_OPT_LR_DECAY_STEPS, value=args.num_epochs)
        if args.optimizer in {'sgdwfastlarsv2'}:
            aggregate_num = int(os.getenv('MXNET_OPTIMIZER_AGGREGATION_SIZE', "4")) + 1
            optimizer_params['base_lr'] = args.lr
            optimizer_params['end_lr'] = 0.0001
            optimizer_params['lr_decay_poly_power'] = 2
            optimizer_params['warmup_steps'] = get_epoch_size(args, kv) * args.warmup_epochs
            optimizer_params['total_steps'] = get_epoch_size(args, kv) * args.num_epochs
    else:
        mllogger.event(
            key=mllogger.constants.OPT_NAME,
            value='sgd')
        mllogger.event(
            key='sgd_opt_weight_decay',
            value=args.wd)
        mllogger.event(key='sgd_opt_momentum', value=optimizer_params['momentum'])
        mllogger.event(key='sgd_opt_learning_rate_decay_steps', value=args.num_epochs)
        mllogger.event(key='opt_learning_rate_warmup_epochs', value=args.warmup_epochs)
        mllogger.event(key='sgd_opt_base_learning_rate', value=optimizer_params['learning_rate'])

    if 'horovod' in args.kv_store:
        # Setting idx2name dictionary, required to mask out entries for weight decay.
        idx2name = {}
        for i,n in enumerate(model._exec_group.param_names):
            idx2name[i] = n

        opt = mx.optimizer.create(args.optimizer, sym=None, param_idx2name=idx2name, **optimizer_params)
        # Horovod: wrap optimizer with DistributedOptimizer
        opt = hvd.DistributedOptimizer(opt, num_groups=int(os.getenv('MXNET_HOROVOD_NUM_GROUPS', 1)))
    else:
        opt = args.optimizer

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create(
            'top_k_accuracy', top_k=args.top_k))

    # callbacks that run after each batch
    batch_end_callbacks = []
    # if 'horovod' in args.kv_store:
    #     # if using horovod, only report on rank 0 with global batch size
    #     if rank == 0:
    #         batch_end_callbacks.append(mx.callback.Speedometer(
    #             num_workers*args.batch_size, args.disp_batches))
    # else:
    #     batch_end_callbacks.append(mx.callback.Speedometer(
    #         args.batch_size, args.disp_batches))

    # init optimizer before update is called
    model.init_optimizer(kvstore=kv, optimizer=opt,
            optimizer_params = optimizer_params)

    # Allocating memory in MxNet and Enabling CUDA Graph Capture
    dummy_data = [mx.nd.zeros(
        shape=(args.batch_size, 224, 224, 4),
        dtype='float16',
        ctx=mx.gpu(hvd.local_rank()))]
    dummy_label = [mx.nd.zeros(
        shape=(args.batch_size,),
        dtype='float32',
        ctx=mx.gpu(hvd.local_rank()))]
    
    validation_metric = None
    if validation_metric is None:
        validation_metric = eval_metrics
    ###########################################################################
    # Adding Correct and Total Count metrics
    ###########################################################################
    if not isinstance(validation_metric, list):
        validation_metric = [validation_metric]

    validation_metric = mx.metric.create(validation_metric)

    if not isinstance(validation_metric, mx.metric.CompositeEvalMetric):
        vm = mx.metric.CompositeEvalMetric()
        vm.append(validation_metric)
        validation_metric = vm

    for m in [CorrectCount(), TotalCount()]:
        validation_metric.metrics.append(m)
    ###########################################################################
    
    output_arr = []
    if args.e2e_cuda_graphs:
        mpiwrapper.barrier()
        # Intialize all GPU buffers
        idata = mx.io.DataBatch(dummy_data,dummy_label)
        model.forward_backward(idata)
        mx.ndarray.waitall()
        model.update()
        mx.ndarray.waitall()
       
        input_arr  = [dummy_data[0], dummy_label[0]]
    
        for i in model._exec_group.param_arrays:
            if type(i) is list:
                output_arr.extend(i)
            else:
                output_arr.append(i)
    
        output_arr.append(model.get_outputs()[0])
    
        output_arr.append(opt.w_sum_sq)
        output_arr.append(opt.g_sum_sq)
        output_arr.append(opt.new_lrs)
        output_arr.append(opt.base_momentum)
        output_arr.append(opt.scaled_momentum)
        output_arr.append(opt.poly_lrs)
        output_arr.append(opt.old_poly_lrs)
        output_arr.append(opt.next_step)
        output_arr.append(opt.cur_step)
        output_arr.append(opt.new_wds)
        output_arr.append(opt.ones_gpu)
    
        for i in model._exec_group.grad_arrays:
            if type(i) is list:
                output_arr.extend(i)
            else:
                output_arr.append(i)
    
        if hvd.local_rank()==0:
            print("Start Graph Capture")
        graph_wrapper.start_capture(0, hvd.local_rank(), input_arr+output_arr)
        model.forward_backward(idata)
        model.update()
        graph_wrapper.end_capture(0, hvd.local_rank(), output_arr+input_arr)
        if hvd.local_rank()==0:
            print("End Graph Capture")
        mx.ndarray.waitall()
        graph_wrapper.finalize(hvd.local_rank())
        mx.ndarray.waitall()
        mpiwrapper.barrier()
    else:
        model.update()
        model.forward_backward(mx.io.DataBatch(dummy_data, dummy_label))
        model.forward_backward(mx.io.DataBatch(dummy_data, dummy_label))
        mx.ndarray.waitall()
        
    dummy_eval_data = SyntheticDataIter(args.num_classes, (args.batch_size, 224, 224, 4), 1, np.float32, args.input_layout)
    res = model.score(dummy_eval_data, validation_metric)
    mx.ndarray.waitall()
    res = model.score(dummy_eval_data, validation_metric)
    mx.ndarray.waitall()

    mllogger.log_init_stop_run_start()
    run_start_time = time.time()

    # initialize data iterators
    (train, val) = data_loader(args, kv)
    if 'dist' in args.kv_store and not 'async' in args.kv_store:
        logging.info('Resizing training data to %d batches per machine', epoch_size)
        # resize train iter to ensure each machine has same number of batches per epoch
        # if not, dist_sync can hang at the end with one machine waiting for other machines
        if not args.use_dali:
            train = mx.io.ResizeIter(train, epoch_size)

    # run
    last_epoch = mlperf_fit(model,
                            args,
                            train,
                            dummy_data,
                            dummy_label,
                            output_arr,
                            begin_epoch=0,
                            num_epoch=args.num_epochs,
                            eval_data=val,
                            eval_metric=eval_metrics,
                            kvstore=kv,
                            optimizer=opt,
                            optimizer_params=optimizer_params,
                            initializer=None if 'horovod' in args.kv_store else initializer,
                            arg_params=arg_params,
                            aux_params=aux_params,
                            batch_end_callback=batch_end_callbacks,
                            epoch_end_callback=epoch_end_callbacks, #checkpoint if args.use_dali else ,,
                            allow_missing=True,
                            eval_offset=args.eval_offset,
                            eval_period=args.eval_period,
                            accuracy_threshold=args.accuracy_threshold,
                            multi_gpu_per_process=(len(devs) > 1),
                            monitor=None,
                            run_start_time=run_start_time)

    # When using horovod, ensure all ops scheduled by the engine complete before exiting
    if 'horovod' in args.kv_store:
        mx.ndarray.waitall()

    if args.profile_server_suffix:
        mx.profiler.set_state(state='run', profile_process='server')
    if args.profile_worker_suffix:
        mx.profiler.set_state(state='run', profile_process='worker')
