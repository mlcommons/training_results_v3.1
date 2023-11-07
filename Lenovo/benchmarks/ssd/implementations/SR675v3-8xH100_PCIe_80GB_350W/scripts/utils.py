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

from collections import defaultdict, deque
import datetime
import errno
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import _allreduce_fut
import weakref
from functools import wraps


class ScratchPad:
    target_n = None
    target_labels_padded = None
    target_boxes_padded = None
    target_matched_idxs = None
    gt_classes_target = None
    batch_size_vector = None
    tensor_one = None


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, group=None):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier(group=group)
        dist.all_reduce(t, group=group)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data, group):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = group.size() if group else get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(object_list=data_list, obj=data, group=group)
    return data_list


def broadcast(data, src, group):
    """
    Run broadcast on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
        src: Source rank from which to broadcast data
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = group.size() if group else get_world_size()
    if world_size == 1:
        return data
    data_list = data if isinstance(data, list) else [data]
    dist.broadcast_object_list(object_list=data_list, src=src, group=group)
    return data_list if isinstance(data, list) else data_list[0]


def reduce_dict(input_dict, group, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = group.size() if group else get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(tensor=values, group=group)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class SimpleTimer(object):
    def __init__(self, prefix=""):
        self.prefix = prefix

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        run_time = self.end - self.start
        print(f"{self.prefix}{run_time}")


class MetricLogger(object):
    def __init__(self, max_iters=None, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.summary = defaultdict(lambda: None)
        self.current_iter = 0
        self.max_iters = max_iters

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self, group=None):
        for meter in self.meters.values():
            meter.synchronize_between_processes(group=group)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        self.current_iter = 0
        self.summary['samples'] = 0
        if not header:
            header = ''
        start_time = time.time()
        self.summary['start_time'] = start_time
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if self.current_iter % print_freq == 0 or self.current_iter == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - self.current_iter)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        self.current_iter, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        self.current_iter, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            self.current_iter += 1
            end = time.time()
            self.summary['samples'] += len(obj[0])
            self.summary['end_time'] = end
            if self.max_iters and self.current_iter >= self.max_iters:
                break

        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def collate_fn(batch):
    return tuple(zip(*batch))


class _LRScheduler(object):

    def __init__(self, optimizer, last_epoch=-1, verbose=False):

        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = [group['initial_lr'].clone().cuda() for group in optimizer.param_groups]
        self.last_epoch = torch.tensor([last_epoch]).cuda()

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = torch.tensor(0).cuda()
        self._step_count = torch.tensor(0).cuda()
        self.verbose = verbose

        self.step()

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self):
        """ Return last computed learning rate by current scheduler.
        """
        return self._last_lr

    def get_lr(self):
        # Compute learning rate using chainable form of the scheduler
        raise NotImplementedError

    def print_lr(self, is_verbose, group, lr, epoch=None):
        """Display the current learning rate.
        """
        if is_verbose:
            if epoch is None:
                print('Adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(group, lr))
            else:
                print('Epoch {:5d}: adjusting learning rate'
                      ' of group {} to {:.4e}.'.format(epoch, group, lr))


    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        #if self._step_count == 1:
        #    if not hasattr(self.optimizer.step, "_with_counter"):
        #        warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
        #                      "initialization. Please, make sure to call `optimizer.step()` before "
        #                      "`lr_scheduler.step()`. See more details at "
        #                      "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
        #    elif self.optimizer._step_count < 1:
        #        warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        #                      "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
        #                      "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
        #                      "will result in PyTorch skipping the first value of the learning rate schedule. "
        #                      "See more details at "
        #                      "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch.copy_(epoch)
                if hasattr(self, "_get_closed_form_lr"):
                    values = self._get_closed_form_lr()
                else:
                    values = self.get_lr()

        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'].copy_(lr[0])
            self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class LambdaLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    times a given function. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # Assuming optimizer has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, optimizer, lr_lambda, last_epoch=-1, verbose=False):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(LambdaLR, self).__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """

        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """

        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


def graphable_warmup_lr_scheduler(optimizer, start_iter, warmup_iters, warmup_factor):

    def f(x):
        if warmup_iters == 0:
            return ScratchPad.tensor_one

        x = x + start_iter
        alpha = x.float() / warmup_iters
        ret = warmup_factor * (1 - alpha) + alpha

        return torch.where(x >= warmup_iters, ScratchPad.tensor_one, ret)

    return LambdaLR(optimizer, f)


def warmup_lr_scheduler(optimizer, start_iter, warmup_iters, warmup_factor):

    def f(x):
        x = x + start_iter
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def barrier(group):
    if not is_dist_avail_and_initialized():
        return
    torch.distributed.barrier(group)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.num_train_ranks = 1
        args.num_eval_ranks = 1
        args.rank = 0
        args.ranks = 1
        args.train_ranks = [0]
        args.eval_ranks = [0]
        args.train_rank = 0
        args.eval_rank = 0
        return None, None

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}')
    if args.cuda_graphs or args.cuda_graphs_eval:
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

    # Disable SHARP for subsequent process groups
    if os.environ.get("NCCL_COLLNET_ENABLE", "0") != "0":
        os.environ["NCCL_SHARP_DISABLE"] = "1"
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    args.ranks = list(range(args.world_size))
    if args.num_eval_ranks is None:
        args.num_train_ranks = args.world_size
        args.num_eval_ranks = args.world_size

        args.train_ranks = args.ranks
        args.eval_ranks = args.ranks
        args.train_rank = args.rank
        args.eval_rank = args.rank
    else:
        args.num_train_ranks = args.world_size - args.num_eval_ranks

        args.train_ranks = args.ranks[:args.num_train_ranks]
        args.eval_ranks = args.ranks[args.num_train_ranks:]
        args.train_rank = args.rank
        args.eval_rank = args.rank - args.num_train_ranks

    assert 1<=args.num_train_ranks<=args.world_size, "Number of training ranks must be between 1 and world size"
    assert 1<=args.num_eval_ranks<=args.world_size, "Number of validation ranks must be between 1 and world size"

    setup_for_distributed(args.train_rank==0 or args.eval_rank==0)

    # Use default process group (None) for both training and validation
    train_group = None
    eval_group = None
    return train_group, eval_group

def barrier_hook(process_group: dist.ProcessGroup, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    # Performs a barrier before allreduce
    dist.barrier(process_group)
    return _allreduce_fut(process_group, bucket.buffer())
