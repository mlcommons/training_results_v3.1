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

import os
import time
import random
import argparse
import datetime
import math

import numpy as np
import torch
import torch.utils.data

import apex_C

from model.frozen_bn import FrozenBatchNorm2d

from mlperf_logger import mllogger
from mlperf_logging.mllog.constants import (SSD, STATUS, SUCCESS, ABORTED, INIT_START, INIT_STOP, RUN_START, RUN_STOP,
                                            SEED, GLOBAL_BATCH_SIZE, TRAIN_SAMPLES, EVAL_SAMPLES, EPOCH_COUNT,
                                            FIRST_EPOCH_NUM, OPT_NAME, ADAM, OPT_BASE_LR, OPT_WEIGHT_DECAY,
                                            OPT_LR_WARMUP_EPOCHS, OPT_LR_WARMUP_FACTOR, GRADIENT_ACCUMULATION_STEPS)

import utils
import presets
from coco.coco_utils import get_coco, get_openimages
from engine import train_one_epoch, evaluate
from model.retinanet import retinanet_from_backbone, cudnn_fusion_warmup
import model_capture
import apex
from syn_dataset import get_cached_dataset
from mlperf_common.scaleoutbridge import init_bridge, ScaleoutBridgeBase as SBridge
from mlperf_common.frameworks.pyt import PyTProfilerHandler, PyTCommunicationHandler
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook, fp16_compress_wrapper

from async_executor import async_executor

try:
    from dali import DaliDataIterator
except ImportError as err:
    print("Could not import DaliDataIterator, it's fine if you do not use --dali")


def get_dataset_fn(dataset, dataset_path):
    dataset_fn = None
    num_classes = None
    train_data_path = None
    train_annotations_file = None
    val_data_path = None
    val_annotations_file = None
    if dataset == 'coco':
        dataset_fn = get_coco
        num_classes = 91
        train_sz = 117266
        val_sz = 5000
        train_data_path = os.path.join(dataset_path, 'train2017')
        train_annotations_file = os.path.join(dataset_path, 'annotations', 'instances_train2017.json')
        val_data_path = os.path.join(dataset_path, 'val2017')
        val_annotations_file = os.path.join(dataset_path, 'annotations', 'instances_val2017.json')
    elif dataset == 'openimages':
        # Full openimages dataset
        dataset_fn = get_openimages
        num_classes = 601
        train_sz = 1743042
        val_sz = 41620
        train_data_path = os.path.join(dataset_path, 'train', 'data')
        train_annotations_file = os.path.join(dataset_path, 'train', 'labels', 'openimages.json')
        val_data_path = os.path.join(dataset_path, 'validation', 'data')
        val_annotations_file = os.path.join(dataset_path, 'validation', 'labels', 'openimages.json')
    elif dataset == 'openimages-mlperf':
        # L0 classes with more than 1000 samples
        dataset_fn = get_openimages
        num_classes = 264
        train_sz = 1170301
        val_sz = 24781
        train_data_path = os.path.join(dataset_path, 'train', 'data')
        train_annotations_file = os.path.join(dataset_path, 'train', 'labels', 'openimages-mlperf.json')
        val_data_path = os.path.join(dataset_path, 'validation', 'data')
        val_annotations_file = os.path.join(dataset_path, 'validation', 'labels', 'openimages-mlperf.json')
    else:
        assert False, "Unknown dataset = {dataset}"

    return (dataset_fn, num_classes, train_sz, val_sz,
            train_data_path, train_annotations_file, val_data_path, val_annotations_file)


def get_transform(train, data_augmentation):
    return presets.DetectionPresetTrain(data_augmentation) if train else presets.DetectionPresetEval()


def cast_frozen_bn_half(module: torch.nn.Module):
    for name, child_module in module.named_children():
        if isinstance(child_module, FrozenBatchNorm2d):
            child_module.half()

        elif len(list(child_module.children())) > 0:
            cast_frozen_bn_half(child_module)


def parse_args(add_help=True):
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    # Model
    parser.add_argument('--backbone', default='resnext50_32x4d',
                        choices=['resnet50', 'resnext50_32x4d', 'resnet101', 'resnext101_32x8d'],
                        help='The model backbone')
    parser.add_argument('--trainable-backbone-layers', default=3, type=int,
                        help='number of trainable layers of backbone')
    parser.add_argument('--sync-bn', dest='sync_bn', action="store_true", help='Use sync batch norm')
    parser.add_argument('--data-layout', default="channels_last", choices=['channels_first', 'channels_last'],
                        help="Model data layout")
    parser.add_argument("--amp", dest='amp', action="store_true",
                        help="Whether to enable Automatic Mixed Precision (AMP). "
                             "When false, uses TF32 on A100 and FP32 on V100 GPUS.")
    parser.add_argument("--no-amp", dest='amp', action="store_false",
                        help="Whether to enable Automatic Mixed Precision (AMP). "
                             "When false, uses TF32 on A100 and FP32 on V100 GPUS.")
    parser.set_defaults(amp=True)

    # Async validation
    parser.add_argument("--async-coco", action="store_true",
                        help="Enable asynchronous coco scoring")
    parser.add_argument("--async-coco-check-freq", default=20, type=int,
                        help="Enable asynchronous coco scoring")
    parser.add_argument("--num-eval-ranks", default=None, type=int,
                        help="Number of validation ranks. default to use")

    # Dataset
    parser.add_argument('--dataset', default='openimages-mlperf',
                        choices=['coco', 'openimages', 'openimages-mlperf'],
                        help='dataset')
    parser.add_argument('--dataset-path', default='/datasets/open-images-v6',
                        help='dataset root path')
    parser.add_argument('--num-classes', default=None, type=int,
                        help='Number of classes in the dataset. By default will be infered from --dataset')
    parser.add_argument('--train-data-path', default=None, type=str,
                        help='Training images folder. By default will be inferred from --dataset')
    parser.add_argument('--train-annotations-file', default=None, type=str,
                        help='Training annotations file. By default will be inferred from --dataset')
    parser.add_argument('--val-data-path', default=None, type=str,
                        help='Validation images folder. By default will be inferred from --dataset')
    parser.add_argument('--val-annotations-file', default=None, type=str,
                        help='Validation annotations file. By default will be inferred from --dataset')
    parser.add_argument('--image-size', default=[800, 800], nargs=2, type=int,
                        help='Image size for training')
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy')

    # Train parameters
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--max-iters-per-epoch', default=None, type=int, metavar='N',
                        help='maximum number of training iterations per epoch')
    parser.add_argument('--max-eval-iters-per-epoch', default=None, type=int, metavar='N',
                        help='maximum number of eval iterations per epoch')
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--output-dir', default=None, help='path where to save checkpoints.')
    parser.add_argument('--target-map', default=0.34, type=float, help='Stop training when target mAP is reached')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                        help="Use pre-trained models from the modelzoo")

    # Hyperparameters
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('-e', '--eval-batch-size', default=None, type=int,
                        help='evaluation images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--warmup-epochs', default=1, type=int,
                        help='how long the learning rate will be warmed up in fraction of epochs')
    parser.add_argument('--warmup-factor', default=1e-3, type=float,
                        help='factor for controlling warmup curve')

    # Other
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--eval-print-freq', default=None, type=int, help='eval print frequency')
    parser.add_argument("--test-only", dest="test_only", action="store_true", help="Only test the model")
    parser.add_argument('--seed', '-s', type=int, default=random.SystemRandom().randint(0, 2**32 - 1),
                        help='manually set random seed')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--cocoeval', default='nvidia',
                        choices=['python', 'nvidia'],
                        help='Choose the cocoeval implementation (nvidia is a much faster c++ implementation)')
    parser.add_argument('--coco-threads', default=8, type=int,
                        help='Number of threads to use with --coco=nvidia')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # optimizations
    parser.add_argument('--frozen-bn-opt', action="store_true", help='calculate frozen BN scale and bias only once')
    parser.add_argument('--frozen-bn-fp16', dest="frozen_bn_fp16",
                        action="store_true", help="cast frozen BN layers to fp16 (use with --amp)")
    parser.add_argument('--jit', action="store_true", help="enable fusing opportunities")
    parser.add_argument('--cuda-graphs', action="store_true", help='enable CUDA graphs')
    parser.add_argument('--cuda-graphs-eval', action="store_true", help='enable CUDA graphs in evaluation')
    parser.add_argument('--cls-head-pad', action="store_true",
                        help='pad classification head (used for CUDA graphs or just parallelization)')
    parser.add_argument('--reg-head-pad', action="store_true",
                        help='pad regression head (used for CUDA graphs or just parallelization)')
    parser.add_argument('--cuda-graphs-syn', action="store_true", help='using synthetic data for model capture')
    parser.add_argument('--model-warmup-epochs', default=16, type=int,
                        help='warmup model for JIT and cuDNN using synthetic data')
    parser.add_argument('--master-weights', default=False, action="store_true",
                        help='cast conv layers to FP16 and keep FP32 master weights for mixed precision training')

    # DALI
    parser.add_argument('--dali', action="store_true",
                        help='use DALI instead of native PyTorch dataloader during training')
    parser.add_argument('--dali-matched-idxs', action="store_true", help='compute matched_idxs within DALI')
    parser.add_argument('--dali-eval', action="store_true",
                        help='use DALI instead of native PyTorch dataloader during evaluation')
    parser.add_argument('--dali-eval-cache', action="store_true",
                        help='Cache test dataset during evaluation')
    parser.add_argument('--dali-prefetch-queue-depth', type=int, default=2, help='set DALI prefetch queue depth')
    parser.add_argument('--dali-cpu-decode', action="store_true",
                        help='use CPU-based DALI decoder instead of the mixed one')
    parser.add_argument('--dali-cmn', default=0, type=int,
                        help='which version of DALI crop-mirror-normalize to use (0: experimental, 1: regular, 2: transpose-normalize')
    parser.add_argument('--dali-cmn-hint', default=0, type=int,
                        help='hint of DALI crop-mirror-normalize output size in bytes')
    parser.add_argument('--dali-decoder-hint-height', default=7360, type=int,
                        help='hint of largest image height for memory preallocation provided to DALI image decoder')
    parser.add_argument('--dali-decoder-hint-width', default=7360, type=int,
                        help='hint of largest image width for memory preallocation provided to DALI image decoder')
    parser.add_argument('--dali-decoder-hw-load', default=0.65, type=float,
                        help='percentage of image data to be processed by DALI HW JPEG decoder')
    parser.add_argument('--dali-input-batch-multiplier', default=1, type=int,
                        help='use input batch multiplier for DALI')
    parser.add_argument('--dali-eval-cmn-hint', default=0, type=int,
                        help='hint of DALI crop-mirror-normalize output size in bytes for eval')
    parser.add_argument('--dali-eval-decoder-hint-height', default=0, type=int,
                        help='hint of largest image height for memory preallocation provided to DALI image decoder for eval')
    parser.add_argument('--dali-eval-decoder-hint-width', default=0, type=int,
                        help='hint of largest image width for memory preallocation provided to DALI image decoder for eval')
    parser.add_argument('--dali-eval-decoder-hw-load', default=0.65, type=float,
                        help='percentage of image data to be processed by DALI HW JPEG decoder for eval')
    parser.add_argument('--dali-eval-input-batch-multiplier', default=1, type=int,
                        help='use input batch multiplier for DALI, for eval')
    parser.add_argument('--dali-sync', action="store_true",
                        help='make DALI pipeline synchronous for debugging purposes')

    # apex optimizations
    parser.add_argument('--apex-adam', action="store_true", help="use APEX implementation of Adam")
    parser.add_argument('--apex-focal-loss', action="store_true", help="use APEX implementation of focal loss")
    parser.add_argument('--apex-backbone-fusion', action="store_true", help='using APEX conv-scale-bias-relu fusion')
    parser.add_argument('--apex-head-fusion', action="store_true", help='using APEX conv-bias-relu fusion')

    # communication optimizations
    parser.add_argument('--disable-ddp-broadcast-buffers', dest='broadcast_buffers', action='store_false',
                        help='disable DDP broadcast buffers (BNs are frozen)')
    parser.add_argument('--fp16-allreduce', action="store_true", help='using fp16 allreduce compression')
    parser.add_argument('--ddp-bucket-sz', default=25, type=int, help='DDP bucket size in MB')
    parser.add_argument('--ddp-first-bucket-sz', default=None, type=int, help='DDP first bucket size in MB')
    parser.add_argument('--no-gradient-as-bucket-view', default=False, action="store_true",
                        help='do not use gradient_as_bucket_view DDP optimization')

    # additional params
    parser.add_argument('--max-boxes', dest='max_boxes', type=int, default=1000,
                        help='pad the number of bboxes to max_boxes, used to make functions parallel')
    parser.add_argument('--cudnn-bench', dest='cudnn_bench', action='store_true',
                        help='set torch.backends.cudnn.benchmark')
    parser.add_argument('--deterministic', default=False, action='store_true',
                        help='set torch.use_deterministic_algorithms')
    parser.add_argument('--not-graphed-prologues', action='store_true', help='')
    parser.add_argument('--metric-loss', default=False, action='store_true',
                        help='record loss metric, incurs an additional allreduce every iteration')
    parser.add_argument('--syn-dataset', dest='syn_dataset', action='store_true',
                        help='it is actually a semi-synthetic dataset, since the original dataset is required')
    parser.add_argument('--sync-after-graph-replay', action='store_true',
                        help='this is a workaround for the scenario in which DALI is blocked due to optimizer sync '
                             'driver lock')
    parser.add_argument('--allreduce-barrier', default=False, action="store_true",
                        help='insert barrier before parameter allreduce (for debugging, cannot be used with CUDA Graphs)')
    parser.add_argument('--skip-eval', default=False, action="store_true",
                        help='skip evaluation, used when obtaining API logs')
    parser.add_argument('--cuda-profiler', action="store_true",
                        help='enable CUDA profiler start/stop for training')
    parser.add_argument('--cuda-profiler-eval', action="store_true",
                        help='enable CUDA profiler start/stop for eval')
    parser.add_argument('--cuda-profiler-start', default=-1, type=int, help='iteration to start CUDA profiler')
    parser.add_argument('--cuda-profiler-stop', default=-1, type=int, help='iteration to stop CUDA profiler')

    # power benchmark
    parser.add_argument("--power-benchmark", dest='power_benchmark', action="store_true",
                        help="If this is a power benchmark run, training epochs will increase by "
                             "10x and won't stop when the target mAP is achieved.")
    parser.add_argument("--not-power-benchmark", dest='power_benchmark', action="store_false",
                        help="If this is a power benchmark run, training epochs will increase by "
                             "10x and won't stop when the target mAP is achieved.")
    parser.set_defaults(power_benchmark="POWERCMDDIR" in os.environ)
    parser.add_argument("--power-sustain-time", default=600, type=int,
                        help="Sustain run for at least the specified seconds for power measurement")

    args = parser.parse_args()

    args.eval_batch_size = args.eval_batch_size or args.batch_size
    args.eval_print_freq = args.eval_print_freq or args.print_freq

    return args


def main(args):
    # CUDA graphs will only work if regression head tensors are padded
    assert((args.cuda_graphs and args.reg_head_pad and args.cls_head_pad) or
           (args.cuda_graphs and args.not_graphed_prologues) or
           (not args.cuda_graphs))
    # Do not use DALI when using synthetic data
    assert((args.dali and not args.syn_dataset) or not args.dali)
    # At the moment, to use JIT FrozenBN fusions, one must use the FrozenBN optimization flag
    assert((args.jit and args.frozen_bn_opt) or not args.jit)

    # Increase epochs to a large number if this is a power benchmark
    if args.power_benchmark:
        args.epochs *= 10

    # Enable JIT
    if args.jit:
        assert args.backbone == 'resnext50_32x4d',"JIT was only tested with ResNeXt50-32x4d."
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_bailout_depth(20)

    # Set whether PyTorch operations should use deterministic algorithms
    torch.use_deterministic_algorithms(args.deterministic)

    # Init distributed mode
    train_group, eval_group = utils.init_distributed_mode(args)

    # Start MLPerf benchmark
    mllogger.mlperf_submission_log(benchmark=SSD)
    mllogger.start(key=INIT_START, sync=True)

    if args.output_dir:
        utils.mkdir(args.output_dir)

    torch.backends.cudnn.benchmark = args.cudnn_bench

    device = torch.device(args.device)

    # set rank seeds according to MLPerf rules
    if args.distributed:
        args.seed = utils.broadcast(args.seed, src=1, group=None)
        args.seed = (args.seed + utils.get_rank()) % 2**32
    torch.manual_seed(args.seed)
    np.random.seed(seed=args.seed)
    mllogger.event(key=SEED, value=args.seed, unique=False)

    # Print args
    mllogger.event(key='local_batch_size', value=args.batch_size)
    mllogger.event(key=GLOBAL_BATCH_SIZE, value=args.batch_size*args.num_train_ranks)
    mllogger.event(key=EPOCH_COUNT, value=args.epochs)
    mllogger.event(key=FIRST_EPOCH_NUM, value=args.start_epoch)
    print(args)

    # Data loading code
    print("Getting dataset information")
    dataset_fn, num_classes, train_sz, val_sz, \
    train_data_path, train_annotations_file, val_data_path, val_annotations_file = \
        get_dataset_fn(dataset=args.dataset, dataset_path=args.dataset_path)
    args.num_classes = args.num_classes or num_classes
    args.train_sz = train_sz
    args.val_sz = val_sz
    args.train_data_path = args.train_data_path or train_data_path
    args.train_annotations_file = args.train_annotations_file or train_annotations_file
    args.val_data_path = args.val_data_path or val_data_path
    args.val_annotations_file = args.val_annotations_file or val_annotations_file

    print("Creating model")
    model = retinanet_from_backbone(backbone=args.backbone,
                                    num_classes=num_classes,
                                    image_size=args.image_size,
                                    data_layout=args.data_layout,
                                    pretrained=args.pretrained,
                                    trainable_backbone_layers=args.trainable_backbone_layers,
                                    jit=args.jit,
                                    backbone_fusion=args.apex_backbone_fusion,
                                    head_fusion=args.apex_head_fusion,
                                    frozen_bn_opt=args.frozen_bn_opt)
    model.to(device)

    if args.data_layout == 'channels_last':
        model = model.to(memory_format=torch.channels_last)

    # cast FrozenBatchNorm2d parameters to FP16
    if args.amp and args.frozen_bn_fp16 and args.frozen_bn_opt:
        cast_frozen_bn_half(module=model)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.master_weights:
        print("Casting convolutional layers to half")
        for m in model.modules():
            if m.__class__ in [torch.nn.Conv2d]:
                m.half()

    model_without_ddp = model
    if args.distributed:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            if args.ddp_first_bucket_sz is not None:
                torch.distributed._DEFAULT_FIRST_BUCKET_BYTES = args.ddp_first_bucket_sz * 1024 * 1024

            process_group = train_group if args.rank in args.train_ranks else eval_group
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              process_group=process_group,
                                                              device_ids=[args.gpu],
                                                              broadcast_buffers=args.broadcast_buffers,
                                                              bucket_cap_mb=args.ddp_bucket_sz,
                                                              gradient_as_bucket_view=not args.no_gradient_as_bucket_view)
            model_without_ddp = model.module

            if args.fp16_allreduce or args.allreduce_barrier:
                if args.fp16_allreduce and args.allreduce_barrier:
                    comm_hook = fp16_compress_wrapper(utils.barrier_hook)
                elif args.fp16_allreduce:
                    comm_hook = fp16_compress_hook
                elif args.allreduce_barrier:
                    comm_hook = utils.barrier_hook
                model.register_comm_hook(state=None, hook=comm_hook)

    params = [p for p in model.parameters() if p.requires_grad]
    if not args.apex_adam:
        optimizer = torch.optim.Adam(params, lr=args.lr)
    else:
        optimizer = apex.optimizers.FusedAdam(params, lr=args.lr, capturable=args.cuda_graphs, master_weights=args.master_weights)

    mllogger.event(key=OPT_NAME, value=ADAM)
    mllogger.event(key=OPT_BASE_LR, value=args.lr)
    mllogger.event(key=OPT_WEIGHT_DECAY, value=0)
    mllogger.event(key=OPT_LR_WARMUP_EPOCHS, value=args.warmup_epochs)
    mllogger.event(key=OPT_LR_WARMUP_FACTOR, value=args.warmup_factor)
    mllogger.event(key=GRADIENT_ACCUMULATION_STEPS, value=1)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    # using default feature sizes to create anchors (happens just once)
    model_without_ddp.update_anchors(torch.Size([args.batch_size, 3, args.image_size[0], args.image_size[1]]), device,
                                     dtype=(torch.float16 if args.amp else torch.float32))

    # no need for eval warmup here, since warmup is also part of graph capture
    if args.model_warmup_epochs > 0 and not args.cuda_graphs_eval:
        print('Model eval warmup')
        assert(args.dataset == 'openimages-mlperf' and args.image_size == [800, 800])
        start_time = time.time()

        bs_list = [args.eval_batch_size]

        eval_sz = args.val_sz
        eval_last_iter_bs = int(math.ceil(float(eval_sz) % (args.num_eval_ranks * args.eval_batch_size) / args.num_eval_ranks))
        if eval_last_iter_bs > 0:
            if eval_last_iter_bs != args.eval_batch_size:
                bs_list.append(eval_last_iter_bs)
            if eval_last_iter_bs > 1:
                bs_list.append(eval_last_iter_bs - 1)

        # Skip eval warmup for last batch size of training
        # train_sz = args.train_sz
        # train_last_iter_bs = int(math.ceil(float(train_sz) % (world_size * args.batch_size) / world_size))
        # if train_last_iter_bs > 0 and train_last_iter_bs != eval_last_iter_bs:
        #     bs_list.append(train_last_iter_bs)

        for bs in bs_list:
            model_capture.model_eval_warmup(model, bs, args.model_warmup_epochs, args)

        #cudnn_fusion_warmup(bs_list)

        total_time = time.time() - start_time
        print('Time: {} sec'.format(total_time))

    # GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    data_loader = None
    data_loader_test = None

    # The dali based data_loader doesn't touch data at init time (lazy_init=True). So we place before after RUN_START
    if args.dali and (args.rank in args.train_ranks) and (not args.test_only):
        print("Creating Dali training dataloader")
        data_loader = DaliDataIterator(data_path=args.train_data_path,
                                       anno_path=args.train_annotations_file,
                                       batch_size=args.batch_size,
                                       num_shards=args.num_train_ranks,
                                       shard_id=args.train_rank,
                                       is_training=True,
                                       image_size=args.image_size,
                                       num_threads=args.workers,
                                       prefetch_queue_depth=args.dali_prefetch_queue_depth,
                                       compute_matched_idxs=args.dali_matched_idxs,
                                       anchors=model_without_ddp.anchors,
                                       cpu_decode=args.dali_cpu_decode,
                                       lazy_init=True,
                                       cache=False,
                                       cmn=args.dali_cmn,
                                       cmn_hint=args.dali_cmn_hint,
                                       decoder_hint_h=args.dali_decoder_hint_height,
                                       decoder_hint_w=args.dali_decoder_hint_width,
                                       decoder_hw_load=args.dali_decoder_hw_load,
                                       input_batch_multiplier=args.dali_input_batch_multiplier,
                                       sync=args.dali_sync,
                                       seed=args.seed)

    if args.dali and (args.rank in args.eval_ranks):
        print("Creating Dali eval dataloader")
        data_loader_test = DaliDataIterator(data_path=args.val_data_path,
                                            anno_path=args.val_annotations_file,
                                            batch_size=args.eval_batch_size,
                                            num_shards=args.num_eval_ranks,
                                            shard_id=args.eval_rank,
                                            is_training=False,
                                            image_size=args.image_size,
                                            num_threads=args.workers,
                                            prefetch_queue_depth=args.dali_prefetch_queue_depth,
                                            compute_matched_idxs=False,
                                            anchors=model_without_ddp.anchors,
                                            cpu_decode=args.dali_cpu_decode,
                                            lazy_init=True,
                                            cache=args.dali_eval_cache,
                                            cmn=args.dali_cmn,
                                            cmn_hint=args.dali_eval_cmn_hint,
                                            decoder_hint_h=args.dali_eval_decoder_hint_height,
                                            decoder_hint_w=args.dali_eval_decoder_hint_width,
                                            decoder_hw_load=args.dali_eval_decoder_hw_load,
                                            input_batch_multiplier=args.dali_eval_input_batch_multiplier,
                                            seed=args.seed)

    # Preparing CUDA graph using the synthetic data
    graphed_model, static_input, static_loss, static_prologues_out = None, None, None, None
    start_iter = torch.tensor([0]).to(device)
    if args.cuda_graphs and args.cuda_graphs_syn:
        # TODO: Compare against case without CUDA Graphs
        dataset_len = int(args.train_sz / args.batch_size / utils.get_world_size())
        warmup_iters = args.warmup_epochs * dataset_len
        utils.ScratchPad.tensor_one = torch.tensor([1]).to(device)
        lr_scheduler = utils.graphable_warmup_lr_scheduler(optimizer, start_iter, warmup_iters, args.warmup_factor)

        graphed_model, static_input, static_loss, static_prologues_out = \
            model_capture.whole_model_capture(model, optimizer, scaler, None, lr_scheduler, args)

    graphed_model_eval, static_input_eval, static_model_output_eval = None, None, None
    if args.cuda_graphs_eval and args.cuda_graphs_syn:
        graphed_model_eval, static_input_eval, static_model_output_eval = \
            model_capture.whole_model_capture_eval(model, None, args)

    mllogger.end(key=INIT_STOP, sync=True)

    start_time = time.time()
    sbridge = init_bridge(PyTProfilerHandler(), PyTCommunicationHandler(), mllogger)
    mllogger.start(key=RUN_START, sync=True)
    sbridge.start_prof(SBridge.LOAD_TIME)

    # The pytorch based data_loader touches data at init time. So we place it after RUN_START
    if not args.dali and (args.rank in args.train_ranks) and (not args.test_only):
        print("Creating PyTorch training dataloader")
        dataset = dataset_fn(dataset_path=args.train_data_path,
                             annotations_file=args.train_annotations_file,
                             transforms=get_transform(True, args.data_augmentation),
                             training=True)
        if args.syn_dataset:
            data_loader = get_cached_dataset(model_without_ddp, dataset, device, args)
        else:
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset,
                                                                                num_replicas=args.num_train_ranks,
                                                                                rank=args.train_rank)
            else:
                train_sampler = torch.utils.data.RandomSampler(dataset)
            train_batch_sampler = torch.utils.data.BatchSampler(sampler=train_sampler,
                                                                batch_size=args.batch_size,
                                                                drop_last=True)
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
                pin_memory=True, collate_fn=utils.collate_fn)

    if not args.dali and (args.rank in args.eval_ranks):
        print("Creating PyTorch eval dataloader")
        dataset_test = dataset_fn(dataset_path=args.val_data_path,
                                    annotations_file=args.val_annotations_file,
                                    transforms=get_transform(False, args.data_augmentation),
                                    training=False)
        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset_test,
                                                                            num_replicas=args.num_eval_ranks,
                                                                            rank=args.eval_rank,
                                                                            shuffle=False)
        else:
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.eval_batch_size or args.batch_size,
            sampler=test_sampler, num_workers=args.workers,
            pin_memory=True, collate_fn=utils.collate_fn)

    sbridge.stop_prof(SBridge.LOAD_TIME)

    if (args.rank in args.train_ranks) and (not args.test_only):
        mllogger.event(key=TRAIN_SAMPLES, value=len(data_loader))

    if args.rank in args.eval_ranks:
        mllogger.event(key=EVAL_SAMPLES, value=len(data_loader_test))

    # CUDA graph capture without synthetic data - currently not functional
    if args.cuda_graphs and not args.cuda_graphs_syn:
        graphed_model, static_input, static_loss, static_prologues_out = \
            model_capture.whole_model_capture(model, optimizer, scaler, data_loader, args)

    start_iter.copy_(torch.tensor([0]).to(device))

    if args.cuda_graphs_eval and not args.cuda_graphs_syn:
        graphed_model_eval, static_input_eval, static_model_output_eval = \
            model_capture.whole_model_capture_eval(model, data_loader_test, args)

    print("Running ...")
    status = ABORTED
    accuracy = None

    def stop_training(args, status, start_time):
        """
        Determine whether to stop the training based on various conditions.
        """
        if status != SUCCESS:
            return False

        if not args.power_benchmark:
            return True

        elapsed_time = time.time() - start_time
        if elapsed_time >= args.power_sustain_time:
            print(f"Ending sustained run for power measurement (target time: {args.power_sustain_time}, elapsed time: {elapsed_time})")
            return True

        print(f"Continuing run for power measurement (target time: {args.power_sustain_time}, elapsed time: {elapsed_time})")
        return False

    if args.test_only and (args.rank in args.eval_ranks):
        accuracy = evaluate(model=model,
                            data_loader=data_loader_test,
                            device=device,
                            epoch=None,
                            eval_group=eval_group,
                            args=args,
                            graphed_model=graphed_model_eval, static_input=static_input_eval,
                            static_output=static_model_output_eval,
                            sbridge=sbridge)
        print(f'Model mAP = {accuracy}')
        if args.target_map and accuracy and accuracy >= args.target_map:
            status = SUCCESS
            mllogger.end(key=RUN_STOP, metadata={"status": status}, sync=True)
    else:
        for epoch in range(args.start_epoch, args.epochs):
            ############################################################################################################
            # Train
            ############################################################################################################
            if args.rank in args.train_ranks:
                if args.distributed and not args.dali and not args.syn_dataset:
                    train_sampler.set_epoch(epoch)

                metric_logger, status = train_one_epoch(model=model,
                                                        optimizer=optimizer,
                                                        scaler=scaler,
                                                        data_loader=data_loader,
                                                        device=device,
                                                        epoch=epoch,
                                                        train_group=train_group,
                                                        args=args,
                                                        graphed_model=graphed_model,
                                                        static_input=static_input,
                                                        static_loss=static_loss,
                                                        static_prologues_out=static_prologues_out,
                                                        sbridge=sbridge,
                                                        status=status)
                if args.output_dir:
                    checkpoint = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch+1,
                        'args': args,
                    }
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(args.output_dir, 'model_{}.pth'.format(epoch+1)))
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(args.output_dir, 'checkpoint.pth'))

                # check if we should stop training
                if stop_training(args, status, start_time):
                    break
            ############################################################################################################

            ############################################################################################################
            # Sync train and val ranks (only if they are on different sets of nodes)
            ############################################################################################################
            if args.eval_ranks != args.train_ranks:
                # Do we need to sync params without grads?
                params = [param for param in model.parameters()]
                # params = [param for param in model.parameters() if param.requires_grad]
                flat_params = apex_C.flatten(params)
                # sync train and val
                utils.barrier(group=None)
                # broadcast train->val (actually train(0)->all)
                torch.distributed.broadcast(flat_params, 0)
            ############################################################################################################

            ############################################################################################################
            # Validation
            ############################################################################################################
            if args.rank in args.eval_ranks and not args.skip_eval:
                accuracy = evaluate(model=model,
                                    data_loader=data_loader_test,
                                    device=device,
                                    epoch=epoch+1,
                                    eval_group=eval_group,
                                    args=args,
                                    graphed_model=graphed_model_eval, static_input=static_input_eval,
                                    static_output=static_model_output_eval,
                                    sbridge=sbridge)
                if args.target_map and accuracy and accuracy >= args.target_map and status != SUCCESS:
                    status = SUCCESS
                    mllogger.end(key=RUN_STOP, metadata={"status": status}, sync=True)

                # check if we should stop training
                if stop_training(args, status, start_time):
                    break
            ############################################################################################################

    # Wait for async coco jobs if necessary
    if args.async_coco:
        while status != SUCCESS and len(async_executor.tags()):
            # FIXME(ahmadki): --num-eval-ranks
            if args.eval_rank == 0:
                results = async_executor.pop_if_done()
                # in case of multiple results are returned, get the highest mAP
                if results and len(results) > 0:
                    accuracy = max([result['bbox'][0] for result in results.values() if result], default=-1)

            if args.distributed:
                accuracy = utils.broadcast(accuracy, 0, group=None)

            if args.target_map and accuracy and accuracy >= args.target_map:
                status = SUCCESS
                mllogger.end(key=RUN_STOP, metadata={"status": status}, sync=True)

    run_time = time.time() - start_time
    run_time_str = str(datetime.timedelta(seconds=int(run_time)))
    print('Run time {}'.format(run_time_str))
    mllogger.event(key=STATUS, value=status, unique=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
