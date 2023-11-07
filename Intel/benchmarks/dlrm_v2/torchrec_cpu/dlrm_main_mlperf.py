import argparse
import itertools
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint

from typing import List, Optional
import time

import numpy as np
import mlperf_logging.mllog as mllog
import mlperf_logging.mllog.constants as mllog_constants
import torch
import torchmetrics as metrics
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.optim import MLPerfAdagrad

import dlrm_dist_mlperf as ext_dist
from torch.utils.data import DataLoader

DEFAULT_CAT_NAMES = [
    'cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
    'cat_6', 'cat_7', 'cat_8', 'cat_9', 'cat_10', 'cat_11',
    'cat_12', 'cat_13', 'cat_14', 'cat_15', 'cat_16', 'cat_17',
    'cat_18', 'cat_19', 'cat_20', 'cat_21', 'cat_22', 'cat_23',
    'cat_24', 'cat_25'
]


DEFAULT_INT_NAMES = ['int_0', 'int_1', 'int_2', 'int_3', 'int_4', 'int_5', 'int_6', 'int_7', 'int_8', 'int_9', 'int_10', 'int_11', 'int_12']

from dlrm_model_mlperf import DLRMMLPerf
from torch.autograd.profiler import record_function
from tqdm import tqdm

from data.dlrm_dataloader import get_dataloader
from lr_scheduler import LRPolicyScheduler
from mlperf_logging_utils import submission_info
from multi_hot import Multihot, RestartableMap


ADAGRAD_LR_DECAY = 0
ADAGRAD_INIT_ACC = 0
ADAGRAD_EPS = 1e-8
WEIGHT_DECAY = 0
mllogger = mllog.get_mllogger()

class InteractionType(Enum):
    ORIGINAL = "original"
    DCN = "dcn"
    PROJECTION = "projection"

    def __str__(self):
        return self.value

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="number of epochs to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size to use for training",
    )
    parser.add_argument(
        "--drop_last_training_batch",
        dest="drop_last_training_batch",
        action="store_true",
        help="Drop the last non-full training batch",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=None,
        help="batch size to use for validation and testing",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="number of test batches",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo_1t",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )
    parser.add_argument(
        "--interaction_branch1_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch1 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--interaction_branch2_layer_sizes",
        type=str,
        default="2048,2048",
        help="Comma separated layer sizes for interaction branch2 (only on dlrm with projection).",
    )
    parser.add_argument(
        "--dcn_num_layers",
        type=int,
        default=3,
        help="Number of DCN layers in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--dcn_low_rank_dim",
        type=int,
        default=512,
        help="Low rank dimension for DCN in interaction layer (only on dlrm with DCN).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="--mmap_mode mmaps the dataset."
        " That is, the dataset is kept on disk but is accessed as if it were in memory."
        " --mmap_mode is intended mostly for faster debugging. Use --mmap_mode to bypass"
        " preloading the dataset when preloading takes too long or when there is "
        " insufficient memory available to load the full dataset.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the Criteo dataset npy files.",
    )
    parser.add_argument(
        "--synthetic_multi_hot_criteo_path",
        type=str,
        default=None,
        help="Directory path containing the MLPerf v2 synthetic multi-hot dataset npz files.",
    )
    parser.add_argument(
        "--dense_labels_path",
        type=str,
        default=None,
        help="Directory path containing the MLPerf v2 synthetic multi-hot dataset npz files.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--shuffle_training_set",
        dest="shuffle_training_set",
        action="store_true",
        help="Shuffle the training set in memory. This will override mmap_mode",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
    )
    parser.add_argument(
        "--validation_auroc",
        type=float,
        default=None,
        help="Validation AUROC threshold to stop training once reached.",
    )
    parser.add_argument(
        "--evaluate_on_epoch_end",
        action="store_true",
        help="Evaluate using validation set on each epoch end.",
    )
    parser.add_argument(
        "--evaluate_on_training_end",
        action="store_true",
        help="Evaluate using test set on training end.",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    parser.add_argument(
        "--interaction_type",
        type=InteractionType,
        choices=list(InteractionType),
        default=InteractionType.ORIGINAL,
        help="Determine the interaction type to be used (original, dcn, or projection)"
        " default is original DLRM with pairwise dot product",
    )
    parser.add_argument(
        "--collect_multi_hot_freqs_stats",
        dest="collect_multi_hot_freqs_stats",
        action="store_true",
        help="Flag to determine whether to collect stats on freq of embedding access.",
    )
    parser.add_argument(
        "--multi_hot_sizes",
        type=str,
        default=None,
        help="Comma separated multihot size per sparse feature. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--multi_hot_distribution_type",
        type=str,
        choices=["uniform", "pareto"],
        default=None,
        help="Multi-hot distribution options.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr_decay_start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--print_lr",
        action="store_true",
        help="Print learning rate every iteration.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Enable TensorFloat-32 mode for matrix multiplications on A100 (or newer) GPUs.",
    )
    parser.add_argument(
        "--print_sharding_plan",
        action="store_true",
        help="Print the sharding plan used for each embedding table.",
    )
    parser.add_argument(
        "--print_progress",
        action="store_true",
        help="Print tqdm progress bar during training and evaluation.",
    )
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        help="Evaluate using test set on training end.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=24,
    )
    return parser.parse_args(argv)

def _evaluate(
        dlrm_model: torch.nn.Module,
        limit_batches: Optional[int],
        eval_dataloader: DataLoader,
        stage: str,
        epoch_num: float,
        log_eval_samples: bool,
        print_progress: bool) -> float:
    """
    Evaluates model. Computes and prints AUROC. Helper function for train_val_test.

    Args:
        limit_batches (Optional[int]): Limits the dataloader to the first `limit_batches` batches.
        eval_pipeline (TrainPipelineSparseDist): pipelined model.
        eval_dataloader (DataLoader): Dataloader for either the validation set or test set.
        stage (str): "val" or "test".
        epoch_num (float): Iterations passed as epoch fraction (for logging purposes).
        log_eval_samples (bool): Whether to print mllog with the number of samples.
        print_progress (bool): Whether to print tqdm progress bar.

    Returns:
        float: auroc result
    """
    is_rank_zero = ext_dist.is_rank_zero()
    if is_rank_zero:
        pbar = tqdm(
            iter(int, 1),
            desc=f"Evaluating {stage} set",
            total=len(eval_dataloader),
            disable=not print_progress,
        )
        mllogger.start(
            key=mllog_constants.EVAL_START,
            metadata={mllog_constants.EPOCH_NUM: epoch_num},
        )


    # Set eval_pipeline._connected to False to cause the pipeline to refill with new batches as if it were newly created and empty.

    # Two filler batches are appended to the end of the iterator to keep the pipeline active while the
    # last two remaining batches are still in progress awaiting results.

    auroc = metrics.AUROC(compute_on_step=False, task='binary')

    it = 0
    with torch.no_grad():
        for (densex, index, offset, labels) in eval_dataloader:
            try:
                # can be merge into pipeline
                _, (_loss, logits, _) = dlrm_model(densex, index, offset, labels)
                labels = labels.float()
                preds = torch.sigmoid(logits)
                it += 1
                if ext_dist.my_size > 1:
                    preds = ext_dist.all_gather(preds, None)
                    # labels = ext_dist.all_gather(labels, None)
                if preds.shape[0] != labels.shape[0]:
                    preds = preds[:labels.shape[0]]
                auroc(preds, labels)
                if is_rank_zero:
                    pbar.update(1)
            except StopIteration:
                # Dataset traversal complete
                break

    auroc_result = auroc.compute().item()
    num_samples = torch.tensor(sum(map(len, auroc.target)), device='cpu')
    #ext_dist.reduce(num_samples, 0, op=ext_dist.dist.ReduceOp.SUM)
    num_samples = num_samples.item()

    if is_rank_zero:
        print(f"AUROC over {stage} set: {auroc_result}.")
        print(f"Number of {stage} samples: {num_samples}")
        mllogger.event(
            key=mllog_constants.EVAL_ACCURACY,
            value=auroc_result,
            metadata={mllog_constants.EPOCH_NUM: epoch_num},
        )
        mllogger.end(
            key=mllog_constants.EVAL_STOP,
            metadata={mllog_constants.EPOCH_NUM: epoch_num},
        )
        if log_eval_samples:
            mllogger.event(
                key=mllog_constants.EVAL_SAMPLES,
                value=num_samples,
                metadata={mllog_constants.EPOCH_NUM: epoch_num},
            )
    return auroc_result

def _train(
        dlrm_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epoch: int,
        lr_scheduler,
        print_lr: bool,
        validation_freq: Optional[int],
        validation_auroc: Optional[float],
        limit_train_batches: Optional[int],
        limit_val_batches: Optional[int],
        print_progress: bool,
        enable_profiling: bool) -> bool:
    """
    Trains model for 1 epoch. Helper function for train_val_test.

    Args:
        train_pipeline (TrainPipelineSparseDist): pipelined model used for training.
        val_pipeline (TrainPipelineSparseDist): pipelined model used for validation.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        epoch (int): The number of complete passes through the training set so far.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.
        print_lr (bool): Whether to print the learning rate every training step.
        validation_freq (Optional[int]): The number of training steps between validation runs within an epoch.
        validation_auroc (Optional[float]): AUROC level desired for stopping training.
        limit_train_batches (Optional[int]): Limits the training set to the first `limit_train_batches` batches.
        limit_val_batches (Optional[int]): Limits the validation set to the first `limit_val_batches` batches.
        print_progress (bool): Whether to print tqdm progress bar.

    Returns:
        bool: Whether the validation_auroc threshold is reached.
    """
    # Set train_pipeline._connected to False to cause the pipeline to refill with new batches as if it were newly created and empty.

    # Two filler batches are appended to the end of the iterator to keep the pipeline active while the
    # last two remaining batches are still in progress awaiting results.

    is_rank_zero = ext_dist.is_rank_zero()
    if is_rank_zero:
        pbar = tqdm(
            # iter(int, 1),
            desc=f"Epoch {epoch}",
            total=len(train_dataloader),
            # disable=not print_progress,
        )

    it = 1
    is_success = False
    is_first_eval = True
    with torch.autograd.profiler.profile(enable_profiling, record_shapes=True) as prof:
        it = 1;
        tf = 0
        tz = 0
        tb = 0
        ts = 0
        for (densex, index, offset, labels) in train_dataloader:
            # with record_function("Prof_itertools"):
            tfs = time.time()
            # with record_function("forward"):
            (losses, output) = dlrm_model(densex, index, offset, labels)
            tfe = time.time()
            tzs = time.time()
            optimizer.zero_grad()
            tze = time.time()
            tbs = time.time()
            # with record_function("backward"):
            diff = torch.sum(losses).backward()
            tbe = time.time()
            tss = time.time()
            optimizer.step()
            lr_scheduler.step()
            tse = time.time()
            tf += tfe - tfs
            tz += tze - tzs
            tb += tbe - tbs
            ts += tse - tss
            if is_rank_zero:
                pbar.update(1)
            if validation_freq and it % validation_freq == 0:
                epoch_num = epoch + it / len(train_dataloader)
                auroc_result = _evaluate(
                    dlrm_model,
                    limit_val_batches,
                    val_dataloader,
                    "val",
                    epoch_num,
                    is_first_eval and epoch == 0,
                    print_progress,
                )
                is_first_eval = False
                if validation_auroc is not None and auroc_result >= validation_auroc:
                    ext_dist.barrier()
                    if is_rank_zero:
                        mllogger.end(
                            key=mllog_constants.RUN_STOP,
                            metadata={
                                mllog_constants.STATUS: mllog_constants.SUCCESS,
                                mllog_constants.EPOCH_NUM: epoch_num,
                            },
                        )
                    is_success = True
                    break
                # train_pipeline._model.train()

            if enable_profiling and it == 40:
                break
            # end of record_function
            it += 1
            if (it % 10 == 0):
                print(f"{it}-th Forward {tf*100:04.2f}  Backward {tb*100:04.2f} "
                      f"Zerograd {tz*100:04.2f} UpdateWeight {ts*100:04.2f} ms/it")
                tf = 0
                tb = 0
                tz = 0
                ts = 0
            # except StopIteration:
            #     # Dataset traversal complete
            #     break

    if is_rank_zero:
        print("Total number of iterations:", it - 1)

        if enable_profiling:
            time_stamp = str(time.strftime('%Y%m%d%H%M', time.localtime(time.time())))
            with open("prof_" + time_stamp + "_shape_cpu_time.prof", "w") as prof_f:
                prof_f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total"))
            with open("prof_" + time_stamp + "_shape_self_cpu_time.prof", "w") as prof_f:
                prof_f.write(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))
            with open("prof_" + time_stamp + "_total.prof", "w") as prof_f:
                prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
            prof.export_chrome_trace("prof_" + time_stamp + ".json")
            print(prof.key_averages().table(sort_by="cpu_time_total"))

    return is_success

@dataclass
class TrainValTestResults:
    val_aurocs: List[float] = field(default_factory=list)
    test_auroc: Optional[float] = None

def train_val_test(
        args: argparse.Namespace,
        dlrm_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        lr_scheduler: LRPolicyScheduler) -> TrainValTestResults:
    """
    Train/validation/test loop.

    Args:
        args (argparse.Namespace): parsed command line args.
        dlrm_model (torch.nn.Module): model to train.
        optimizer (torch.optim.Optimizer): optimizer to use.
        device (torch.device): device to use.
        train_dataloader (DataLoader): Training set's dataloader.
        val_dataloader (DataLoader): Validation set's dataloader.
        test_dataloader (DataLoader): Test set's dataloader.
        lr_scheduler (LRPolicyScheduler): Learning rate scheduler.

    Returns:
        TrainValTestResults.
    """

    results = TrainValTestResults()

    is_rank_zero = ext_dist.is_rank_zero()

    epoch = 0
    is_success = False

    for epoch in range(args.epochs):
        if is_rank_zero:
            mllogger.start(
                key=mllog_constants.EPOCH_START,
                metadata={mllog_constants.EPOCH_NUM: epoch},
            )

        is_success = _train(
            dlrm_model,
            optimizer,
            train_dataloader,
            val_dataloader,
            epoch,
            lr_scheduler,
            args.print_lr,
            args.validation_freq_within_epoch,
            args.validation_auroc,
            args.limit_train_batches,
            args.limit_val_batches,
            args.print_progress,
            args.enable_profiling)
        if args.evaluate_on_epoch_end:
            val_auroc = _evaluate(
                dlrm_model,
                args.limit_val_batches,
                val_dataloader,
                "val",
                epoch + 1,
                False,
                args.print_progress,
            )
            results.val_aurocs.append(val_auroc)
        if is_rank_zero:
            mllogger.end(
                key=mllog_constants.EPOCH_STOP,
                metadata={mllog_constants.EPOCH_NUM: epoch},
            )
        if is_success:
            break

    ext_dist.barrier()
    if is_rank_zero and not is_success:
        # Run status "aborted" is reported in the case AUROC threshold is not met
        mllogger.end(
            key=mllog_constants.RUN_STOP,
            metadata={
                mllog_constants.STATUS: mllog_constants.ABORTED,
                mllog_constants.EPOCH_NUM: epoch + 1,
            },
        )

    if args.evaluate_on_training_end:
        test_auroc = _evaluate(
            dlrm_model,
            args.limit_test_batches,
            test_dataloader,
            "test",
            epoch + 1,
            False,
            args.print_progress,
        )
        results.test_auroc = test_auroc

    return results

def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """

    # The reference implementation does not clear the cache currently
    # but the submissions are required to do so
    mllogger.event(key=mllog_constants.CACHE_CLEAR, value=True)
    mllogger.start(key=mllog_constants.INIT_START)

    args = parse_args(argv)
    for name, val in vars(args).items():
        try:
            vars(args)[name] = list(map(int, val.split(",")))
        except (ValueError, AttributeError):
            pass

    if args.multi_hot_sizes is not None:
        assert (
            args.num_embeddings_per_feature is not None
            and len(args.multi_hot_sizes) == len(args.num_embeddings_per_feature)
            or args.num_embeddings_per_feature is None
            and len(args.multi_hot_sizes) == len(DEFAULT_CAT_NAMES)
        ), "--multi_hot_sizes must be a comma delimited list the same size as the number of embedding tables."
    assert (
        args.in_memory_binary_criteo_path is None
        or args.synthetic_multi_hot_criteo_path is None
    ), "--in_memory_binary_criteo_path and --synthetic_multi_hot_criteo_path are mutually exclusive CLI arguments."
    assert (
        args.multi_hot_sizes is None or args.synthetic_multi_hot_criteo_path is None
    ), "--multi_hot_sizes is used to convert 1-hot to multi-hot. It's inapplicable with --synthetic_multi_hot_criteo_path."
    assert (
        args.multi_hot_distribution_type is None
        or args.synthetic_multi_hot_criteo_path is None
    ), "--multi_hot_distribution_type is used to convert 1-hot to multi-hot. It's inapplicable with --synthetic_multi_hot_criteo_path."
    if args.synthetic_multi_hot_criteo_path is not None:
        assert(args.dense_labels_path is not None)

    device: torch.device = torch.device("cpu")
    backend = "ccl"
    ext_dist.init_distributed(backend=backend)

    is_rank_zero = ext_dist.is_rank_zero()
    if is_rank_zero:
        pprint(vars(args))
        submission_info(mllogger, "dlrm_dcnv2", "reference_implementation")
        mllogger.event(
            key=mllog_constants.GLOBAL_BATCH_SIZE,
            value=args.batch_size,
        )
        mllogger.event(
            key=mllog_constants.GRADIENT_ACCUMULATION_STEPS,
            value=1,  # Gradient accumulation is not supported in the reference implementation
        )
        mllogger.event(
            key=mllog_constants.SEED,
            value=args.seed,  # Seeding model is not supported in the reference implementation
        )
    # set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.num_embeddings_per_feature is not None:
        args.num_embeddings = None

    # Sets default limits for random dataloader iterations when left unspecified.
    if (
        args.in_memory_binary_criteo_path is None
        and args.synthetic_multi_hot_criteo_path is None
    ):
        for split in ["train", "val", "test"]:
            attr = f"limit_{split}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)
    assert(args.over_arch_layer_sizes is not None), "must provide over_arch_layer_sizes"
    assert(args.dense_arch_layer_sizes is not None), "must provide dense_arch_layer_sizes"

    multihot_enabled = False
    if args.multi_hot_sizes is None:
        multihot_enabled = True
        args.multi_hot_sizes = [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1]

    dlrm_model = DLRMMLPerf(
        embedding_dim=args.embedding_dim,
        num_embeddings_pool=args.num_embeddings_per_feature,
        multi_hot_sizes=args.multi_hot_sizes,
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=args.dense_arch_layer_sizes,
        over_arch_layer_sizes=args.over_arch_layer_sizes,
        dcn_num_layers=args.dcn_num_layers,
        dcn_low_rank_dim=args.dcn_low_rank_dim)

    model = dlrm_model

    def optimizer_with_params():
        if args.adagrad:
            return lambda params: MLPerfAdagrad(
                params,
                lr=args.learning_rate,
                lr_decay=ADAGRAD_LR_DECAY,
                weight_decay=WEIGHT_DECAY,
                initial_accumulator_value=ADAGRAD_INIT_ACC,
                eps=ADAGRAD_EPS,
            )
        else:
            return lambda params: torch.optim.SGD(
                params,
                lr=args.learning_rate,
                weight_decay=WEIGHT_DECAY,
            )

    optimizer = optimizer_with_params()(model.parameters())
    dlrm_model, optimizer = ipex.optimize(dlrm_model, optimizer=optimizer, dtype=torch.bfloat16, inplace=True)
    dlrm_model.sparse_arch.embedding_bag_collection.set_optimizer(optimizer)
    lr_scheduler = LRPolicyScheduler(
        optimizer, args.lr_warmup_steps, args.lr_decay_start, args.lr_decay_steps
    )
    if ext_dist.my_size > 1:
        dlrm_model.dense_arch = ext_dist.DDP(dlrm_model.dense_arch, gradient_as_bucket_view=True, broadcast_buffers=False) #, find_unused_parameters=True
        dlrm_model.inter_arch = ext_dist.DDP(dlrm_model.inter_arch, gradient_as_bucket_view=True, broadcast_buffers=False)
        dlrm_model.over_arch = ext_dist.DDP(dlrm_model.over_arch, gradient_as_bucket_view=True, broadcast_buffers=False)

    if is_rank_zero:
        mllogger.event(
            key=mllog_constants.OPT_NAME,
            value="adagrad" if args.adagrad else mllog_constants.SGD,
        )
        mllogger.event(
            key=mllog_constants.OPT_BASE_LR,
            value=args.learning_rate,
        )
        mllogger.event(
            #key="opt_adagrad_lr_decay",
            key="opt_adagrad_learning_rate_decay",
            value=ADAGRAD_LR_DECAY,
        )
        mllogger.event(
            key=mllog_constants.OPT_WEIGHT_DECAY,
            value=WEIGHT_DECAY,
        )
        mllogger.event(
            #key="opt_adagrad_init_acc_value",
            key="opt_adagrad_initial_accumulator_value",
            value=ADAGRAD_INIT_ACC,
        )
        mllogger.event(
            #key="opt_adagrad_eps",
            key="opt_adagrad_epsilon",
            value=ADAGRAD_EPS,
        )
        mllogger.event(
            key=mllog_constants.OPT_LR_WARMUP_STEPS,
            value=args.lr_warmup_steps,
        )
        mllogger.event(
            key=mllog_constants.OPT_LR_DECAY_START_STEP,
            value=args.lr_decay_start,
        )
        mllogger.event(
            key=mllog_constants.OPT_LR_DECAY_STEPS,
            value=args.lr_decay_steps,
        )

    ext_dist.barrier()
    if is_rank_zero:
        mllogger.start(key=mllog_constants.INIT_STOP)
    ext_dist.barrier()
    if is_rank_zero:
        mllogger.start(key=mllog_constants.RUN_START)
    ext_dist.barrier()

    train_dataloader = get_dataloader(args, backend, "train")
    val_dataloader = get_dataloader(args, backend, "val")
    test_dataloader = get_dataloader(args, backend, "test")
    if args.multi_hot_sizes is not None:
        multihot = Multihot(
            args.multi_hot_sizes,
            args.num_embeddings_per_feature,
            args.batch_size,
            collect_freqs_stats=args.collect_multi_hot_freqs_stats,
            dist_type=args.multi_hot_distribution_type,
        )
        multihot.pause_stats_collection_during_val_and_test(model)
        train_dataloader = RestartableMap(
            multihot.convert_to_multi_hot, train_dataloader
        )
        val_dataloader = RestartableMap(multihot.convert_to_multi_hot, val_dataloader)
        test_dataloader = RestartableMap(multihot.convert_to_multi_hot, test_dataloader)
    if is_rank_zero:
        mllogger.event(
            key=mllog_constants.TRAIN_SAMPLES,
            value=len(train_dataloader) * args.batch_size,
        )

    train_val_test(
        args,
        model,
        optimizer,
        device,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        lr_scheduler,
    )

if __name__ == "__main__":
    main(sys.argv[1:])
