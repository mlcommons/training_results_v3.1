import torch
from .logger import logging
from maskrcnn_benchmark.utils.comm import get_rank, get_world_size

def per_gpu_batch_size(cfg, log_info=False):
    rank = get_rank()
    world_size = get_world_size()

    dedicated_evaluation_ranks = max(0,cfg.DEDICATED_EVALUATION_RANKS)
    num_training_ranks = world_size - dedicated_evaluation_ranks
    num_evaluation_ranks = world_size if dedicated_evaluation_ranks == 0 else dedicated_evaluation_ranks

    images_per_batch_train = cfg.SOLVER.IMS_PER_BATCH
    images_per_batch_test = cfg.TEST.IMS_PER_BATCH

    if images_per_batch_train >= num_training_ranks:
        spatial_group_size_train = 1
    else:
        spatial_group_size_train = num_training_ranks // images_per_batch_train
        assert(spatial_group_size_train * images_per_batch_train == num_training_ranks), "Number of training ranks is not a multiple of global batch size"
        if log_info:
            logger = logging.getLogger('maskrcnn_benchmark.trainer')
            logger.info("Enabled spatial parallelism for trainer with group_size = %d" % (spatial_group_size_train))
    images_per_gpu_train = (images_per_batch_train * spatial_group_size_train) // num_training_ranks
    num_training_ranks = num_training_ranks // spatial_group_size_train
    if rank >= num_training_ranks * spatial_group_size_train:
        rank_train = -1
        rank_in_group_train = -1
    else:
        rank_train = rank // spatial_group_size_train
        rank_in_group_train = rank % spatial_group_size_train

    if images_per_batch_test >= num_evaluation_ranks:
        spatial_group_size_test = 1
    else:
        spatial_group_size_test = num_evaluation_ranks // images_per_batch_test
        assert(spatial_group_size_test * images_per_batch_test == num_evaluation_ranks), "Number of evaluation ranks is not a multiple of global batch size"
        if log_info:
            logger = logging.getLogger('maskrcnn_benchmark.tester')
            logger.info("Enabled spatial parallelism for tester with group_size = %d" % (spatial_group_size_test))
    images_per_gpu_test = (images_per_batch_test * spatial_group_size_test) // num_evaluation_ranks
    num_evaluation_ranks = num_evaluation_ranks // spatial_group_size_test
    if dedicated_evaluation_ranks > 0: 
        # using dedicated evaluation ranks
        if rank < num_training_ranks * spatial_group_size_train:
            # training rank
            rank_test = -1
            rank_in_group_test = -1
        else:
            # evaluation rank
            rank_test = (rank - num_training_ranks * spatial_group_size_train) // spatial_group_size_test
            rank_in_group_test = (rank - num_training_ranks * spatial_group_size_train) % spatial_group_size_test
    else:
        # not using dedicated evaluation ranks
        rank_test = rank // spatial_group_size_test
        rank_in_group_test = rank % spatial_group_size_test

    #print("%d :: dedicated_evaluation_ranks=%d, num_training_ranks=%d, images_per_batch_train=%d, images_per_gpu_train=%d, rank_train=%d, rank_in_group_train=%d, spatial_group_size_train=%d, num_evaluation_ranks=%d, images_per_batch_test=%d, images_per_gpu_test=%d, rank_test=%d, rank_in_group_test=%d, spatial_group_size_test=%d" % (get_rank(), dedicated_evaluation_ranks, num_training_ranks, images_per_batch_train, images_per_gpu_train, rank_train, rank_in_group_train, spatial_group_size_train, num_evaluation_ranks, images_per_batch_test, images_per_gpu_test, rank_test, rank_in_group_test, spatial_group_size_test))
    return dedicated_evaluation_ranks, num_training_ranks, images_per_batch_train, images_per_gpu_train, rank_train, rank_in_group_train, spatial_group_size_train, num_evaluation_ranks, images_per_batch_test, images_per_gpu_test, rank_test, rank_in_group_test, spatial_group_size_test
