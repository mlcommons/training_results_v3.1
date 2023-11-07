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

import math
from time import time
from tqdm import tqdm

import horovod.mxnet as hvd
from mxnet.contrib import amp
from mxnet import nd, autograd
from runtime.inference import evaluate
from runtime.distributed import sync_training_and_evaluation

from runtime.logging import sbridge
from mlperf_common.scaleoutbridge import ScaleoutBridgeBase as SBridge


def train(flags, model, train_loader, val_loader, score_fn, sw_inference,
          comm, eval_comm, transfer_comm, train_ranks, eval_ranks,
          transfer_ranks, ctx, callbacks, mllogger, run_start_time):
    rank = comm.Get_rank()
    stop_training = False
    converged = False
    diverged = False
    eval_warmup = flags.nodes_for_eval > 0

    if rank in train_ranks:
        train_size = hvd.size()
        samples_per_epoch = math.ceil(168 / ((train_size // flags.spatial_group_size) * flags.batch_size))
        samples_per_epoch = samples_per_epoch * flags.batch_size * (train_size // flags.spatial_group_size)
        mllogger.event(key='samples_per_epoch', value=samples_per_epoch, sync=False)

    for callback in callbacks:
        callback.on_fit_start()

    global_epoch = 1
    max_cycles = flags.epochs if flags.epochs < flags.evaluate_every else (flags.epochs // flags.evaluate_every + 1)

    sbridge.start_epoch_prof()

    for cycle in range(1, max_cycles):
        mllogger.start(key=mllogger.constants.BLOCK_START, sync=False,
                       metadata={mllogger.constants.FIRST_EPOCH_NUM: global_epoch,
                                 mllogger.constants.EPOCH_COUNT: flags.evaluate_every})
        for callback in callbacks:
            callback.on_cycle_start()

        if rank in train_ranks:
            cycle_start_time = time()

            for training_epoch in range(0, flags.evaluate_every):
                for i, batch in enumerate(tqdm(train_loader, disable=(rank != 0) or not flags.verbose)):
                    sbridge.start_prof(SBridge.ITER_TIME)

                    image, label = batch
                    if flags.static_cast:
                        image = image.astype(dtype='float16')

                    with autograd.record():
                        sbridge.start_prof(SBridge.FWD_TIME)

                        loss_value = model(image, label)

                        sbridge.stop_start_prof(SBridge.FWD_TIME, SBridge.BWD_TIME)

                        if flags.amp:
                            with amp.scale_loss(loss_value, model.trainer) as scaled_loss:
                                autograd.backward(scaled_loss)
                        elif flags.static_cast:
                            scaled_loss = loss_value * flags.static_loss_scale
                            autograd.backward(scaled_loss)
                        else:
                            loss_value.backward()

                        sbridge.stop_prof(SBridge.BWD_TIME)

                    sbridge.start_prof(SBridge.OPT_TIME)

                    model.trainer.step(image.shape[0] / flags.spatial_group_size)
                    loss_value.asnumpy()  # to prevent hang

                    sbridge.stop_prof(SBridge.OPT_TIME)
                    sbridge.stop_prof(SBridge.ITER_TIME)

            throughput = samples_per_epoch * flags.evaluate_every / (time() - cycle_start_time)
            loss_scale = flags.static_loss_scale if flags.static_cast else model.trainer._amp_loss_scaler.loss_scale
            mllogger.event(key='tracked_stats', metadata={'step': global_epoch}, sync=False,
                           value={"throughput": throughput, "loss_scale": loss_scale,
                                  'current_lr': model.trainer.learning_rate})
            if cycle in flags.loss_scale_inc_cycles and flags.static_cast:
                flags.static_loss_scale *= 2.0
                model.trainer._scale /= 2.0

        mllogger.end(key=mllogger.constants.BLOCK_STOP, sync=False,
                        metadata={mllogger.constants.FIRST_EPOCH_NUM: global_epoch,
                                mllogger.constants.EPOCH_COUNT: flags.evaluate_every})

        # Sync training and eval nodes
        global_epoch = cycle * flags.evaluate_every
        if (global_epoch >= flags.start_eval_at) and flags.nodes_for_eval:
            sbridge.start_eval_prof()
            stop_training, diverged, model = sync_training_and_evaluation(comm, transfer_comm, rank, model,
                                                                          eval_ranks, transfer_ranks,
                                                                          stop_training, diverged)
            sbridge.stop_eval_prof()

        if stop_training:
            break

        if rank in eval_ranks and eval_warmup:
            eval_metrics = evaluate(flags, model, val_loader, sw_inference, score_fn, ctx, eval_comm, global_epoch)
            eval_warmup = False

        if rank in eval_ranks and (global_epoch >= flags.start_eval_at):
            sbridge.start_eval_prof()
            mllogger.start(key=mllogger.constants.EVAL_START, value=global_epoch, sync=False,
                           unique_log_rank=eval_ranks[0], metadata={mllogger.constants.EPOCH_NUM: global_epoch})
            eval_metrics = evaluate(flags, model, val_loader, sw_inference, score_fn, ctx, eval_comm, global_epoch)
            mllogger.event(key=mllogger.constants.EVAL_ACCURACY, sync=False,
                           value=eval_metrics["mean_dice"], unique_log_rank=eval_ranks[0],
                           metadata={mllogger.constants.EPOCH_NUM: global_epoch})
            mllogger.end(key=mllogger.constants.EVAL_STOP, value=global_epoch, sync=False,
                         unique_log_rank=eval_ranks[0], metadata={mllogger.constants.EPOCH_NUM: global_epoch})
            sbridge.stop_eval_prof()

            if eval_metrics["mean_dice"] >= flags.quality_threshold:
                if not converged:
                    converged = True
                    mllogger.log_run_stop(status=mllogger.constants.SUCCESS, epoch=global_epoch,
                                          sync=False, unique_log_rank=eval_ranks[0])
            elif eval_metrics["mean_dice"] < 1e-4 and not converged:
                stop_training = True
                diverged = True
                mllogger.log_run_stop(status=mllogger.constants.ABORTED, epoch=global_epoch,
                                      sync=False, unique_log_rank=eval_ranks[0])

            if converged and (time() - run_start_time) / 60 > flags.sustained_training_time:
                stop_training = True
            else:
                if rank == eval_ranks[0]:
                    print(f"Training for {round((time() - run_start_time) / 60, 2)} min. "
                          f"Continuing till {flags.sustained_training_time} min.")

            for callback in callbacks:
                callback.on_cycle_end(epoch=global_epoch, metrics=eval_metrics, model=model)

    sbridge.stop_epoch_prof()

    for callback in callbacks:
        callback.on_fit_end(model=model)

    nd.waitall()

