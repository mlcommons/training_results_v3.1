# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import os
import pandas as pd
import argparse
from mlperf_logging_utils import mllogger, constants, extract_timestamp_from_ckpt_name, extract_step_from_ckpt_name
from common import simple_init_distributed

TARGET_FID = 90.0
TARGET_CLIP = 0.15


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fid", type=str, required=True)
    parser.add_argument("--clip", type=str, required=True)
    args = parser.parse_args()
    return args


def print0(*args, **kwds):
    if int(os.environ["SLURM_PROCID"]) == 0:
        print(*args, **kwds)


def main():
    args = parse_command_line_args()
    strategy = simple_init_distributed()

    fid = pd.read_csv(args.fid)
    clip = pd.read_csv(args.clip)

    # For missing FID and CLIP fields
    # We fill FID with a large value (because the target is <=90)
    # We fill CLIP with a small value (because the target is >=0.15)

    fid.fillna(999, inplace=True)
    clip.fillna(0, inplace=True)

    timestamp_to_metrics = {}
    timestamp_to_step = {}

    for row in fid.itertuples():
        step = row.step
        fid_score = row.fid
        timestamp = row.timestamp

        timestamp_to_metrics.setdefault(timestamp, {})
        timestamp_to_metrics[timestamp]["FID"] = fid_score
        timestamp_to_step[timestamp] = step

    for row in clip.itertuples():
        step = row.step
        clip_score = row.clip
        timestamp = row.timestamp

        timestamp_to_metrics.setdefault(timestamp, {})
        timestamp_to_metrics[timestamp]["CLIP"] = clip_score

    for timestamp in sorted(timestamp_to_metrics.keys()):
        step = timestamp_to_step[timestamp]
        print0(f"step {step} ts {timestamp} :: {timestamp_to_metrics[timestamp]}")

    # Browse throught the checkpoints from the earliest to the latest
    # Report the timestamp of the earliest checkpoint that reaches the target metrics

    success_timestamp = None
    for timestamp in sorted(timestamp_to_metrics.keys()):
        metrics = timestamp_to_metrics[timestamp]
        step = timestamp_to_step[timestamp]

        if metrics["FID"] <= TARGET_FID and metrics["CLIP"] >= TARGET_CLIP:
            success_timestamp = timestamp
            print0(f"Found checkpoint with {metrics} at step {step} ts {timestamp}")
            mllogger.event(
                key=constants.EVAL_ACCURACY,
                value=metrics["FID"],
                metadata={
                    constants.STEP_NUM: step,
                    "metric": "FID",
                },
                time_ms=success_timestamp,
            )
            mllogger.event(
                key=constants.EVAL_ACCURACY,
                value=metrics["CLIP"],
                metadata={
                    constants.STEP_NUM: step,
                    "metric": "CLIP",
                },
                time_ms=success_timestamp,
            )
            break
    else:
        print0(f"Could not find checkpoint matching targets")
    print0(f"Targets are FID={TARGET_FID} and CLIP={TARGET_CLIP}")

    if success_timestamp is not None:
        status = constants.SUCCESS
        step = timestamp_to_step[success_timestamp]
        mllogger.end(
            key=constants.RUN_STOP,
            unique=True,
            sync=True,
            metadata={
                "status": status,
                constants.STEP_NUM: step,
            },
            internal_call=True,
            time_ms=success_timestamp,
        )

    else:
        status = constants.ABORTED
        mllogger.log_run_stop(status=status)


if __name__ == "__main__":
    main()
