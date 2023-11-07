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

import functools

import torch
import numpy as np

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

import utils
from coco.coco_eval import CocoEvaluator


class NVCocoEvaluator(CocoEvaluator):
    def __init__(self, annotations_file, iou_types, num_threads=1, group=None):
        assert isinstance(iou_types, (list, tuple))
        self.annotations_file = annotations_file
        self.coco_gt = None
        self.num_threads = num_threads
        self.multi_procs = (1, None)  # TODO(ahmadki): support nvcoco multi processing
        self.iou_types = iou_types
        self.coco_eval = {}
        self._results = {}
        for iou_type in iou_types:
            self._results[iou_type] = []

    @property
    def results(self):
        return self._results

    def update(self, predictions):
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            results = np.array(results, dtype=np.float32)
            self._results[iou_type].extend(results)

    def synchronize_between_processes(self, group=None):
        for iou_type in self.iou_types:
            self._results[iou_type] = np.vstack(self._results[iou_type])
            self._results[iou_type] = utils.all_gather(self._results[iou_type], group=group)
            self._results[iou_type] = np.concatenate(self._results[iou_type], axis=0)

    def accumulate(self):
        if self.coco_gt is None:
            self.coco_gt = NVCocoEvaluator.get_coco_gt(annotations_file=self.annotations_file,
                                                       num_threads=self.num_threads)

        for iou_type in self.iou_types:
            coco_dt = self.coco_gt.loadRes(self._results[iou_type])
            coco_eval = COCOeval(self.coco_gt,
                                 coco_dt,
                                 iouType=iou_type,
                                 num_threads=self.num_threads,
                                 multi_procs=self.multi_procs,
                                 use_ext=True)
            coco_eval.evaluate()
            coco_eval.accumulate()
            self.coco_eval[iou_type] = coco_eval

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def get_stats(self):
        stats = {}
        for iou_type, coco_eval in self.coco_eval.items():
            stats[iou_type] = coco_eval.stats
        return stats

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_coco_gt(annotations_file=None, num_threads=8):
        multi_procs = (1, None)  # TODO(ahmadki)
        return COCO(annotation_file=annotations_file,
                    multi_procs=multi_procs,
                    num_threads=num_threads,
                    use_ext=True)

    @staticmethod
    def get_stats_from_evaluator(evaluator):
        evaluator.accumulate()
        evaluator.summarize()
        return evaluator.get_stats()

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            for (bbox, label, score) in zip(boxes, labels, scores):
                coco_results.extend([[original_id, bbox[0], bbox[1], bbox[2], bbox[3], score, label]])
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
