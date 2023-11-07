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
import copy
from contextlib import redirect_stdout

import numpy as np

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

import utils
from coco.coco_eval import CocoEvaluator


class PythonCocoEvaluator(CocoEvaluator):
    def __init__(self, annotations_file, iou_types, group=None):
        assert isinstance(iou_types, (list, tuple))
        self.annotations_file = annotations_file
        self.coco_gt = PythonCocoEvaluator.get_coco_gt(annotations_file=annotations_file)

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(self.coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(None):
                coco_dt = self.coco_gt.loadRes(results)
            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self, group=None):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type], group=group)

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

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
    def get_coco_gt(annotations_file=None):
        return COCO(annotation_file=annotations_file)

    @staticmethod
    def get_stats_from_evaluator(evaluator):
        evaluator.accumulate()
        evaluator.summarize()
        return evaluator.get_stats()


def merge(img_ids, eval_imgs, group=None):
    all_img_ids = utils.all_gather(img_ids, group=group)
    all_eval_imgs = utils.all_gather(eval_imgs, group=group)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs, group=None):
    img_ids, eval_imgs = merge(img_ids, eval_imgs, group=group)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


def evaluate(imgs):
    with redirect_stdout(None):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))
