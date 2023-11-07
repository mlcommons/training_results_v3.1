# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
import logging
import tempfile
import pickle
import time
import os
import torch
from collections import OrderedDict
from multiprocessing import Pool
from functools import partial
import itertools
import pycocotools.mask as mask_util
import numpy as np
import ujson

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.comm import is_main_process, all_gather, get_world_size, get_rank
from maskrcnn_benchmark.utils.async_evaluator import get_evaluator, get_tag
from maskrcnn_benchmark.utils.timed_section import TimedSection

def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, dedicated_evaluation_ranks, eval_ranks_comm):
    world_size = get_world_size() if dedicated_evaluation_ranks == 0 else dedicated_evaluation_ranks
    all_predictions = all_gather(predictions_per_gpu, world_size=world_size, group=eval_ranks_comm)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions

#
# Remark by Thor Johnsen (01/28/2023)
#
# Preparing inference results for evaluation involves three things:
#
# 1. Gather results onto master rank (a single rank runs coco evaluation code)
# 2. Eliminate duplicates
# 3. Format results
#
# The easiest way to eliminate duplicates is to serialize all the result objects
# before removing duplicates using a string set. Since coco evaluation routines expect
# input in json format, we serialize with json for both steps 2 and 3. Json dumps is
# expensive, to alleviate this we do json dumps of all the invidual result objects
# before gathering results. Individual json strings are concatenated into a single large
# string which is then copied into a tensor and gathered on master rank with torch.distributed.
# Rather than putting all the individual json strings in a list and then doing json.dumps
# on the list, we concatenate with string join, which is orders of magnitude faster.
# For this approach to work, the join string has to be unique, i.e. it must never occur
# in any of the serialized data. This is hard to guarantee because the serialized data
# includes a serialized byte buffer, as a trade-off we use a long join string,
# which is exceedingly unlikely to occur in the serialized byte buffer. The join string
# has 10 characters which are encoded as bytes. It has a likelihoof of occuring
# in the serialized byte buffer of 1 / 255^10, i.e. less than 10^-24 chance.
# If the join string does occur in any of the serialized byte buffers,
# evaluation will fail and the job will fail to converge. Given up to 20% of the jobs
# are allowed to fail, using join with the selected join string increases likelihood
# of job failing to converge by a negligible amount, thus this is a good trade off.
#
# Evaluation steps are thus:
#
# 1. All evaluation ranks do inference.
# 2. All evaluation ranks format inference results.
# 3. All evaluation ranks do json.dumps on individual inference results.
# 4. All evaluation ranks concatenate resulting json strings with string join.
# 5. Concatenated json strings are gathered on evaluation master rank.
# 6. Evaluation master rank splits concatenated json strings.
# 7. Evaluation master rank removes duplicate strings with string set.
# 8. Evaluation master rank unpacks remaining individual results with json.loads.
# 9. Evaluation master rank calls coco evaluation code for each iou type.
#
# For maskrcnn there are two iou types, "bbox" and "segm".
#

# Remove any duplicates in a list
def remove_dup(l):
    seen = set()
    seen_add = seen.add
    return [x for x in l if not (x in seen or seen_add(x))]

def remove_dup_and_unpack(temp_list):
    coco_results = [buffer.numpy().tobytes().decode('utf-8').split("!@#$%^&*()") for buffer in temp_list]
    coco_results = [i for j in coco_results for i in j]
    coco_results = remove_dup(coco_results)
    return coco_results

def evaluate_single_iou(output_folder, coco, iou_type, coco_results):
    with tempfile.NamedTemporaryFile() as f:
        file_path = f.name
        if output_folder:
            file_path = os.path.join(output_folder, iou_type + ".json")
        res = evaluate_predictions_on_coco(
            coco, coco_results, file_path, iou_type
        )
        return res

def evaluate_coco(coco, coco_results, iou_types, output_folder):
    with TimedSection("Evaluating predictions took %.3fs"):
        # Unpack the gathered results into a single List[Entry]
        results = COCOResults(*iou_types)
        from multiprocessing import Pool
        from functools import partial
        if "segm" in iou_types and len(iou_types) > 1:
            pool = Pool(1)
            r_segm = pool.apply_async(remove_dup_and_unpack, (coco_results["segm"],))
            for iou_type in iou_types:
                if iou_type != "segm":
                    results.update( evaluate_single_iou(output_folder, coco, iou_type, remove_dup_and_unpack(coco_results[iou_type])) )
            results.update( evaluate_single_iou(output_folder, coco, "segm", r_segm.get()) )
        else:
            for iou_type in iou_types:
                results.update( evaluate_single_iou(output_folder, coco, iou_type, remove_dup_and_unpack(coco_results[iou_type])) )
        return results

def do_coco_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    eval_segm_numprocs,
    eval_mask_virtual_paste,
    dedicated_evaluation_ranks,
    eval_ranks_comm,
):
    logger = logging.getLogger("maskrcnn_benchmark.inference")

    with TimedSection("EXPOSED: Launching evaluation preparation tasks took %.3fs"):
        # Different path here, fast parallel method not available, fall back to effectively the old
        # path.
        if box_only:
            predictions = _accumulate_predictions_from_multiple_gpus(predictions, dedicated_evaluation_ranks, eval_ranks_comm)
            if not is_main_process():
                return

            logger.info("Evaluating bbox proposals")
            areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
            res = COCOResults("box_proposal")
            for limit in [100, 1000]:
                for area, suffix in areas.items():
                    stats = evaluate_box_proposals(
                        predictions, dataset, area=area, limit=limit
                    )
                    key = "AR{}@{:d}".format(suffix, limit)
                    res.results["box_proposal"][key] = stats["ar"].item()
            logger.info(res)
            check_expected_results(res, expected_results, expected_results_sigma_tol)
            if output_folder:
                torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
            return

        eval_prep_args = []
        for image_id, prediction in predictions.items():
            original_id = dataset.id_to_img_map[image_id]
            if len(prediction) == 0:
                continue

            img_info = dataset.get_img_info(image_id)
            image_width = img_info["width"]
            image_height = img_info["height"]
            prediction = prediction.resize((image_width, image_height))
            #prediction = prediction.convert("xywh")

            labels = prediction.get_field("labels").tolist()
            mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]

            eval_prep_args.append( (original_id, image_width, image_height, mapped_labels, prediction,) )

        get_evaluator().submit_task(get_tag(),
                                    prepare_for_evaluation,
                                    iou_types,
                                    eval_prep_args,
                                    dataset.coco,
                                    eval_segm_numprocs,
                                    eval_mask_virtual_paste,
                                    output_folder)

    # Note: results is now empty, the relevant future is held in the hidden
    # AsyncEvaluator object
    return None, {}


def prepare_for_evaluation(iou_types, eval_prep_args, coco, eval_segm_numprocs, eval_mask_virtual_paste, output_folder):

    coco_results = {}
    with TimedSection("Preparing for evaluation took %.3f seconds total"):
        if "segm" in iou_types:
            r = launch_prepare_for_coco_segmentation(eval_prep_args, eval_segm_numprocs, eval_mask_virtual_paste)
        if "bbox" in iou_types:
            coco_results["bbox"] = prepare_for_coco_detection(eval_prep_args)
        if 'keypoints' in iou_types:
            coco_results['keypoints'] = prepare_for_coco_keypoint(eval_prep_args)
        if "segm" in iou_types:
            coco_results["segm"] = get_prepare_for_coco_segmentation(r)
        return coco_results, iou_types, coco, output_folder


def all_gather_prep_work(coco_results, dedicated_evaluation_ranks, eval_ranks_comm):
    with TimedSection("All-gathering preparation work took %.3fs"):
        world_size = get_world_size() if dedicated_evaluation_ranks == 0 else dedicated_evaluation_ranks
        if "segm" in coco_results: coco_results["segm"] = all_gather(coco_results["segm"], world_size=world_size, group=eval_ranks_comm)
        if "bbox" in coco_results: coco_results["bbox"] = all_gather(coco_results["bbox"], world_size=world_size, group=eval_ranks_comm)
        if "keypoints" in coco_results: coco_results["keypoints"] = all_gather(coco_results["keypoints"], world_size=world_size, group=eval_ranks_comm)
        return coco_results


def prepare_for_coco_detection(eval_prep_args):
    coco_results = []
    for original_id, image_width, image_height, mapped_labels, prediction in eval_prep_args:
        prediction = prediction.convert("xywh")

        scores = prediction.get_field("scores").tolist()
        boxes = prediction.bbox.tolist()

        coco_results.extend(
            [
                ujson.dumps(
                    {
                        "image_id": original_id,
                        "category_id": mapped_labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                )
                for k, box in enumerate(boxes)
            ]
        )
    return "!@#$%^&*()".join(coco_results).encode('utf-8')


def single_sample_prepare_for_coco_segmentation(masker, eval_mask_virtual_paste, original_id, image_width, image_height, mapped_labels, prediction):
    scores = prediction.get_field("scores").tolist()

    masks = prediction.get_field("mask")
    # Masker is necessary only if masks haven't been already resized.
    if list(masks.shape[-2:]) != [image_height, image_width]:
        masks = masker(masks.expand(1, -1, -1, -1, -1), prediction, paste=not eval_mask_virtual_paste)
        masks = masks[0]

        if eval_mask_virtual_paste:
            rles = []
            for y0, x0, im_h, im_w, boxed_mask in masks:
                #print("y0=%d, x0=%d, im_h=%d, im_w=%d, boxed_mask.size()=%s" % (y0, x0, im_h, im_w, str(boxed_mask.size())))
                c = np.array(boxed_mask[ :, :, np.newaxis], order="F")
                rles.append( mask_util.encode(c, paste_args=dict(oy=y0, ox=x0, oh=im_h, ow=im_w))[0] )
        else:
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]
    else:
        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("utf-8")

    coco_results = [
        ujson.dumps(
            {
                "image_id": original_id,
                "category_id": mapped_labels[k],
                "segmentation": rle,
                "score": scores[k],
            }
        )
        for k, rle in enumerate(rles)
    ]
    return coco_results


def launch_prepare_for_coco_segmentation(eval_prep_args, eval_segm_numprocs, eval_mask_virtual_paste):
    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)

    pool = Pool(eval_segm_numprocs)
    r = pool.starmap_async(partial(single_sample_prepare_for_coco_segmentation, masker, eval_mask_virtual_paste), eval_prep_args)
    return r

def get_prepare_for_coco_segmentation(r):
    coco_results = r.get()
    coco_results = list(itertools.chain(*coco_results))

    return "!@#$%^&*()".join(coco_results).encode('utf-8')


def prepare_for_coco_keypoint(eval_prep_args):
    coco_results = []
    for original_id, image_width, image_height, mapped_labels, prediction in eval_prep_args:
        prediction = prediction.convert('xywh')

        scores = prediction.get_field('scores').tolist()

        boxes = prediction.bbox.tolist()
        keypoints = prediction.get_field('keypoints')
        keypoints = keypoints.resize((image_width, image_height))
        keypoints = keypoints.keypoints.view(keypoints.keypoints.shape[0], -1).tolist()

        coco_results.extend(
            [
                ujson.dumps(
                    {
                        'image_id': original_id,
                        'category_id': mapped_labels[k],
                        'keypoints': keypoint,
                        'score': scores[k]
                    }
                    for k, keypoint in enumerate(keypoints)
                )
            ]
        )
    return "!@#$%^&*()".join(coco_results).encode('utf-8')

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, area="all", limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("objectness").sort(descending=True)[1]
        prediction = prediction[inds]

        ann_ids = dataset.coco.getAnnIds(imgIds=original_id)
        anno = dataset.coco.loadAnns(ann_ids)
        gt_boxes = [obj["bbox"] for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_coco(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    with TimedSection("Wrote json_result_file in %.3fs"):
        with open(json_result_file, "w") as f:
            f.write('[' + ','.join(coco_results) + ']')

    with TimedSection("Evaluating '%s' predictions on COCO took %%.3fs" % (iou_type)):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_dt = coco_gt.loadRes(str(json_result_file), use_ext=True) if coco_results else COCO()

        # coco_dt = coco_gt.loadRes(coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, iou_type, use_ext=True, num_threads=8)
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm", "keypoints")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)
