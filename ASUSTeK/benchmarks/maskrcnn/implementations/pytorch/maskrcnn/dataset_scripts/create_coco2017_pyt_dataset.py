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

import os
import argparse
import pickle
import hashlib
import time
import torch
from maskrcnn_benchmark.data.datasets.coco import COCODataset

def create_tensor_based_input_data(instances, image_folder, include_targets, output_file_name):
    # Load coco dataset.
    # If targets are included, skip instances without valid targets.
    cd = COCODataset(instances, image_folder, include_targets)

    header_size = 32
    header = [0]*header_size
    header[0] = header_size
    header[2] = 1 if include_targets else 0
    header[3] = len(cd)
    print(len(header))

    per_image_mask_idx = [] # index of first mask for image
    per_mask_poly_idx = [] # index of first polygon for mask
    per_poly_sample_idx = [] # index of first sample for polygon
    dense_xy = [] # polygon samples, densely packed
    for i in range(len(cd)):
        target = cd.get_target(i)
        per_image_mask_idx.append(len(per_mask_poly_idx))
        #print("i ==%d :: per_image_mask_idx[-1] = %d" % (i, per_image_mask_idx[-1]))
        for mask in target.get_field('masks'):
            #print("mask=",mask)
            per_mask_poly_idx.append(len(per_poly_sample_idx))
            #print("i == %d :: per_mask_poly_idx[-1] = %d" % (i, per_mask_poly_idx[-1]))
            for poly in mask.polygons:
                #print("poly=",poly)
                per_poly_sample_idx.append(len(dense_xy))
                #print("i == %d :: per_poly_sample_idx[-1] = %d" % (i, per_poly_sample_idx[-1]))
                dense_xy.extend(poly.tolist())
    per_image_mask_idx.append(len(per_mask_poly_idx))
    per_mask_poly_idx.append(len(per_poly_sample_idx))
    per_poly_sample_idx.append(len(dense_xy))
    nn = len(header) + len(per_image_mask_idx)
    per_image_mask_idx = [i+nn for i in per_image_mask_idx]
    nn = nn + len(per_mask_poly_idx)
    per_mask_poly_idx = [i+nn for i in per_mask_poly_idx]

    # Cast dense_xy to fp32 by passing it through FloatTensor
    # Done to prevent small round-off errors, since values will be stored as FloatTensor
    dense_xy = torch.FloatTensor(dense_xy).tolist()

    # Determine if dense_xy values can be stored in look-up table.
    # This allows us to store values as int16 instead of fp32,
    # effectively halving the resulting file size.
    histogram = {}
    for v in dense_xy:
        if v in histogram:
            histogram[v] = histogram[v] + 1
        else:
            histogram[v] = 1

    output_files = []

    fname2 = "%sdense_xy.pyt" % (output_file_name)
    if len(histogram) < 65536:
        lut_size = len(histogram)
        header[1] = lut_size
        print("Using LUT with %d entries" % (lut_size))
        lut = list(histogram.keys())
        lut.sort()
        inverse_lut = {}
        for k, v in enumerate(lut):
            inverse_lut[v] = k
        lut = torch.FloatTensor(lut)
        fname1 = "%slut.pyt" % (output_file_name)
        torch.save(lut, fname1)
        output_files.append(fname1)
        dense_xy = [(int(inverse_lut[v])-32768) for v in dense_xy]
        dense_xy = torch.ShortTensor(dense_xy)
        torch.save(dense_xy, fname2)
        output_files.append(fname2)
    else:
        header[1] = 0
        dense_xy = torch.FloatTensor(dense_xy)
        torch.save(dense_xy, fname2)
        output_files.append(fname2)

    indexes = torch.IntTensor(header+per_image_mask_idx+per_mask_poly_idx+per_poly_sample_idx)
    fname3 = "%sindexes.pyt" % (output_file_name)
    torch.save(indexes, fname3)
    output_files.append(fname3)

    img_infos, bboxes = [], []
    for i in range(len(cd)):
        img_info = cd.get_img_info(i)
        target = cd.get_target(i)
        height, width, id, bbox_offset = img_info['height'], img_info['width'], img_info['id'], len(bboxes)
        img_infos.extend( [height, width, id, bbox_offset] )
        bboxes.extend( target.bbox.flatten().tolist() )
        bboxes.extend( target.get_field('labels').tolist() )
    # add dummy img_info tuple so reader can determine length of bbox tensor for last index
    img_infos.extend( [0,0,0,len(bboxes)] )
    fname4 = "%simg_info.pyt" % (output_file_name)
    fname5 = "%sbboxes_and_labels.pyt" % (output_file_name)
    img_infos = torch.IntTensor(img_infos)
    bboxes = torch.FloatTensor(bboxes)
    torch.save(img_infos, fname4)
    torch.save(bboxes, fname5)
    output_files.append(fname4)
    output_files.append(fname5)

    md5_originals = [
            "62b3e924bf826a26ff1be163d762715a",
            "100c8c4e1816c8b91326a21ff0438979",
            "7809fb431a58b10cb790f3ee432df662",
            "6bc80ba9d02ef0e85689c133c4f9fe25",
            "cf309a199221abdcae3f4f0d9139f449",
            ]
    for output_file, md5_orig in zip(output_files, md5_originals):
        # NB!
        # The output produced by torch.save includes the output filename,
        # so we can't verify correctness by comparing md5 checksums of
        # the output files. 
        # The workaround here is to load each tensor with torch.load,
        # then compute an md5 hash on the tensor data as a pickled list.
        to_check = torch.load(output_file)
        to_check = pickle.dumps(to_check.tolist())
        md5_returned = hashlib.md5(to_check).hexdigest()
        assert(md5_orig == md5_returned)

    print("\nThe following files were created:\n")
    os.system("ls -lh %s" % (" ".join(output_files)))

def main(args):
    output_folder = os.path.abspath(os.path.dirname(args.output))
    if not os.path.exists(output_folder):
        print("Creating output folder %s" % (output_folder))
        os.mkdir(output_folder)
    basename = os.path.basename(args.output)
    assert(len(basename) > 0)
    outputs = os.path.join(output_folder, basename)
    if os.path.exists("%simg_info.pyt" % (outputs)):
        print("Output files already exist in folder %s. Exiting" % (output_folder))
    else:
        create_tensor_based_input_data(args.ann_file, args.root, True, outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store annotation data as pytorch tensors")
    parser.add_argument("--root", help="detectron2 dataset directory", default="/coco")
    parser.add_argument("--ann_file", help="coco training annotation file path",
                            default="/coco/annotations/instances_train2017.json")
    parser.add_argument("--output", help="base name for output files",
                            default="/coco_train2017_pyt/coco_train2017_")
    args=parser.parse_args()
    main(args)

