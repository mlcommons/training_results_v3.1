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
import argparse
import os
import tempfile

import pandas as pd


def get_coco_compatible_name(image_id):
    """
    Generate a COCO compatible filename for a given image ID.

    This is necessary because captions should have the same names as COCO images.

    Args:
        image_id (int): The image ID.

    Returns:
        str: The COCO compatible filename.
    """
    name = f"COCO_val2014_{image_id:0>12}.txt"
    return name


def main(args):
    """
    Main method for loading captions from tsv and writing them as txt files.

    Args:
        args (argparse.Namespace): The command line arguments.

    Returns:
        str: The path to the output directory where the caption files are stored.
    """
    captions_df = pd.read_csv(args.captions_tsv, sep="\t")
    image_ids = captions_df["image_id"]
    captions = captions_df["caption"]

    if args.output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

    for image_id, caption in zip(image_ids, captions):
        image_name = get_coco_compatible_name(image_id)
        with open(os.path.join(output_dir, image_name), "w") as f:
            f.write(caption)
    print(output_dir)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-tsv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    main(args)
