#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import string
import random
import json
import os
import tarfile


#Step1:
#Extraction logic

input_path = os.environ.get('DATASET_PATH_UNCOMPRESSED')
output_path = os.environ.get('DATASET_PATH_OUTPUT')
# at the end of the run perform a tar on the files inside the directory ( not on the directory itself):
#  tar -cvf output_file.tar *

if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Directory '{output_path}' created successfully.")
else:
    print(f"Directory '{output_path}' already exists.")

all_files = os.listdir(input_path)

# Filter for files with the .npy suffix
npy_files = [file for file in all_files if file.endswith('.npy')]

index =1
num_of_files = len(npy_files)
for file in npy_files:
    file_name = file[:-4]
    print(f"{file_name} is {index} out of {num_of_files} files \n")
    index = index+1
    input_file_path=input_path+'/'+ file_name
    output_file_path = output_path+'/'+file_name

    # Modify the json file
    try:
        input_json_file_path = input_file_path + '.json'
        output_json_file_path = output_file_path + '.json'

        caption_length  = random.randint(5, 20)
        with open(input_json_file_path, 'r') as input_json_file:
            data = json.load(input_json_file)
            data['url'] = 'synthetic_data'
            data['caption']= ' '.join(random.choice(string.ascii_letters + string.digits) for _ in range(caption_length))
        with open(output_json_file_path, 'w') as output_json_file:
            json.dump(data, output_json_file)

        input_json_file.close()
        output_json_file.close()

    except Exception as e:
        print(f"An error occurred: {e}")

    # Modify the output txt file
    try:
        output_txt_file_path = output_file_path + '.txt'

        with open(output_txt_file_path, 'w') as text_file:
            # Write the string to the file
            text_file.write(data['caption'])

    except Exception as e:
        print(f"An error occurred: {e}")

    # Modify the numpy file
    try:
        input_npy_file_path = input_file_path+'.npy'
        output_npy_file_path = output_file_path+'.npy'
        data = np.load(input_npy_file_path)
        random_array = np.random.randn(*data.shape).astype(np.float32)
        np.save(output_npy_file_path, random_array) 

    except Exception as e:
        print(f"An error occurred: {e}")

