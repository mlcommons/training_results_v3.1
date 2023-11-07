# Download and prepare the data

Building the Docker container
```shell
docker build --pull -t <docker/registry>/mlperf-nvidia:language_model .
docker push <docker/registry>/mlperf-nvidia:language_model
```

Go through standard data preparation upt to the moment where you have unpacked "results4" sources

Assuming /data/mlperf/bert/ contains 'download/results4' and 'phase1' directories

Start the container interactively, mounting the directory you want to store the expieriment data as `/workspace/bert_data`
```
docker run -it --runtime=nvidia --ipc=host (...) -v /data/mlperf/bert:/workspace/bert_data mlperf-nvidia:language_model
```

To prepare the packed version of data, we need first to group the trainig sequences be lenght (as number of valid tokens). To easily parallelize the process each shard is processed separately and at the end results are merged.
```
mkdir -p /workspace/bert_data/per_seqlen_parts
for shard in `seq -w 00000 00499`; do
    mkdir -p /workspace/bert_data/per_seqlen_parts/part-${shard}
done
```

Parallelize over $CPUS cores
```
CPUS=64
seq -w 00000 00499 | xargs --max-args=1 --max-procs=$CPUS -I{} python create_per_seqlength_data.py --input_file /workspace/bert_data/download/results4/part-{}-of-00500 --output_file /workspace/bert_data/per_seqlen_parts/part-{} --vocab_file /workspace/bert_data/phase1/vocab.txt --do_lower_case=True    --max_seq_length=512    --max_predictions_per_seq=76    --masked_lm_prob=0.15    --random_seed=12345    --dupe_factor=10
```

Merge all results
```
mkdir -p /workspace/bert_data/per_seqlen
seq 1 512 | xargs --max-args=1 --max-procs=$CPUS -I{} python ./gather_per_seqlength_data.py --input_hdf5 /workspace/bert_data/per_seqlen_parts --output_hdf5 /workspace/bert_data/per_seqlen --seq_length {}
```

Generate sub-optimal packing strategy based on lenghts distribution of training set and store samples-based lists per shard
```
mkdir -p /workspace/bert_data/packed_data
python ./generate_packing_strategy.py --input_hdf5 /workspace/bert_data/per_seqlen --output_hdf5 /workspace/bert_data/packed_data --max_seq_length 512 --max_seq_per_sample 3 --shards_num 4320 
```
This step should result in generation of packing strategy and report similar packing statistics:

> _________________________Done solving for packing order_________________________  
> Number of sequences dropped:  0.0  
> Number of strategies utilized: 1129  
> New number of samples: 78404631.00, original 156726829. A compression ratio of 0.500  
> The expected speed-up from packing: 1.999  
> Theoretical upper bound on speed-up: 2.012  
> Average sequences/sample 1.99895  
> Added 2.67e+08 padding tokens. Original dataset used 4.04e+10 padding tokens  

Create training set shards based on generated lists
```
python create_packed_trainset.py --input_hdf5 /workspace/bert_data/per_seqlen --assignment_file /workspace/bert_data/packed_data --output_hdf5 /workspace/bert_data/packed_data
```
