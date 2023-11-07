# Instruction for GPT3 MLPerf workload

## 1. Problem

Large Language Model - GPT3 175B

### Requirements

*   [Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
*   [GKE (Google Kubernetes Engine) verson: 1.27.4-gke.900](https://cloud.google.com/kubernetes-engine)

## 2. Directions

### Environment

Create GKE cluster

```bash
export PROJECT_ID=${project_name} # use your own project id
export ZONE=${zone_name}

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}

CLUSTER_NAME=${USER}-mlperf

GKE_VERSION="1.27.4-gke.900"
gcloud beta container clusters create "$CLUSTER_NAME" \
    --zone "$ZONE" \
    --release-channel="${RELEASE_CHANNEL}" \
    --network=mtu9k \
    --subnetwork=mtu9k \
    --machine-type=${CLUSTER_MACHINE_TYPE} \
    --cluster-version=${GKE_VERSION} \
    --enable-autoscaling --max-nodes=100 --min-nodes=5

echo "cluster created:"
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone "${ZONE}"
gcloud config set container/cluster "${CLUSTER_NAME}"
```

Create DNS

```
sudo apt install jq -y
pushd ~
rm -rf deployment && git clone https://github.com/coredns/deployment.git
pushd deployment/kubernetes
./deploy.sh | kubectl apply -f -
kubectl scale deployment --replicas=0 kube-dns-autoscaler --namespace=kube-system
kubectl scale deployment --replicas=0 kube-dns --namespace=kube-system
popd
popd
```

Create TPU node

```
MACHINE_TYPE=ct5lp-hightpu-4t  # recommended machine for tpu v5e
TPU_TOPOLOGY=16x16
NUM_CHIPS=$(( $(echo $TPU_TOPOLOGY |  sed 's/x/*/g') ))
NUM_CHIPS_PER_NODE=4
NUM_NODES=$(( $NUM_CHIPS / $NUM_CHIPS_PER_NODE ))
MAINTENANCE_INTERVAL="PERIODIC"
GKE_VERSION="1.27.4-gke.900"

NUM_SLICES=16

for i in $(seq 0 $((NUM_SLICES - 1))); do
  gcloud beta container node-pools create "${CLUSTER_NAME}-${i}" \
    --zone="${ZONE}" \
    --node-version="${GKE_VERSION}" \
    --cluster="${CLUSTER_NAME}" \
    --node-locations="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --tpu-topology="${TPU_TOPOLOGY}" \
    --num-nodes="${NUM_NODES}" \
    --placement-type=COMPACT \
    --host-maintenance-interval="${MAINTENANCE_INTERVAL}" \
    --enable-gvnic \
    --scopes=storage-full,gke-default \
    --disk-size=50 \
    --max-pods-per-node=24
done
```

all these setup steps can be found under setup.sh

### Steps to launch training

Launch configuration and system-specific hyperparameters for the appropriate GKE
TPU submission are in the `jobset_16_v5_256.yaml` scripts.

```
####1.  Launch the training
```
./run_and_time.sh
```

Please run all prerequisite in `./setup.sh` first before running
`./run_and_time.sh`.

####2. E2E training launch (including both steps above)

```
ENABLE_LOCAL_AQT=true ENABLE_CHECKPOINT=true CLUSTER_NAME=${CLUSTER_NAME} JAX_LIBTPU_IMAGE=gcr.io/cloud-tpu-v2-images/pax-jax-libtpu:2023_09_14_final_sanitized TPU_TOPOLOGY=16x16 EXP_NAME=C4SpmdGpt3AdamDataParallel16x16x16Int8 NUM_SLICES=16 bash scripts/run.sub
```

## 3. Dataset

Please refer to the
[instructions](https://github.com/mlcommons/training/blob/master/large_language_model/paxml/README.md)
from the reference to download the dataset.

The C4 dataset location: `gs://mlperf-llm-public2/c4`

The tokenizer location as the SPM variable are: `gs://mlperf-llm-public2/vocab`.

## 4. Model

The model largely follows the GPT-3 paper, with key model architecture configs
refer
[here](https://github.com/mlcommons/training/blob/master/large_language_model/paxml/README.md#3-model)

### List of Layers

The model largely follows the GPT3 [paper](https://arxiv.org/abs/2005.14165),
refer
[here](https://github.com/mlcommons/training/blob/master/large_language_model/paxml/README.md#3-model)
for model details.

### Model checkpoint

In the benchmarking region, we resume training from a reference checkpoint which
is trained with Global Batch Size of 1536 for 4000 iterations.

To resume training, firstly the checkpoint needs to be converted from the Paxml
reference checkpoint using `tools/convert_checkpoint.sh`.

## 5. Quality

### Quality metric

Log Perplexity

### Quality target

2.69

### Evaluation frequency

Evaluate after every 24576 samples (=50.33B tokens)

### Evaluation thoroughness

Evaluation on the validation subset that consists of 24567 examples.

## 6. Additional notes

postproces for MLLOG from raw run

```
cat ${job_dir}/large_scale_multislice_test_log_0_0 | uniq | grep MLLOG  > ${job_dir}/result_0.txt
```
