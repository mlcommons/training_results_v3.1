#!/bin/bash
set -eox pipefail

SCRIPTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" && pwd )"

# need to setup project/zone before proceed
export PROJECT_ID=${PROJECT_ID:-mlperf-high-priority-project}  # use your own project id
export REGION=${region} # use your own project region
export ZONE=${zone}  # use your own project zone

gcloud config set project "${PROJECT_ID}"
gcloud config set compute/zone "${ZONE}"

export ACCELERATOR=tpu-v5-lite-podslice
export CLUSTER_MACHINE_TYPE=e2-standard-4
export MACHINE_TYPE=ct5lp-hightpu-4t  # recommended machine for tpu v5
export TPU_TOPOLOGY=${TPU_TOPOLOGY:-16x16}
export GKE_VERSION="1.27.4-gke.900"
export RELEASE_CHANNEL=rapid

unset CLUSTER_NAME
CLUSTER_NAME=${CLUSTER_NAME:-mlperf-cluster-1}
EXP_NAME=${EXP_NAME:-C4SpmdGpt3AdamDataParallel16x16x16Int8}

export USE_EXISTING_TPUS=${USE_EXISTING_TPUS:-true}

if [[ "${USE_EXISTING_TPUS}" = false ]]; then
  # the following command creates a new GKE regional cluster subscribed to the rapid release channel and with a Kubernetes node pool that initially contains one node per zone
  retry gcloud beta container clusters create "$CLUSTER_NAME" \
    --zone="${ZONE}" \
    --release-channel="${RELEASE_CHANNEL}" \
    --network=mtu9k \
    --subnetwork=mtu9k \
    --machine-type=${CLUSTER_MACHINE_TYPE} \
    --cluster-version=${GKE_VERSION} \
    --enable-autoscaling --max-nodes=100 --min-nodes=5
fi


echo "cluster created:"
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone "${ZONE}"
gcloud config set container/cluster "${CLUSTER_NAME}"

export DEPLOY_CORE_DNS=${DEPLOY_CORE_DNS:-true}

if [[ "${USE_EXISTING_TPUS}" = false ]]; then
  if [[ "${DEPLOY_CORE_DNS}" = true ]]; then
    # install CoreDNS https://buganizer.corp.google.com/issues/298037363
    sudo apt install jq -y
    pushd ~
    rm -rf deployment && git clone https://github.com/coredns/deployment.git
    pushd deployment/kubernetes
    ./deploy.sh | kubectl apply -f -
    kubectl scale deployment --replicas=0 kube-dns-autoscaler --namespace=kube-system
    kubectl scale deployment --replicas=0 kube-dns --namespace=kube-system
    popd
    popd
  fi
fi

NUM_CHIPS=$(( $(echo "${TPU_TOPOLOGY}" |  sed 's/x/*/g') ))
NUM_CHIPS_PER_NODE=4
NUM_NODES=$(( $NUM_CHIPS / $NUM_CHIPS_PER_NODE ))
MAINTENANCE_INTERVAL="PERIODIC"

export NUM_SLICES=${NUM_SLICES:-16}

if [[ "${USE_EXISTING_TPUS}" = false ]]; then
  for i in $(seq 0 $((NUM_SLICES - 1))); do
    retry gcloud beta container node-pools create "${CLUSTER_NAME}-${i}" \
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
      --max-pods-per-node=24 &
    if (( $((i+1)) % 5 == 0 )) || (( $((i+1)) == $NUM_SLICES )); then
      wait
    fi
  done

  echo "Node pools created:"
  # set container/cluster
  gcloud beta container node-pools list --zone="$ZONE" --cluster="$CLUSTER_NAME"
  # deploy server side manifests
  VERSION=v0.2.3
  kubectl apply --server-side -f https://github.com/kubernetes-sigs/jobset/releases/download/${VERSION}/manifests.yaml
  # wait for the deployment in the server side
  sleep 120
fi

##config experiment and rebuild docker
# cloud-tpu-v2-images is a public accessible account for images across different projects
JAX_LIBTPU_IMAGE=${JAX_LIBTPU_IMAGE:-gcr.io/cloud-tpu-v2-images/pax-jax-libtpu:2023_09_14_final_sanitized}
TAG=${JAX_LIBTPU_IMAGE}
ACCELERATOR_TYPE=$(echo "${ACCELERATOR}" | awk -F"-" '{ print $2 }')
GS_PREFIX=mlperf-exp/submissions/${USER}/${EXP_NAME}/${NUM_SLICES}_${ACCELERATOR_TYPE}_${NUM_CHIPS}

ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-true}

batch_size=$((NUM_SLICES*128))
start_step=$((1536 * 4000 / batch_size))
step_str=$(printf "%08d" ${start_step})
CKP_INIT_PATH=gs://mlperf-exp/gpt3/checkpoints/checkpoint_"${step_str}"

gsutil -m cp -r "${SCRIPTS_DIR}"/src gs://"${GS_PREFIX}"/mlperf_test_script/

gsutil cp "${SCRIPTS_DIR}"/src/pax_src/checkpoint_creators.py gs://"${GS_PREFIX}"/mlperf_test_script/patch_src/
gsutil cp "${SCRIPTS_DIR}"/src/pax_src/programs.py gs://"${GS_PREFIX}"/mlperf_test_script/patch_src/

gsutil -m cp -r "${SCRIPTS_DIR}"/tools/*.sh gs://"${GS_PREFIX}"/mlperf_test_script/
gsutil -m cp -r "${SCRIPTS_DIR}"/scripts/*.sh gs://"${GS_PREFIX}"/mlperf_test_script/

if [[ "${ENABLE_CHECKPOINT}" = true ]]; then
  gsutil -m cp -r "${CKP_INIT_PATH}" gs://"${GS_PREFIX}"/job_log_dir/checkpoints/
fi
