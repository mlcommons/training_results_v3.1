# MLPerf Training v3.1 Supermicro & Red Hat Inc Submission


This is a repository for Supermicro & Red Hat's submission to the MLPerf Training v3.1 benchmark.  It
includes implementations of the benchmark code optimized for running on Supermicro AS-4125GS-TNRT servers with NVIDIA H100 GPUs.  

# Contents

Each model implementation in the `benchmarks` subdirectory has:
 
* Code that implements the model 
* A Dockerfile which can be used to build a container for the benchmark
* Documentation on the dataset, model, and machine setup

# Hardware & Software requirements

These benchmarks have been tested on the following machine configuration:

* A Supermicro GPU A+ Server, the AS-4125GS-TNRT ; with NVIDIA H100 GPUS directly attached 
* The required software stack includes:
    - [Red Hat OpenShift](https://access.redhat.com/documentation/en-us/openshift_container_platform/4.13/html/installing/index)
    - [OpenShift Local Storage Operator](https://access.redhat.com/documentation/en-us/red_hat_openshift_container_storage/4.8/html/deploying_openshift_container_storage_using_ibm_z_infrastructure/deploy-using-local-storage-devices-ibmz)
    - [Node Feature Discovery Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-01.html)
    - [Nvidia GPU Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-03.html)

Each benchmark can be run with the following steps:

1. Follow the instructions in the README to download and format the input data (note: data dowload and formatting is documented by Nvidia in their `benchmarks` subdirectory for each model, we used the same method to download and prepare the data)
2. Copy the data to the locally attached NVME drive
3. Build the Dockerfile using podman
5. Create the pod to run the model training with this command ```oc create -f pod-<model-name>.yaml```
6. Use the OpenShift console to observe resource ultilzation (e.g. GPU utilization) for the model training.
