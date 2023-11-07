# 1. Problem

This problem uses the ResNet-50 CNN to do image classification.

## Requirements
* [Red Hat OpenShift](https://access.redhat.com/documentation/en-us/openshift_container_platform/4.13/html/installing/index)
* [OpenShift Local Storage Operator](https://access.redhat.com/documentation/en-us/red_hat_openshift_container_storage/4.8/html/deploying_openshift_container_storage_using_ibm_z_infrastructure/deploy-using-local-storage-devices-ibmz)
* [Node Feature Discovery Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-01.html) 
* [Nvidia GPU Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-03.html) 
* [Podman](https://developers.redhat.com/blog/2018/08/29/intro-to-podman) 

# 2. Directions

## Steps to download and verify data

1. Clone the public DeepLearningExamples repository
```
git clone https://github.com/NVIDIA/DeepLearningExamples
cd DeepLearningExamples/MxNet/Classification/RN50v1.5
git checkout 81ee705868a11d6fe18c12d237abe4a08aab5fd6
```

2. Build a ResNet50 MXNet container using this [Nvidia ResNet50 Dockerfile](https://github.com/mlcommons/submissions_training_v3.1/blob/main/NVIDIA/benchmarks/resnet/implementations/mxnet/Dockerfile).
```
podman build . -t nvidia_rn50_mx
```
3. Start an interactive session in the NGC container to run preprocessing
```
podman run --rm -it --ipc=host -v <path/to/store/raw/&/processed/data>:/data nvidia_rn50_mx
```

4. Download and unpack the data
* Download **Training images (Task 1 &amp; 2)** and **Validation images (all tasks)** at http://image-net.org/challenges/LSVRC/2012/2012-downloads (require an account)
* Extract the training data:
    ```
    mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
    tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    cd ..
    ```
    
* Extract the validation data:
    ```
    mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val 
    tar -xvf ILSVRC2012_img_val.tar && rm -f ILSVRC2012_img_val.tar
    wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
    ```

5. Preprocess the dataset
```
./scripts/prepare_imagenet.sh <path/to/raw/imagenet> <path/to/save/preprocessed/data>
```

## Steps to load data into local NVME drive

1. Create local storage namespace 
```
oc new-project openshift-local-storage
```

2. Install Local Storage Operator, LSO, in namespace openshift-local-storage
[Follow these steps](https://access.redhat.com/documentation/en-us/red_hat_openshift_container_storage/4.6/html/deploying_openshift_container_storage_using_bare_metal_infrastructure/deploy-using-local-storage-devices-bm) (see section 1.4). 


3. Create a permanent volume (PV) for each NVMe drive using LSO.  
LSO will automatically discover the NVMe devices, and will create PVs for them, after you create the following Local Volume Custom Resourse (CR) with your devicePaths specified:

```
kind: "LocalVolume"
metadata:
  name: "local-disks"
  namespace: "openshift-local-storage" 
spec:
  nodeSelector: 
    nodeSelectorTerms:
    - matchExpressions:
        - key: cluster.ocs.openshift.io/openshift-storage
          operator: In
          values:
          - ""
  storageClassDevices:
    - storageClassName: "local-sc" 
      volumeMode: Filesystem 
      fsType: xfs 
      devicePaths: 
        - /dev/disk/by-id/nvme-Dell_Ent_NVMe_P5600_MU_U.2_3.2TB_PHAB1222045E3P8EGN
```

3. Create a permanent volume claim (PVC) for your locally attached NVMe drive.  
   Save the following to file mypvc_0.yaml, specifying the storage size of your NVME (e.g. below the storage size is "2980Gi)

```
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-amazing-pvc0
spec:
  accessModes:
  - ReadWriteOnce
  volumeMode: Filesystem
  resources:
    requests:
      storage: 2980Gi
  storageClassName: local-sc
```
  

```
oc create -f mypvc_0.yaml
```
4. Create a filesystem on the NVME drive. 

    Get the names of your nodes
```
oc get nodes
```

example output:
```
[myaccount@bastion]# oc get nodes
NAME           STATUS   ROLES                         AGE    VERSION
e27-h13-r750   Ready    control-plane,master,worker   124d   v1.25.7+eab9cc9
e27-h15-r750   Ready    control-plane,master,worker   124d   v1.25.7+eab9cc9
e28-h15-r750   Ready    control-plane,master,worker   124d   v1.25.7+eab9cc9
```

On your worker node create an xfs filesystem, create a mount directory and mount the filesystem. We will rsync the training data to this filesystem on our NVMe drive.

```
oc debug node/<my-node-name-with-locally-attached-NMMe>
sh-4.4# chroot /host
sh-4.4# lsblk
NAME    MAJ:MIN RM   SIZE RO TYPE MOUNTPOINT
.
.
nvme2n1 259:0    0   1.5T  0 disk 
nvme3n1 259:1    0   1.5T  0 disk 
nvme4n1 259:2    0   2.9T  0 disk 
nvme5n1 259:3    0   1.5T  0 disk 
nvme1n1 259:4    0   1.5T  0 disk 
nvme0n1 259:5    0   1.5T  0 disk

sh-4.4# mkfs.xfs -f /dev/nvme4n1
sh-4.4# mkdir /mnt/data
sh-4.4# mount /dev/nvme4n1 /mnt/data
```
5. Create a pod that mounts your NVMe drive and rsync your data to it.  
Create a file, mypod.yaml, containing the following pod specification. 
```
apiVersion: v1
kind: Pod
metadata:
  name: mypod
spec:
  containers:
    - name: mytestpod
      image: redhat/ubi8
      command: ["/bin/bash", "-ec", "tail -f /dev/null" ]
      volumeMounts:
      - mountPath: "/data"
        name: mypd
  volumes:
    - name: mypd
      persistentVolumeClaim:
        claimName: my-amazing-pvc0
```
Create the pod. 
``` 
oc create -f mypodm.yaml
```
6. Copy the training dataset to the volume mounted in the the running pod.  Verify that the data was copied to the volume, and then delete the pod.

```
oc rsync <training-data> mypod:/data/
oc exec --stdin --tty mypod -- /bin/bash  # connect to the running pod and verify that the training data is loaded
cd /data
ls
```

7.  Run Resnet-50 model training. 

```
oc create -f pod-renet50.yaml
```
