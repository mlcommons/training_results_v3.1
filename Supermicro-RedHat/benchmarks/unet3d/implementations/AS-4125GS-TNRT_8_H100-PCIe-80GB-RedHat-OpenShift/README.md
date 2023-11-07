# 1. Problem

This benchmark represents a 3D medical image segmentation task using a [2019 Kidney Tumor Segmentation Challenge](https://kits19.grand-challenge.org/) dataset called [KiTS19](https://github.com/neheller/kits19). The task is carried out using a U-Net3D model variant based on the [No New-Net](https://arxiv.org/pdf/1809.10483.pdf) paper.

## Requirements
* [Red Hat OpenShift](https://access.redhat.com/documentation/en-us/openshift_container_platform/4.13/html/installing/index)
* [OpenShift Local Storage Operator](https://access.redhat.com/documentation/en-us/red_hat_openshift_container_storage/4.8/html/deploying_openshift_container_storage_using_ibm_z_infrastructure/deploy-using-local-storage-devices-ibmz)
* [Node Feature Discovery Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-01.html) 
* [Nvidia GPU Operator](https://docs.nvidia.com/launchpad/infrastructure/openshift-it/latest/openshift-it-step-03.html) 
* [Podman](https://developers.redhat.com/blog/2018/08/29/intro-to-podman) 

## Dataset

The data is stored in the [KiTS19 github repository](https://github.com/neheller/kits19).

## Publication/Attribution
Heller, Nicholas and Isensee, Fabian and Maier-Hein, Klaus H and Hou, Xiaoshuai and Xie, Chunmei and Li, Fengyi and Nan, Yang and Mu, Guangrui and Lin, Zhiyong and Han, Miofei and others.
"The state-of-the-art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 Challenge".
Medical Image Analysis, 101821, Elsevier (2020).

Heller, Nicholas and Sathianathen, Niranjan and Kalapara, Arveen and Walczak, Edward and Moore, Keenan and Kaluzniak, Heather and Rosenberg, Joel and Blake, Paul and Rengel, Zachary and Oestreich, Makinna and others.
"The kits19 challenge data: 300 kidney tumor cases with clinical context, ct semantic segmentations, and surgical outcomes".
arXiv preprint arXiv:1904.00445 (2019).

# 2. Directions

## Steps to download and verify data

1. Build and run the dataset preprocessing container. Run the following on your bastion host (note: bastion host has ssh access to your OpenShift cluster). Use the [Nvidia Unet3d Dockerfile](https://github.com/mlcommons/submissions_training_v3.1/blob/main/NVIDIA/benchmarks/unet3d/implementations/mxnet/Dockerfile).
    
    ```bash
    podman build -t preprocessing -f Dockerfile_pyt .
    podman run --ipc=host -it --rm -v DATADIR:/data preprocessing:latest 
    ```
   Where DATADIR is the target directory used to store the dataset after preprocessing.

   
2. Download and preprocess the data

    ```bash
    bash download_dataset.sh 
    ```


## Steps to load data into local NVME drive

1. Create local storage namespace.   Run the "oc" commands from your bastion host.
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

7.  Run Unet3D model training. 

```
oc create -f pod-unet3d.yaml
```
