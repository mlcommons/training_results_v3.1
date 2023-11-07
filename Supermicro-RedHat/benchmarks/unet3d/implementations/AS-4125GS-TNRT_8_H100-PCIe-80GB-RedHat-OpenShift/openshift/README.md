# Requirements:  
*  [Install Openshift](https://access.redhat.com/documentation/en-us/openshift_container_platform/4.13/html/installing/index) (and operators listed below)
*  8 Nvidia H100 GPUs
*  7TB locally attached NVME

# Install these Operators from the Operator Hub
*  Node Feature Discovery Operator
*  Nvidia GPU Operator
*  Local Storage Operator

# Create PVs and PVCs for your training data

Local Storage Operator discovers the NVMe device and the other devices and creates PVs.

IMPORTANT: PVCs are namespaced - PVs are not. Make sure your PVC is in the same namespace where you will run your model. 

Create a PVC for the PV that was automatically created for your locally attached NVMe drive (which will contain your training data)


# Copy the Training data to the correct directories for each of the models: 

bertBackup/  

checkpoint_dir/  

hdf5/  

packed_data/  

per_seqlen/  

per_seqlen_parts/  

phase1/  

phase2/  

results/  

undet3d/  

unet3dtraining/  


# Run each of the models using the yaml provided under benchmarks: 

for Bert:
oc create -f pod-bert.yaml

for resnet50:
oc create -f pod-resnet50.yaml

for 3d U-Net:
oc create -f pod-unet3d.yaml 






