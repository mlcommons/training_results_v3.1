## Steps to launch training

### Dell Precision 7920 Tower with 2x A5000 using MxNet Release 22.04

Build the ResNet50 MXNet NGC Release 22.04 container
```
docker build \
--build-arg FROM_IMAGE_NAME=nvcr.io/nvidia/mxnet:22.04-py3 \
-t nvidia_rn50_mx -f Dockerfile .
```

Please see [here](../mxnet/README.md) for the detail instructions in running the benchmark. 
