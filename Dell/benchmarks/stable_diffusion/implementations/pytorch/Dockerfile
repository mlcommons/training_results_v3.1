ARG FROM_IMAGE_NAME=nvcr.io/nvdlfwea/mlperfv31/ea-mm-mlperf-sd:23.09-py3
FROM ${FROM_IMAGE_NAME}

RUN pip install --upgrade webdataset

# install LDM
COPY . /source
RUN pip install -r /source/requirements.txt

# Patch PyTorch Lightning to remove additional syncs
# We have merged this with PyTorch Lightning v2 at:
#     https://github.com/Lightning-AI/lightning/pull/17334
# TODO: remove the patch once NeMo upgrades to PL v2
RUN PL_ROOT=$(python -c "import pytorch_lightning; print(pytorch_lightning.__file__.replace('/__init__.py',''))"); \
    patch -p3 -d${PL_ROOT} -i /source/lightning.v1.9.4.patch
