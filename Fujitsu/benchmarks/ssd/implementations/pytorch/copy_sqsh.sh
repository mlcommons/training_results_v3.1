#!/bin/bash

set -x

TMP_SQSH="${SCRATCH_SPACE}/tmp.sqsh"
REQ_SIZE="573332353024"

# Check if squash file exists and is the right size
if [[ -f ${TMP_SQSH} ]]; then
  TMP_SQSH_SIZE=$(stat --printf='%s' ${TMP_SQSH})

  if [[ ${TMP_SQSH_SIZE} -eq ${REQ_SIZE} ]]; then
    echo "squash file exists and is not corrupted"
    exit 0
  else
    echo "squash file exists but is corrupted, copying..."
  fi
else
  echo "squash file doesn't exist, copying..."
fi

dd bs=4M if=${LOCALDISK_FROM_SQUASHFS} of=${SCRATCH_SPACE}/tmp.sqsh oflag=direct
