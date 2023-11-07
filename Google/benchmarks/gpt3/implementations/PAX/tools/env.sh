#!/bin/bash

function retry {
  local command="$*"
  local max=3
  local delay=3

  while true; do
    bash -c "${command}" && break || {
      if [[ $n -lt $max ]]; then
        n=$((n+1))
        echo "Retry after an error: ${command}. "
        sleep $delay;
      else
        echo "[ERROR] Failed after $max attempts: ${command}"
        exit 1
      fi
    }
  done
}

function check_var_exist {
  for var in "$@"; do
    if [[ -z "${!var}" ]];then
      echo "[ERROR] $var is empty or not defined. Please pass it and try again"
      exist 1
    fi
  done
}

function get_ip {
  local website=$1
  # example format:
  #   PING www.googleapis.com (142.251.33.74): 56 data bytes
  ip_address=$(ping "${website}" | head -n 1 | awk '{ print $3 }' | sed "s/.*(//;s/).*//")
  echo "${ip_address}"
}
