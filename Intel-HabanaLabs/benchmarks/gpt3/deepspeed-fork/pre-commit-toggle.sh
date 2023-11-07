#!/bin/bash

# Path to the pre-commit configuration file
PRE_COMMIT_CONFIG=".pre-commit-config.yaml"

# Install pre-commit
pip install pre-commit
pip install clang-format==15.0.7

# Enable pre-commit
function enable_pre_commit() {
    # Install pre-commit hooks
    pre-commit install
}

# Disable pre-commit
function disable_pre_commit() {
    # Install pre-commit hooks
    pre-commit uninstall
}

# Check if the pre-commit configuration file exists
if [ -f "$PRE_COMMIT_CONFIG" ]; then
  echo "Pre-commit configuration file found: $PRE_COMMIT_CONFIG"
else
  echo "Pre-commit configuration file not found: $PRE_COMMIT_CONFIG"
  exit 1
fi

# Check the command-line argument to enable or disable pre-commit
if [ "$1" == "enable" ]; then
  enable_pre_commit
elif [ "$1" == "disable" ]; then
  disable_pre_commit
else
  echo "Usage: ./pre-commit-toggle.sh [enable|disable]"
  exit 1
fi
