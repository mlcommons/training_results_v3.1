## System config params

# ${BASH_SOURCE[1]} assumes this is a depth-1 nested call
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[1]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export DGXSOCKETCORES=56
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

export DGXNGPU=8  # can be overwritten
