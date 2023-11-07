: "${DGXNNODES:?DGXNNODES not set}"
: "${DGXNGPU:?DGXNGPU not set}"
: "${BATCHSIZE:?BATCHSIZE not set}"

# NOTE: GRAD_ACCUMULATION_STEPS does not affect GBS (only LBS)
_GBS=$(( $DGXNNODES * $DGXNGPU * $BATCHSIZE ))
_PROXY_GBS=512  # hparams used for proxy runs (when convergence is not important)

function hparams_fpath() {
  echo "$(dirname ${BASH_SOURCE[0]})/hyperparameters_${1}.sh"
}

if [ -f "$(hparams_fpath ${_GBS})" ]; then
  echo "Loading hyperparameters for GBS=${_GBS}."
  source $(hparams_fpath ${_GBS})

elif [ ${_GBS} -lt 200 ]; then
  echo "Small GBS detected. Assuming a proxy run with hyperparameters for GBS=${_PROXY_GBS}."
  source $(hparams_fpath ${_PROXY_GBS})

else
  echo "ERROR: No hyperparameters defined for GBS=${_GBS}. Do either of:"
  echo " 1) Fix GBS"
  echo " 2) create a $(hparams_fpath ${_GBS}) file"
  echo " 3) load a specific hyperparameters_XXX.sh file instead of hyperparameters_auto.sh"
fi


