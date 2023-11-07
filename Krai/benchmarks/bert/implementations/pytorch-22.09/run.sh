: "${CLEAR_CACHES:=1}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${LOGDIR:=$(pwd)/results}"
: "${NEXP:=10}"
: "${LOG_FILE_BASE:="${LOGDIR}/${DATESTAMP}"}"

for _experiment_index in $(seq 1 "${NEXP}"); do
    (
        echo "Beginning trial ${_experiment_index} of ${NEXP}"

        # Clear caches
        if [ "${CLEAR_CACHES}" -eq 1 ]; then
            sync && sudo /sbin/sysctl vm.drop_caches=3
            python -c "
from mlperf_logging.mllog import constants
from mlperf_logger import mllogger
mllogger.event(key=mllogger.constants.CACHE_CLEAR, value=True)"
        fi

        # Run experiment
        ./run_and_time.sh
    ) |& tee "${LOG_FILE_BASE}_${_experiment_index}.log"
done
