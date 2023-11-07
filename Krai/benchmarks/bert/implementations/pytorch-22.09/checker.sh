# Generate a timestamp
timestamp=$(date +%Y%m%d%H%M%S)

for _experiment_index in $(seq 1 10); do
  python3 -m mlperf_logging.compliance_checker --usage training \
           --ruleset "3.1.0"                                 \
           --log_output "/results/compliance_${timestamp}.out" \
           "/results/result_${_experiment_index}.txt" \
          || true
done
