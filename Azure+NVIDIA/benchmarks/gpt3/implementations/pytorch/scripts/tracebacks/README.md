# Traceback dumping scripts

## Scripts
1. [dump_tracebacks_node.sh](dump_tracebacks_node.sh) - when run inside the container, dumps `py-spy`, `gdb` and `cuda-gdb` tracebacks for all processes listed in `nvidia-smi`
2. [hang_monitor.sh](hang_monitor.sh) - provides functions to start a background process that launches given commands a few minutes before the slurm timout.

## Usage
In run.sub `hang_monitor` is set up to launch an overlapping srun job `HANG_MONITOR_TIMEOUT` minutes before the slurm timeout.
The srun job starts one `dump_tracebacks_node.sh` script instance on every node.
By default, `HANG_MONITOR_TIMEOUT` is set to 0 for jobs shorter than 1h (based on WALLTIME variable) and to 5 otherwise.

## Results
The output of the `hang_monitor` and `dump_tracebacks_node.sh` script is saved to `${LOGDIR}/<prefix>_hang_monitor.log` log file.
The tracebacks are dumped to `${LOGDIR}/tracebacks/<prefix>/rank<rank>_pid<pid>.<ext>` where:
- `<rank>` is the global rank of the given process (SLURM_PROCID set for that process by Slurm)
- `<pid>` is the process id the given process (unique within a node)
- `<ext>` is `pyspy`, `gdb` or `cuda-gdb` depending on the tracing program

## Known limitations
1. `cuda-gdb` is currently known to crash the training.
This won't matter for a hung job, but to make sure pyspy and gdb traceback are collected properly,
cuda-gdb dumping is done in a separate loop over processes.

## Examples of analyzing the tracebacks
1. Check active Python functions: `grep -A 1 -e "(active" *.pyspy`