# Hydra HTCondor Launcher Plugin

A Hydra launcher plugin for submitting jobs to HTCondor clusters using the HTCondor Python bindings.

## Overview

This plugin provides a custom Launcher for Hydra that submits multirun jobs to HTCondor clusters. It uses the [HTCondor Python bindings](https://htcondor.readthedocs.io/en/24.x/apis/python-bindings/tutorials/index.html) to interact with the HTCondor scheduler.

## Installation

1. Install the plugin:
```bash
pip install -e .
```

2. Install the HTCondor Python bindings when you plan to submit to a real cluster. Use the optional `htcondor` extra so dependency managers keep the requirement in sync (the launcher imports `htcondor2` first, then falls back to `htcondor`):
```bash
pip install ".[htcondor]"
# or, if you use uv:
uv sync --group htcondor
```

## Configuration

The HTCondor launcher configuration supports all standard HTCondor submission parameters:

```yaml
# Custom executable (optional) - if not specified, uses Python
executable: "/path/to/executable.sh"

# Output files with HTCondor variable substitution
error: "outputs/$(Cluster)_$(Process).err"
output: "outputs/$(Cluster)_$(Process).out" 
log: "outputs/$(Cluster)_$(Process).log"

# HTCondor job requirements
request_memory: "64000"  # Memory in MB
request_cpus: "8"        # Number of CPUs
request_gpus: "1"        # Number of GPUs
priority: 10             # Optional HTCondor priority boost

# HTCondor job constraints
requirements: "TARGET.CUDAGlobalMemoryMb > 64000"

# Maximum job runtime in seconds
MaxTime: 28800  # 8 hours

# Any additional HTCondor parameters
periodic_remove: "(JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))"

# Control blocking behavior (defaults shown)
use_local_mode: false        # Run jobs locally when true
wait_for_jobs: false         # When false, exit right after queueing the jobs
submission_cache_file: ${hydra.sweep.dir}/.htcondor/submitted_jobs.json
```

Any other HTCondor submit attribute (for example `environment`, `periodic_remove`, or custom file paths) can be added directly under `hydra.launcher` and will be forwarded untouched.

### HTCondor Variable Substitution

The launcher supports HTCondor's built-in variable substitution:
- `$(Cluster)` - The cluster ID assigned by HTCondor
- `$(Process)` - The process ID (0, 1, 2, ... for each job in the array)
- Any custom variables you define

### Custom Executables

You can specify a custom executable (like a wrapper script) that will receive the Hydra job runner as an argument:
```yaml
executable: "/path/to/your/cuda_wrapper.sh"
```

The wrapper will be called as:
```bash
/path/to/your/cuda_wrapper.sh /path/to/hydra_job_runner.py $(Process)
```

## Usage

To use the HTCondor launcher, specify it in your Hydra configuration:

```yaml
defaults:
  - override hydra/launcher: htcondor
```

Or use it from the command line:
```bash
python my_app.py --multirun hydra/launcher=htcondor db=postgresql,mysql
```

## Example

Run the example application (this defaults to `use_local_mode=true`, so it runs entirely locally and does not require an HTCondor installation):
```bash
uv run example/my_app.py --multirun
```

Expected output:
```text
[2024-01-01 10:00:00,000] - HTCondor launcher submitting 2 jobs
[2024-01-01 10:00:00,000] - Sweep output dir : multirun/2024-01-01/10-00-00
[2024-01-01 10:00:00,000] -     #0 : db=postgresql
[2024-01-01 10:00:00,000] -     #1 : db=mysql
[2024-01-01 10:00:01,000] - Submitted HTCondor cluster 12345 with 2 jobs
```

To submit to a real cluster, set `hydra.launcher.use_local_mode=false` (either in `example/config.yaml` or via the command line) so that jobs are sent through the HTCondor scheduler.
Make sure the `htcondor` extra is installed first, e.g.:
```bash
pip install ".[htcondor]"
# or
uv sync --group htcondor
```

### Tracking & canceling queued jobs

By default the launcher queues jobs and returns immediately. Every submission is appended to `submitted_jobs.json` inside the `.htcondor` folder (override via `submission_cache_file`). The file contains each HTCondor `ClusterId`, so you can later cancel the sweep with:
```bash
condor_rm <ClusterId>
```
Set `wait_for_jobs=true` if you prefer the launcher to block until all jobs finish.

## Features

- **HTCondor Integration**: Uses HTCondor Python bindings for native cluster interaction
- **Resource Management**: Configurable CPU, memory, and GPU requirements
- **Job Monitoring**: Automatic job status monitoring and result collection
- **Error Handling**: Robust error handling and job cleanup
- **File Transfer**: Automatic file transfer for job scripts and results

## Requirements

- Python 3.8+
- Hydra Core 1.3.2+
- HTCondor Python bindings 24.0.0+
- Access to an HTCondor cluster
