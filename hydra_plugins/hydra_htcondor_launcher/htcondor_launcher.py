# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import importlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cloudpickle

from hydra.core.singleton import Singleton
from hydra.core.utils import (
    JobReturn,
    JobStatus,
    filter_overrides,
    run_job,
    setup_globals,
)
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf, open_dict

# Import config module to trigger ConfigStore registration
from . import config as _  # noqa: F401

log = logging.getLogger(__name__)

# Runner script that gets executed by HTCondor on compute nodes
RUNNER_SCRIPT = '''#!/usr/bin/env python3
"""HTCondor job runner - unpickles and executes the Hydra task."""
import sys
from pathlib import Path

import cloudpickle

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <job_pickle_file>", file=sys.stderr)
        sys.exit(1)

    job_pickle = Path(sys.argv[1])
    result_pickle = job_pickle.with_suffix(".result.pkl")

    try:
        # Load the pickled job
        with open(job_pickle, "rb") as f:
            job_data = cloudpickle.load(f)

        launcher = job_data["launcher"]
        args = job_data["args"]

        # Execute the job
        result = launcher(*args)

        # Save the result
        with open(result_pickle, "wb") as f:
            cloudpickle.dump({"status": "success", "result": result}, f)

    except Exception as e:
        import traceback
        # Save the exception
        with open(result_pickle, "wb") as f:
            cloudpickle.dump({
                "status": "error",
                "exception": e,
                "traceback": traceback.format_exc()
            }, f)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''


class HTCondorLauncher(Launcher):
    """HTCondor launcher for Hydra multirun jobs using HTCondor Python bindings."""

    def __init__(self, **params: Any) -> None:
        self.params = {}
        for k, v in params.items():
            if OmegaConf.is_config(v):
                v = OmegaConf.to_container(v, resolve=True)
            self.params[k] = v

        log.info(f"HTCondor launcher initialized with params: {self.params}")

        self.config: Optional[DictConfig] = None
        self.task_function: Optional[TaskFunction] = None
        self.hydra_context: Optional[HydraContext] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        """Launch jobs using HTCondor."""

        assert self.config is not None
        assert self.hydra_context is not None
        assert self.task_function is not None

        num_jobs = len(job_overrides)
        assert num_jobs > 0

        log.info(f"HTCondor launcher submitting {num_jobs} jobs")
        log.info(f"Sweep output dir: {self.config.hydra.sweep.dir}")

        # Create sweep directory
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        use_local_mode = bool(self.params.get("use_local_mode", False))

        if use_local_mode:
            log.info("HTCondor launcher running in local mode (no HTCondor submission)")
        else:
            log.info("Submitting jobs to HTCondor")
        log.info(
            f"HTCondor config: memory={self.params.get('request_memory', '4000')}MB, "
            f"cpus={self.params.get('request_cpus', '1')}, "
            f"gpus={self.params.get('request_gpus', '0')}"
        )

        # Build HTCondor executor
        htcondor_folder = self.params.get(
            "htcondor_folder", "${hydra.sweep.dir}/.htcondor"
        )
        htcondor_folder = htcondor_folder.replace("${hydra.sweep.dir}", str(sweep_dir))
        htcondor_dir = Path(htcondor_folder)
        htcondor_dir.mkdir(parents=True, exist_ok=True)

        # Create job parameters
        job_params: List[Any] = []
        for idx, overrides in enumerate(job_overrides):
            job_idx = initial_job_idx + idx
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{job_idx} : {lst}")
            job_params.append(
                (
                    list(overrides),
                    "hydra.sweep.dir",
                    job_idx,
                    f"job_id_for_{job_idx}",
                    Singleton.get_state(),
                )
            )

        if use_local_mode:
            return [self._execute_job(params) for params in job_params]

        htcondor = self._load_htcondor_module()

        # Create HTCondor executor with reference to this launcher
        executor = HTCondorExecutor(htcondor_dir, self.params, htcondor, self)

        # Submit jobs (returns submitted job metadata for tracking)
        jobs, submissions = executor.map_array(job_params)
        self._record_submissions(submissions, sweep_dir, htcondor_dir)

        wait_for_jobs = bool(self.params.get("wait_for_jobs", False))
        if wait_for_jobs:
            # Block until HTCondor finishes processing the array
            return [j.result() for j in jobs]

        log.info(
            "Non-blocking mode enabled; returning immediately after job submission."
        )
        submission_lookup = {entry["job_index"]: entry for entry in submissions}
        return [
            self._build_nonblocking_return(job_param, submission_lookup.get(job_param[2]))
            for job_param in job_params
        ]

    def _execute_job(self, job_param: Any) -> JobReturn:
        """Execute a single job locally (used for development mode)."""
        overrides, job_dir_key, job_idx, job_id, singleton_state = job_param
        return self(
            overrides,
            job_dir_key,
            job_idx,
            job_id,
            singleton_state,
        )

    @staticmethod
    def _load_htcondor_module() -> Any:
        """Import the htcondor module (htcondor2 fallback) with a clearer error message."""
        for module_name in ("htcondor2", "htcondor"):
            try:
                return importlib.import_module(module_name)
            except ImportError:
                continue
        raise RuntimeError(
            "htcondor Python bindings are required. Install the `htcondor` extra "
            "(imports htcondor2/htcondor) or set `hydra.launcher.use_local_mode=true` "
            "for local testing."
        )

    def _build_nonblocking_return(
        self,
        job_param: Any,
        submission: Optional[Dict[str, Any]],
    ) -> JobReturn:
        """Create a placeholder JobReturn for non-blocking submissions."""
        overrides, job_dir_key, job_idx, job_id, _ = job_param
        job_return = JobReturn()
        job_return.overrides = overrides
        job_return.status = JobStatus.COMPLETED
        job_return.return_value = {
            "status": "submitted",
            "cluster_id": submission.get("cluster_id") if submission else None,
            "proc_id": submission.get("proc_id") if submission else None,
            "job_index": job_idx,
            "job_dir": submission.get("job_dir") if submission else None,
            "message": (
                "Job submitted to HTCondor and continues running asynchronously."
            ),
        }
        return job_return

    def _record_submissions(
        self,
        submissions: List[Dict[str, Any]],
        sweep_dir: Path,
        htcondor_dir: Path,
    ) -> None:
        """Persist submitted job metadata for easier follow-up or cancellation."""
        if not submissions:
            return

        record_path_value = self.params.get("submission_cache_file")
        if record_path_value:
            record_path_str = record_path_value.replace(
                "${hydra.sweep.dir}", str(sweep_dir)
            )
        else:
            record_path_str = str(htcondor_dir / "submitted_jobs.json")

        record_path = Path(record_path_str)
        record_path.parent.mkdir(parents=True, exist_ok=True)

        existing: List[Dict[str, Any]] = []
        if record_path.exists():
            try:
                existing = json.loads(record_path.read_text())
            except Exception as exc:  # pragma: no cover - best effort
                log.warning(
                    "Failed to read existing submission cache %s: %s", record_path, exc
                )

        entry = {
            "submitted_at": datetime.now(tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "sweep_dir": str(sweep_dir),
            "jobs": submissions,
        }
        existing.append(entry)
        record_path.write_text(json.dumps(existing, indent=2))
        log.info(
            "Recorded %s HTCondor job(s) to %s",
            len(submissions),
            record_path,
        )

    def __call__(
        self,
        sweep_overrides: List[str],
        job_dir_key: str,
        job_num: int,
        job_id: str,
        singleton_state: Dict[type, Singleton],
    ) -> JobReturn:
        """Execute a single job - called by HTCondor on compute nodes."""
        assert self.hydra_context is not None
        assert self.config is not None
        assert self.task_function is not None

        Singleton.set_state(singleton_state)
        setup_globals()

        sweep_config = self.hydra_context.config_loader.load_sweep_config(
            self.config, sweep_overrides
        )

        with open_dict(sweep_config.hydra.job) as job:
            job.id = job_id
            job.num = job_num

        return run_job(
            hydra_context=self.hydra_context,
            task_function=self.task_function,
            config=sweep_config,
            job_dir_key=job_dir_key,
            job_subdir_key="hydra.sweep.subdir",
        )


class HTCondorJob:
    """HTCondor job wrapper for tracking and result collection."""

    def __init__(
        self,
        cluster_id: int,
        job_id: int,
        htcondor_module: Any,
        log_file: str,
        output_file: str,
        error_file: str,
        result_pickle: str,
    ):
        self.cluster_id = cluster_id
        self.job_id = job_id
        self.htcondor = htcondor_module
        self.log_file = Path(log_file)
        self.output_file = Path(output_file)
        self.error_file = Path(error_file)
        self.result_pickle = Path(result_pickle)

    def result(self, timeout: Optional[float] = None) -> JobReturn:
        """Wait for job completion and return JobReturn."""
        import time

        start_time = time.time()

        # Poll job status until completion
        schedd = self.htcondor.Schedd()

        while True:
            # Check if timeout exceeded
            if timeout and (time.time() - start_time) > timeout:
                result = JobReturn()
                result.status = JobStatus.FAILED
                result.exception = TimeoutError(
                    f"Job {self.cluster_id}.{self.job_id} timed out after {timeout}s"
                )
                return result

            # Query job status
            try:
                jobs = list(
                    schedd.query(
                        f"ClusterId == {self.cluster_id} && ProcId == {self.job_id}"
                    )
                )
                if not jobs:
                    # Job not found, might be completed and cleaned up
                    break

                job = jobs[0]
                job_status = job.get("JobStatus", 0)

                # HTCondor job status codes:
                # 1 = Idle, 2 = Running, 3 = Removed, 4 = Completed, 5 = Held, 6 = Transferring output
                if job_status in [4, 3]:  # Completed or Removed
                    break
                elif job_status == 5:  # Held
                    result = JobReturn()
                    result.status = JobStatus.FAILED
                    hold_reason = job.get("HoldReason", "Unknown hold reason")
                    result.exception = RuntimeError(f"Job held: {hold_reason}")
                    return result

            except Exception as e:
                log.warning(f"Error querying job status: {e}")

            time.sleep(5)  # Poll every 5 seconds

        # Job completed - read result from pickle file
        return self._load_result()

    def _load_result(self) -> JobReturn:
        """Load job result from pickle file."""
        if not self.result_pickle.exists():
            # No result file - check error file for clues
            result = JobReturn()
            result.status = JobStatus.FAILED
            error_msg = "Job completed but no result file found"
            if self.error_file.exists():
                try:
                    stderr_content = self.error_file.read_text().strip()
                    if stderr_content:
                        error_msg += f"\nStderr: {stderr_content}"
                except Exception:
                    pass
            result.exception = RuntimeError(error_msg)
            return result

        try:
            with open(self.result_pickle, "rb") as f:
                data = cloudpickle.load(f)

            if data["status"] == "success":
                return data["result"]
            else:
                # Job failed with exception
                result = JobReturn()
                result.status = JobStatus.FAILED
                result.exception = data.get("exception", RuntimeError("Unknown error"))
                return result

        except Exception as e:
            result = JobReturn()
            result.status = JobStatus.FAILED
            result.exception = RuntimeError(f"Failed to load result pickle: {e}")
            return result


class HTCondorExecutor:
    """HTCondor executor that serializes jobs via pickle."""

    def __init__(
        self,
        folder: Path,
        params: Dict[str, Any],
        htcondor_module: Any,
        launcher: HTCondorLauncher,
    ):
        self.folder = Path(folder)
        self.params = params
        self.htcondor = htcondor_module
        self.launcher = launcher
        self._setup_runner_script()

    def _setup_runner_script(self) -> Path:
        """Create the runner script in the htcondor folder."""
        runner_path = self.folder / "htcondor_runner.py"
        runner_path.write_text(RUNNER_SCRIPT)
        runner_path.chmod(0o755)
        self._runner_path = runner_path
        return runner_path

    def map_array(self, job_params: List[Any]) -> Tuple[List["HTCondorJob"], List[Dict[str, Any]]]:
        """Submit array of jobs to HTCondor using pickle serialization."""
        jobs: List[HTCondorJob] = []
        submissions: List[Dict[str, Any]] = []
        schedd = self.htcondor.Schedd()

        for job_param in job_params:
            overrides, job_dir_key, job_idx, job_id, singleton_state = job_param
            lst = " ".join(filter_overrides(overrides))
            log.info(f"\t#{job_idx} : {lst}")

            # Create job-specific paths
            job_dir = self.folder / f"job_{job_idx}"
            job_dir.mkdir(exist_ok=True)

            job_pickle = job_dir / "job.pkl"
            result_pickle = job_dir / "job.result.pkl"
            job_output = job_dir / "job.out"
            job_error = job_dir / "job.err"
            job_log = job_dir / "job.log"

            # Serialize the launcher and job arguments
            job_data = {
                "launcher": self.launcher,
                "args": (overrides, job_dir_key, job_idx, job_id, singleton_state),
            }

            with open(job_pickle, "wb") as f:
                cloudpickle.dump(job_data, f)

            # Create HTCondor submit description
            submit_dict = {
                "executable": str(self.params.get("executable", sys.executable)),
                "arguments": f"{self._runner_path} {job_pickle}",
                "output": str(self.params.get("output", job_output)),
                "error": str(self.params.get("error", job_error)),
                "log": str(self.params.get("log", job_log)),
                "request_memory": str(self.params.get("request_memory", "4000")),
                "request_cpus": str(self.params.get("request_cpus", "1")),
                "request_gpus": str(self.params.get("request_gpus", "0")),
                "should_transfer_files": str(
                    self.params.get("should_transfer_files", "YES")
                ),
                "when_to_transfer_output": str(
                    self.params.get("when_to_transfer_output", "ON_EXIT")
                ),
                "getenv": str(self.params.get("getenv", "True")),
                "initialdir": str(job_dir),
            }

            priority = self.params.get("priority")
            if priority is not None:
                submit_dict["priority"] = str(priority)

            transfer_inputs = [str(job_pickle), str(self._runner_path)]
            user_transfer_inputs = self.params.get("transfer_input_files")
            if user_transfer_inputs:
                transfer_inputs.append(str(user_transfer_inputs))
            submit_dict["transfer_input_files"] = ",".join(transfer_inputs)

            transfer_outputs = [str(result_pickle.name)]
            user_transfer_outputs = self.params.get("transfer_output_files")
            if user_transfer_outputs:
                transfer_outputs.insert(0, str(user_transfer_outputs))
            submit_dict["transfer_output_files"] = ",".join(transfer_outputs)

            transfer_output_remaps = [f'"{result_pickle.name}={result_pickle}"']
            user_output_remaps = self.params.get("transfer_output_remaps")
            if user_output_remaps:
                transfer_output_remaps.insert(0, str(user_output_remaps))
            submit_dict["transfer_output_remaps"] = ";".join(transfer_output_remaps)

            # Add requirements if specified
            if "requirements" in self.params:
                submit_dict["requirements"] = str(self.params["requirements"])

            # Add MaxTime and periodic_remove if specified
            if "MaxTime" in self.params:
                submit_dict["MaxTime"] = str(self.params["MaxTime"])
                if "periodic_remove" not in self.params:
                    submit_dict["periodic_remove"] = (
                        "(JobStatus =?= 2) && "
                        f"((CurrentTime - JobCurrentStartDate) >= {self.params['MaxTime']})"
                    )

            # Add any additional custom parameters
            reserved_keys = {
                "executable",
                "arguments",
                "output",
                "error",
                "log",
                "request_memory",
                "request_cpus",
                "request_gpus",
                "should_transfer_files",
                "transfer_input_files",
                "when_to_transfer_output",
                "transfer_output_files",
                "transfer_output_remaps",
                "getenv",
                "initialdir",
                "use_htcondor",
                "output_dir",
                "htcondor_folder",
                "requirements",
                "MaxTime",
                "periodic_remove",
                "use_local_mode",
                "wait_for_jobs",
                "submission_cache_file",
                "priority",
            }
            for key, value in self.params.items():
                if key not in reserved_keys:
                    submit_dict[key] = str(value)

            # Submit the job
            submit_obj = self.htcondor.Submit(submit_dict)
            submit_result = schedd.submit(submit_obj)

            cluster_id = submit_result.cluster()
            log.info(f"Submitted job {job_idx} as HTCondor job {cluster_id}.0")
            submissions.append(
                {
                    "job_index": job_idx,
                    "cluster_id": cluster_id,
                    "proc_id": 0,
                    "job_dir": str(job_dir),
                    "overrides": list(overrides),
                }
            )

            # Create HTCondorJob wrapper
            htcondor_job = HTCondorJob(
                cluster_id=cluster_id,
                job_id=0,
                htcondor_module=self.htcondor,
                log_file=str(job_log),
                output_file=str(job_output),
                error_file=str(job_error),
                result_pickle=str(result_pickle),
            )
            jobs.append(htcondor_job)

        return jobs, submissions
