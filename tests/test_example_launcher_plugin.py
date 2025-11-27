# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cloudpickle
import pytest
from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn, JobStatus
from hydra.plugins.launcher import Launcher
from omegaconf import OmegaConf


# Create a mock htcondor module before importing the launcher
mock_htcondor = MagicMock()
sys.modules["htcondor"] = mock_htcondor
sys.modules["htcondor2"] = mock_htcondor


from hydra_plugins.hydra_htcondor_launcher.config import HTCondorQueueConf
from hydra_plugins.hydra_htcondor_launcher.htcondor_launcher import (
    RUNNER_SCRIPT,
    HTCondorExecutor,
    HTCondorJob,
    HTCondorLauncher,
)


class TestPluginDiscovery:
    """Test that the plugin is properly discovered by Hydra."""

    def test_discovery(self) -> None:
        """Tests that this plugin can be discovered via the plugins subsystem."""
        assert HTCondorLauncher.__name__ in [
            x.__name__ for x in Plugins.instance().discover(Launcher)
        ]

    def test_launcher_is_subclass(self) -> None:
        """Test that HTCondorLauncher is a proper Launcher subclass."""
        assert issubclass(HTCondorLauncher, Launcher)


class TestHTCondorLauncherInit:
    """Test HTCondorLauncher initialization."""

    def test_init_with_empty_params(self) -> None:
        """Test initialization with no parameters."""
        launcher = HTCondorLauncher()
        assert launcher.params == {}
        assert launcher.config is None
        assert launcher.task_function is None
        assert launcher.hydra_context is None

    def test_init_with_params(self) -> None:
        """Test initialization with parameters."""
        params = {
            "request_memory": "8000",
            "request_cpus": "4",
            "request_gpus": "1",
        }
        launcher = HTCondorLauncher(**params)
        assert launcher.params["request_memory"] == "8000"
        assert launcher.params["request_cpus"] == "4"
        assert launcher.params["request_gpus"] == "1"

    def test_init_converts_omegaconf(self) -> None:
        """Test that OmegaConf values are converted to containers."""
        cfg = OmegaConf.create({"nested": {"value": 42}})
        launcher = HTCondorLauncher(config_value=cfg)
        assert launcher.params["config_value"] == {"nested": {"value": 42}}
        assert not OmegaConf.is_config(launcher.params["config_value"])


class TestHTCondorLauncherSetup:
    """Test HTCondorLauncher setup method."""

    def test_setup(self) -> None:
        """Test that setup correctly stores config, context, and task function."""
        launcher = HTCondorLauncher()

        mock_config = OmegaConf.create({"key": "value"})
        mock_context = MagicMock()
        mock_task_function = MagicMock()

        launcher.setup(
            hydra_context=mock_context,
            task_function=mock_task_function,
            config=mock_config,
        )

        assert launcher.config == mock_config
        assert launcher.hydra_context == mock_context
        assert launcher.task_function == mock_task_function


class TestHTCondorLauncherLocalMode:
    """Test the launcher local/dev mode."""

    def test_local_mode_runs_jobs_without_htcondor(self, tmp_path: Path) -> None:
        """Local mode should execute jobs directly instead of submitting to HTCondor."""
        launcher = HTCondorLauncher(use_local_mode=True)
        launcher.config = OmegaConf.create({"hydra": {"sweep": {"dir": str(tmp_path)}}})
        launcher.hydra_context = MagicMock()
        launcher.task_function = MagicMock()

        job_return = JobReturn()
        job_return.status = JobStatus.COMPLETED

        launcher._execute_job = MagicMock(return_value=job_return)

        results = launcher.launch([["db=postgresql"]], initial_job_idx=0)

        assert results == [job_return]
        launcher._execute_job.assert_called_once()


class TestSubmissionTracking:
    """Tests for submission caching and non-blocking helpers."""

    def test_build_nonblocking_return_marks_unknown(self) -> None:
        """Non-blocking placeholder should mark status as UNKNOWN."""
        launcher = HTCondorLauncher()
        job_param = (["a=1"], "hydra.sweep.dir", 0, "job_0", {})
        submission = {
            "cluster_id": 12345,
            "proc_id": 0,
            "job_dir": "/tmp/job_0",
        }
        result = launcher._build_nonblocking_return(job_param, submission)

        assert result.status == JobStatus.COMPLETED
        assert result.overrides == ["a=1"]
        assert result.return_value["cluster_id"] == 12345
        assert result.return_value["job_index"] == 0

    def test_record_submissions_writes_cache(self, tmp_path: Path) -> None:
        """Submission metadata should be appended to cache file."""
        cache_file = tmp_path / "submitted_jobs.json"
        launcher = HTCondorLauncher(submission_cache_file=str(cache_file))
        sweep_dir = tmp_path / "sweep"
        sweep_dir.mkdir()

        submissions = [
            {
                "job_index": 0,
                "cluster_id": 1001,
                "proc_id": 0,
                "job_dir": str(tmp_path / "job_0"),
                "overrides": ["a=1"],
            }
        ]

        launcher._record_submissions(submissions, sweep_dir, tmp_path)

        assert cache_file.exists()
        data = json.loads(cache_file.read_text())
        assert data[-1]["jobs"] == submissions
        assert data[-1]["sweep_dir"] == str(sweep_dir)


class TestHTCondorExecutor:
    """Test HTCondorExecutor functionality."""

    def test_executor_init(self, tmp_path: Path) -> None:
        """Test executor initialization."""
        test_htcondor = MagicMock()
        params = {"request_memory": "4000", "request_cpus": "1"}
        launcher = HTCondorLauncher(**params)
        executor = HTCondorExecutor(tmp_path, launcher.params, test_htcondor, launcher)

        assert executor.folder == tmp_path
        assert executor.params == params
        assert executor.htcondor == test_htcondor
        assert executor.launcher == launcher

    def test_executor_creates_runner_script(self, tmp_path: Path) -> None:
        """Test that executor creates the runner script on init."""
        test_htcondor = MagicMock()
        params = {"request_memory": "4000"}
        launcher = HTCondorLauncher(**params)
        executor = HTCondorExecutor(tmp_path, launcher.params, test_htcondor, launcher)

        runner_path = tmp_path / "htcondor_runner.py"
        assert runner_path.exists()
        assert runner_path.read_text() == RUNNER_SCRIPT
        # Check it's executable
        assert runner_path.stat().st_mode & 0o111

    def test_executor_creates_submit_dict(self, tmp_path: Path) -> None:
        """Test that executor creates proper submit dictionary."""
        params = {
            "request_memory": "8000",
            "request_cpus": "4",
            "request_gpus": "2",
            "requirements": "TARGET.CUDAGlobalMemoryMb > 40000",
            "MaxTime": 3600,
            "priority": 50,
        }

        test_htcondor = MagicMock()
        mock_schedd = MagicMock()
        mock_submit_result = MagicMock()
        mock_submit_result.cluster.return_value = 12345
        mock_schedd.submit.return_value = mock_submit_result
        test_htcondor.Schedd.return_value = mock_schedd

        launcher = HTCondorLauncher(**params)
        executor = HTCondorExecutor(tmp_path, launcher.params, test_htcondor, launcher)

        job_params = [
            (["db=mysql"], "hydra.sweep.dir", 0, "job_0", {}),
        ]

        jobs, submissions = executor.map_array(job_params)

        # Verify submit was called
        assert mock_schedd.submit.called

        # Get the submit object that was created
        submit_call = test_htcondor.Submit.call_args
        submit_dict = submit_call[0][0]

        assert submit_dict["request_memory"] == "8000"
        assert submit_dict["request_cpus"] == "4"
        assert submit_dict["request_gpus"] == "2"
        assert submit_dict["requirements"] == "TARGET.CUDAGlobalMemoryMb > 40000"
        assert submit_dict["MaxTime"] == "3600"
        assert submit_dict["priority"] == "50"
        assert submissions[0]["cluster_id"] == 12345
        assert submissions[0]["job_index"] == 0

    def test_executor_creates_pickle_files(self, tmp_path: Path) -> None:
        """Test that executor creates pickle files for each job."""
        params = {"request_memory": "4000"}

        test_htcondor = MagicMock()
        mock_schedd = MagicMock()
        mock_submit_result = MagicMock()
        mock_submit_result.cluster.return_value = 12345
        mock_schedd.submit.return_value = mock_submit_result
        test_htcondor.Schedd.return_value = mock_schedd

        launcher = HTCondorLauncher(**params)
        executor = HTCondorExecutor(tmp_path, launcher.params, test_htcondor, launcher)

        job_params = [
            (["db=mysql"], "hydra.sweep.dir", 0, "job_0", {}),
            (["db=postgres"], "hydra.sweep.dir", 1, "job_1", {}),
        ]

        jobs, _ = executor.map_array(job_params)

        # Check pickle files were created
        for i in range(2):
            job_dir = tmp_path / f"job_{i}"
            assert job_dir.exists()

            job_pickle = job_dir / "job.pkl"
            assert job_pickle.exists()

            # Verify pickle contents
            with open(job_pickle, "rb") as f:
                data = cloudpickle.load(f)

            assert "launcher" in data
            assert "args" in data
            assert data["args"][2] == i  # job_idx

    def test_executor_returns_htcondor_jobs(self, tmp_path: Path) -> None:
        """Test that map_array returns HTCondorJob instances."""
        params = {"request_memory": "4000"}

        test_htcondor = MagicMock()
        mock_schedd = MagicMock()
        mock_submit_result = MagicMock()
        mock_submit_result.cluster.return_value = 99999
        mock_schedd.submit.return_value = mock_submit_result
        test_htcondor.Schedd.return_value = mock_schedd

        launcher = HTCondorLauncher(**params)
        executor = HTCondorExecutor(tmp_path, launcher.params, test_htcondor, launcher)

        job_params = [
            (["param=1"], "hydra.sweep.dir", 0, "job_0", {}),
            (["param=2"], "hydra.sweep.dir", 1, "job_1", {}),
        ]

        jobs, _ = executor.map_array(job_params)

        assert len(jobs) == 2
        assert all(type(job).__name__ == "HTCondorJob" for job in jobs)
        assert all(job.cluster_id == 99999 for job in jobs)

    def test_executor_configures_file_transfer(self, tmp_path: Path) -> None:
        """Test that executor configures proper file transfer for HTCondor."""
        params = {"request_memory": "4000"}

        test_htcondor = MagicMock()
        mock_schedd = MagicMock()
        mock_submit_result = MagicMock()
        mock_submit_result.cluster.return_value = 12345
        mock_schedd.submit.return_value = mock_submit_result
        test_htcondor.Schedd.return_value = mock_schedd

        launcher = HTCondorLauncher(**params)
        executor = HTCondorExecutor(tmp_path, launcher.params, test_htcondor, launcher)

        job_params = [
            (["db=mysql"], "hydra.sweep.dir", 0, "job_0", {}),
        ]

        executor.map_array(job_params)

        submit_call = test_htcondor.Submit.call_args
        submit_dict = submit_call[0][0]

        # Check file transfer configuration
        assert submit_dict["should_transfer_files"] == "YES"
        assert "transfer_input_files" in submit_dict
        assert "job.pkl" in submit_dict["transfer_input_files"]
        assert "htcondor_runner.py" in submit_dict["transfer_input_files"]
        assert submit_dict["when_to_transfer_output"] == "ON_EXIT"
        assert "transfer_output_files" in submit_dict

    def test_executor_respects_custom_paths(self, tmp_path: Path) -> None:
        """Test that user-specified submit parameters override defaults when allowed."""
        params = {
            "executable": "/bin/bash",
            "error": "outputs/custom.err",
            "output": "outputs/custom.out",
            "log": "outputs/custom.log",
            "transfer_input_files": "shared.dat",
            "transfer_output_files": "metrics.json",
            "transfer_output_remaps": '"metrics.json=metrics.json"',
        }

        test_htcondor = MagicMock()
        mock_schedd = MagicMock()
        mock_submit_result = MagicMock()
        mock_submit_result.cluster.return_value = 123
        mock_schedd.submit.return_value = mock_submit_result
        test_htcondor.Schedd.return_value = mock_schedd

        launcher = HTCondorLauncher(**params)
        executor = HTCondorExecutor(tmp_path, launcher.params, test_htcondor, launcher)

        executor.map_array(
            [
                (["db=mysql"], "hydra.sweep.dir", 0, "job_0", {}),
            ]
        )

        submit_call = test_htcondor.Submit.call_args
        submit_dict = submit_call[0][0]

        expected_job_dir = (tmp_path / "job_0").resolve()
        assert submit_dict["executable"] == "/bin/bash"
        assert submit_dict["error"] == str(expected_job_dir / "outputs/custom.err")
        assert submit_dict["output"] == str(expected_job_dir / "outputs/custom.out")
        assert submit_dict["log"] == str(expected_job_dir / "outputs/custom.log")
        assert "shared.dat" in submit_dict["transfer_input_files"]
        assert "job.pkl" in submit_dict["transfer_input_files"]
        assert "htcondor_runner.py" in submit_dict["transfer_input_files"]
        assert submit_dict["transfer_output_files"].startswith("metrics.json")
        assert "job.result.pkl" in submit_dict["transfer_output_files"]
        assert submit_dict["transfer_output_remaps"].startswith('"metrics.json=metrics.json"')
        assert "job.result.pkl" in submit_dict["transfer_output_remaps"]

    def test_executor_skips_none_requirements(self, tmp_path: Path) -> None:
        """None requirements should not emit invalid constraint."""
        params = {"request_memory": "4000", "requirements": None}

        test_htcondor = MagicMock()
        mock_schedd = MagicMock()
        mock_submit_result = MagicMock()
        mock_submit_result.cluster.return_value = 123
        mock_schedd.submit.return_value = mock_submit_result
        test_htcondor.Schedd.return_value = mock_schedd

        launcher = HTCondorLauncher(**params)
        executor = HTCondorExecutor(tmp_path, launcher.params, test_htcondor, launcher)
        executor.map_array(
            [
                (["db=mysql"], "hydra.sweep.dir", 0, "job_0", {}),
            ]
        )

        submit_call = test_htcondor.Submit.call_args
        submit_dict = submit_call[0][0]
        assert "requirements" not in submit_dict

    def test_executor_preserves_absolute_output_paths(self, tmp_path: Path) -> None:
        """Absolute paths should be respected as-is."""
        abs_error = str((tmp_path / "custom.err").resolve())
        abs_output = str((tmp_path / "custom.out").resolve())
        abs_log = str((tmp_path / "custom.log").resolve())
        params = {
            "request_memory": "4000",
            "error": abs_error,
            "output": abs_output,
            "log": abs_log,
        }

        test_htcondor = MagicMock()
        mock_schedd = MagicMock()
        mock_submit_result = MagicMock()
        mock_submit_result.cluster.return_value = 123
        mock_schedd.submit.return_value = mock_submit_result
        test_htcondor.Schedd.return_value = mock_schedd

        launcher = HTCondorLauncher(**params)
        executor = HTCondorExecutor(tmp_path, launcher.params, test_htcondor, launcher)
        executor.map_array(
            [
                (["db=mysql"], "hydra.sweep.dir", 0, "job_0", {}),
            ]
        )

        submit_call = test_htcondor.Submit.call_args
        submit_dict = submit_call[0][0]
        assert submit_dict["error"] == abs_error
        assert submit_dict["output"] == abs_output
        assert submit_dict["log"] == abs_log


class TestHTCondorJob:
    """Test HTCondorJob result collection."""

    def test_job_init(self, tmp_path: Path) -> None:
        """Test job initialization."""
        log_file = tmp_path / "job.log"
        output_file = tmp_path / "job.out"
        error_file = tmp_path / "job.err"
        result_pickle = tmp_path / "job.result.pkl"

        job = HTCondorJob(
            cluster_id=12345,
            job_id=0,
            htcondor_module=mock_htcondor,
            log_file=str(log_file),
            output_file=str(output_file),
            error_file=str(error_file),
            result_pickle=str(result_pickle),
        )

        assert job.cluster_id == 12345
        assert job.job_id == 0
        assert job.log_file == log_file
        assert job.output_file == output_file
        assert job.error_file == error_file
        assert job.result_pickle == result_pickle

    def test_job_result_from_pickle_success(self, tmp_path: Path) -> None:
        """Test result collection from successful pickle result."""
        result_pickle = tmp_path / "job.result.pkl"

        # Create a successful result pickle
        expected_result = JobReturn()
        expected_result.status = JobStatus.COMPLETED
        expected_result.return_value = {"answer": 42}

        with open(result_pickle, "wb") as f:
            cloudpickle.dump({"status": "success", "result": expected_result}, f)

        mock_schedd = MagicMock()
        mock_schedd.query.return_value = []  # Job completed
        mock_htcondor.Schedd.return_value = mock_schedd

        job = HTCondorJob(
            cluster_id=12345,
            job_id=0,
            htcondor_module=mock_htcondor,
            log_file=str(tmp_path / "job.log"),
            output_file=str(tmp_path / "job.out"),
            error_file=str(tmp_path / "job.err"),
            result_pickle=str(result_pickle),
        )

        result = job.result()
        assert result.status == JobStatus.COMPLETED
        assert result.return_value == {"answer": 42}

    def test_job_result_from_pickle_error(self, tmp_path: Path) -> None:
        """Test result collection from error pickle result."""
        result_pickle = tmp_path / "job.result.pkl"

        # Create an error result pickle
        with open(result_pickle, "wb") as f:
            cloudpickle.dump(
                {
                    "status": "error",
                    "exception": ValueError("Something went wrong"),
                    "traceback": "Traceback...",
                },
                f,
            )

        mock_schedd = MagicMock()
        mock_schedd.query.return_value = []  # Job completed
        mock_htcondor.Schedd.return_value = mock_schedd

        job = HTCondorJob(
            cluster_id=12345,
            job_id=0,
            htcondor_module=mock_htcondor,
            log_file=str(tmp_path / "job.log"),
            output_file=str(tmp_path / "job.out"),
            error_file=str(tmp_path / "job.err"),
            result_pickle=str(result_pickle),
        )

        result = job.result()
        assert result.status == JobStatus.FAILED
        assert isinstance(result.exception, ValueError)
        assert "Something went wrong" in str(result.exception)

    def test_job_result_no_pickle_file(self, tmp_path: Path) -> None:
        """Test result when pickle file doesn't exist."""
        mock_schedd = MagicMock()
        mock_schedd.query.return_value = []  # Job completed
        mock_htcondor.Schedd.return_value = mock_schedd

        job = HTCondorJob(
            cluster_id=12345,
            job_id=0,
            htcondor_module=mock_htcondor,
            log_file=str(tmp_path / "job.log"),
            output_file=str(tmp_path / "job.out"),
            error_file=str(tmp_path / "job.err"),
            result_pickle=str(tmp_path / "nonexistent.pkl"),
        )

        result = job.result()
        assert result.status == JobStatus.FAILED
        assert "no result file found" in str(result.exception)

    def test_job_result_held(self, tmp_path: Path) -> None:
        """Test result collection for held job."""
        mock_schedd = MagicMock()
        mock_schedd.query.return_value = [
            {"JobStatus": 5, "HoldReason": "Memory limit exceeded"}
        ]
        mock_htcondor.Schedd.return_value = mock_schedd

        job = HTCondorJob(
            cluster_id=12345,
            job_id=0,
            htcondor_module=mock_htcondor,
            log_file=str(tmp_path / "job.log"),
            output_file=str(tmp_path / "job.out"),
            error_file=str(tmp_path / "job.err"),
            result_pickle=str(tmp_path / "job.result.pkl"),
        )

        result = job.result()
        assert result.status == JobStatus.FAILED
        assert "Memory limit exceeded" in str(result.exception)

    def test_job_result_timeout(self, tmp_path: Path) -> None:
        """Test result collection with timeout."""
        mock_schedd = MagicMock()
        mock_schedd.query.return_value = [{"JobStatus": 2}]  # Job always running
        mock_htcondor.Schedd.return_value = mock_schedd

        job = HTCondorJob(
            cluster_id=12345,
            job_id=0,
            htcondor_module=mock_htcondor,
            log_file=str(tmp_path / "job.log"),
            output_file=str(tmp_path / "job.out"),
            error_file=str(tmp_path / "job.err"),
            result_pickle=str(tmp_path / "job.result.pkl"),
        )

        result = job.result(timeout=0.1)
        assert result.status == JobStatus.FAILED
        assert isinstance(result.exception, TimeoutError)


class TestRunnerScript:
    """Test the runner script functionality."""

    def test_runner_script_is_valid_python(self) -> None:
        """Test that the runner script is valid Python code."""
        # This will raise SyntaxError if the script is invalid
        compile(RUNNER_SCRIPT, "<runner_script>", "exec")

    def test_runner_script_executes_job(self, tmp_path: Path) -> None:
        """Test that the runner script can execute a pickled job."""
        # Create a simple callable for testing (using lambda which cloudpickle handles)
        simple_task = lambda x: x * 2  # noqa: E731

        # Create job pickle
        job_pickle = tmp_path / "test_job.pkl"
        job_data = {
            "launcher": simple_task,
            "args": (21,),
        }
        with open(job_pickle, "wb") as f:
            cloudpickle.dump(job_data, f)

        # Write runner script
        runner_path = tmp_path / "runner.py"
        runner_path.write_text(RUNNER_SCRIPT)

        # Execute runner script
        import subprocess

        result = subprocess.run(
            [sys.executable, str(runner_path), str(job_pickle)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Runner failed: {result.stderr}"

        # Check result pickle was created
        result_pickle = job_pickle.with_suffix(".result.pkl")
        assert result_pickle.exists()

        with open(result_pickle, "rb") as f:
            result_data = cloudpickle.load(f)

        assert result_data["status"] == "success"
        assert result_data["result"] == 42

    def test_runner_script_handles_errors(self, tmp_path: Path) -> None:
        """Test that the runner script properly handles job errors."""
        # Use a lambda that raises an error (cloudpickle can handle this)
        failing_task = lambda: (_ for _ in ()).throw(ValueError("Intentional failure"))  # noqa: E731

        # Create job pickle
        job_pickle = tmp_path / "failing_job.pkl"
        job_data = {
            "launcher": failing_task,
            "args": (),
        }
        with open(job_pickle, "wb") as f:
            cloudpickle.dump(job_data, f)

        # Write runner script
        runner_path = tmp_path / "runner.py"
        runner_path.write_text(RUNNER_SCRIPT)

        # Execute runner script
        import subprocess

        result = subprocess.run(
            [sys.executable, str(runner_path), str(job_pickle)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1  # Should exit with error

        # Check result pickle was created with error info
        result_pickle = job_pickle.with_suffix(".result.pkl")
        assert result_pickle.exists()

        with open(result_pickle, "rb") as f:
            result_data = cloudpickle.load(f)

        assert result_data["status"] == "error"
        assert isinstance(result_data["exception"], ValueError)
        assert "Intentional failure" in str(result_data["exception"])
        assert "traceback" in result_data


class TestHTCondorQueueConf:
    """Test configuration dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        conf = HTCondorQueueConf()

        assert conf.request_memory == "4000"
        assert conf.request_cpus == "1"
        assert conf.request_gpus == "0"
        assert conf.MaxTime == 28800  # 8 hours in seconds
        assert conf.requirements is None
        assert conf.htcondor_folder == "${hydra.sweep.dir}/.htcondor"
        assert conf.priority is None
        assert conf.wait_for_jobs is False
        assert (
            conf.submission_cache_file
            == "${hydra.sweep.dir}/.htcondor/submitted_jobs.json"
        )

    def test_target_class(self) -> None:
        """Test that _target_ points to correct class."""
        conf = HTCondorQueueConf()
        assert (
            conf._target_
            == "hydra_plugins.hydra_htcondor_launcher.htcondor_launcher.HTCondorLauncher"
        )
