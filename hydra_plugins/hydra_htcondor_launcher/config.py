# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class HTCondorQueueConf:
    """HTCondor launcher configuration."""

    _target_: str = (
        "hydra_plugins.hydra_htcondor_launcher.htcondor_launcher.HTCondorLauncher"
    )

    # HTCondor resource requests
    request_memory: str = "4000"
    request_cpus: str = "1"
    request_gpus: str = "0"

    # Optional job constraints (e.g., "TARGET.CUDAGlobalMemoryMb > 40000")
    requirements: Optional[str] = None

    # Maximum job runtime in seconds (default: 8 hours)
    MaxTime: int = 28800

    # HTCondor working directory
    htcondor_folder: str = "${hydra.sweep.dir}/.htcondor"

    # Development helper – run jobs locally without HTCondor
    use_local_mode: bool = False


ConfigStore.instance().store(
    group="hydra/launcher",
    name="htcondor",
    node=HTCondorQueueConf(),
    provider="htcondor_launcher",
)
