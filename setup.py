# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# type: ignore
from setuptools import find_namespace_packages, setup

with open("README.md") as fh:
    LONG_DESC = fh.read()
    setup(
        name="hydra-htcondor-launcher",
        version="0.1.0",
        author="Your Name",
        author_email="your@email.com",
        description="HTCondor Launcher plugin for Hydra",
        long_description=LONG_DESC,
        long_description_content_type="text/markdown",
        url="https://github.com/facebookresearch/hydra/",
        packages=find_namespace_packages(include=["hydra_plugins.*"]),
        classifiers=[
            # Feel free to use another license.
            "License :: OSI Approved :: MIT License",
            # Hydra uses Python version and Operating system to determine
            # In which environments to test this plugin
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Operating System :: OS Independent",
        ],
        install_requires=[
            "hydra-core>=1.3.2",
            "cloudpickle>=3.0.0",
        ],
        # If this plugin is providing configuration files, be sure to include them in the package.
        # See MANIFEST.in.
        # For configurations to be discoverable at runtime, they should also be added to the search path.
        include_package_data=True,
    )
