import json
import os
import subprocess
import sys

# Python 3.11+ has built-in tomllib; older versions use the tomli shim
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        print(
            "ERROR: The 'tomli' package is required to parse pyproject.toml. "
            "Please run: pip install tomli",
            file=sys.stderr,
        )
        sys.exit(1)

# Path to the CI version configuration file (ci_versions.json, same dir as this script).
# File format: { "ci_<version>": { "<extra_pkg>": "<version>", ... }, ... }
CI_VERSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ci_versions.json")

# MindSpeed-MM repository clone URL
REPO_URL = "https://gitcode.com/Ascend/MindSpeed-MM.git"


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Print and execute a shell command.

    Args:
        cmd:   The shell command string to execute.
        check: If True (default), raise an exception on non-zero exit code.

    Returns:
        subprocess.CompletedProcess containing stdout, stderr, and returncode.
    """
    print(f"[RUN] {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def run_args(args: list, check: bool = True) -> subprocess.CompletedProcess:
    """Print and execute a command given as an argument list, without shell
    interpretation.

    Use this instead of run() when an argument may contain shell
    metacharacters (e.g. PEP 508 dependency specifiers such as
    'pkg>=1.0', 'pkg[extra]==2.0', or markers with spaces), so that each
    specifier is passed to the target program verbatim.

    Args:
        args:  The command and its arguments as a list.
        check: If True (default), raise an exception on non-zero exit code.

    Returns:
        subprocess.CompletedProcess containing stdout, stderr, and returncode.
    """
    print(f"[RUN] {' '.join(args)}")
    return subprocess.run(args, check=check)


def main():
    """Main flow: iterate over every environment in ci_versions.json and build it."""
    # 0. Auto-accept Anaconda ToS to avoid errors in non-interactive environments.
    tos_channels = [
        "https://repo.anaconda.com/pkgs/main",
        "https://repo.anaconda.com/pkgs/r",
    ]
    for ch in tos_channels:
        run(f"conda tos accept --override-channels --channel {ch}", check=False)

    # 1. Read the CI version configuration file.
    #    Example format: {"ci_26.0.0": {"numpy": "1.26.0"}, "ci_26.1.0": {}}
    with open(CI_VERSIONS_FILE, "r", encoding="utf-8") as f:
        versions = json.load(f)

    print(f"[INFO] Found {len(versions)} CI environment(s) to build: {list(versions.keys())}")

    # 2. Build each CI environment one by one
    for env_name, extra_pkgs in versions.items():
        # 2a. Extract branch name from environment name: "ci_26.0.0" -> "26.0.0"
        branch = env_name.removeprefix("ci_")
        print(f"\n{'=' * 60}")
        print(f"[INFO] Building environment: {env_name} (branch: {branch})")
        print(f"{'=' * 60}")

        # 2b. Create a new conda environment by cloning from base.
        #     --clone base: copies all packages already installed in base,
        #     avoiding the need to reinstall large dependencies like PyTorch from scratch.
        run(f"conda create -n {env_name} --clone base -y")

        # 2c. Shallow-clone the corresponding branch to a temp directory.
        #     --depth 1: fetches only the latest commit to reduce network and disk usage.
        tmpdir = f"/tmp/mindspeed_mm_{branch}"
        # Remove any leftover temp directory from a previous failed build
        if os.path.exists(tmpdir):
            run(f"rm -rf {tmpdir}")
        run(f"git clone {REPO_URL} -b {branch} {tmpdir} --depth 1")

        # 2d. Parse [project].dependencies from pyproject.toml to get the
        #     base dependency list declared by that branch (e.g. torch, transformers).
        pyproject_path = os.path.join(tmpdir, "pyproject.toml")
        with open(pyproject_path, "rb") as f_toml:
            data = tomllib.load(f_toml)
        dependencies = data.get("project", {}).get("dependencies", [])

        if not dependencies:
            print(f"[WARN] No [project].dependencies found in pyproject.toml for branch {branch}")

        # 2e. Install base dependencies into the target conda environment.
        #     conda run -n <env> pip install ... ensures pip runs inside the correct env.
        if dependencies:
            print(f"[INFO] Installing base dependencies ({len(dependencies)} package(s))...")
            run_args(["conda", "run", "-n", env_name, "pip", "install", *dependencies])

        # 2f. Install extra dependencies declared in the JSON (with pinned versions).
        #     Format: {"numpy": "1.26.0", "pandas": "2.1.0"}
        if extra_pkgs:
            print(f"[INFO] Installing extra dependencies: {extra_pkgs}")
            for pkg, version in extra_pkgs.items():
                run_args(["conda", "run", "-n", env_name, "pip", "install", f"{pkg}=={version}"])

        # 2g. Remove the temp directory to free disk space
        run(f"rm -rf {tmpdir}")
        print(f"[INFO] Environment {env_name} build completed.")

    print("\n[INFO] All CI environments built successfully.")


if __name__ == "__main__":
    main()
