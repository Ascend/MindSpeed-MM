import json
import os
import re
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
# File format:
#   {
#     "ci_<env>": {
#       "torch": "<ver>", "torch_npu": "<ver>", "python": "<x.y>",
#       "<extra_pkg>": "<version>", ...
#     }, ...
#   }
CI_VERSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ci_versions.json")

# MindSpeed-MM repository clone URL
REPO_URL = "https://gitcode.com/Ascend/MindSpeed-MM.git"

# Directory where the decord source is prepared by the CI image build
# (Dockerfile.ci). The decord C++ core libraries are shared in /usr/local/lib
# (from the dev image); this source is used to compile the per-python binding.
# Absent on x86_64 (decord comes from pip there).
DECORD_SRC = "/opt/decord_src"

# Packages that we manage explicitly (installed by this script at a pinned
# version) and therefore MUST be excluded from the pyproject.toml dependency
# list. In particular, excluding "torch" prevents pip from downgrading the
# pre-installed torch when a release branch still pins an older torch version
# in pyproject.toml (the master 2.7.1 -> 2.10.0 transition).
EXCLUDED_DEPS = {
    "torch",
    "torchvision",
    "torchdata",
    "torchaudio",
    "torch_npu",
    "torch-npu",
}


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Print and execute a shell command."""
    print(f"[RUN] {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


def run_args(args: list, check: bool = True) -> subprocess.CompletedProcess:
    """Print and execute a command given as an argument list, without shell
    interpretation.

    Use this instead of run() when an argument may contain shell
    metacharacters (e.g. PEP 508 dependency specifiers such as
    'pkg>=1.0', 'pkg[extra]==2.0', or markers with spaces), so that each
    specifier is passed to the target program verbatim.
    """
    print(f"[RUN] {' '.join(args)}")
    return subprocess.run(args, check=check)


def package_base_name(spec: str) -> str:
    """Return the bare package name of a PEP 508 specifier, ignoring version
    constraints and extras. E.g. 'torch==2.7.1' -> 'torch', 'pkg[extra]>=2' -> 'pkg'."""
    name = spec.strip()
    # Strip extras: pkg[extra] -> pkg
    name = name.split("[", 1)[0]
    m = re.match(r"^[A-Za-z0-9_.-]+", name)
    return m.group(0).lower() if m else name.lower()


def filter_out_excluded(dependencies: list) -> list:
    """Drop any dependency whose package name is in EXCLUDED_DEPS."""
    return [d for d in dependencies if package_base_name(d) not in EXCLUDED_DEPS]


def get_base_python() -> tuple:
    """Return the (major, minor) python version of the conda base environment."""
    out = subprocess.check_output(
        ["conda", "run", "-n", "base", "python", "-c",
         "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
        text=True,
    ).strip()
    major, minor = out.split(".")
    return (int(major), int(minor))


def build_decord_binding(env_name: str):
    """Compile the decord python binding from DECORD_SRC into the given env.

    The decord C++ core libraries are shared system-wide in /usr/local/lib
    (installed by the dev image), so only the per-python pybind11 binding needs
    to be compiled here. This is fully independent of the dev image (no reliance
    on any CI-specific artifact baked into it) and works for any python version
    (3.12, 3.13, ...) simply because each fresh env drives its own build.
    """
    python_dir = os.path.join(DECORD_SRC, "python")
    if not os.path.isdir(python_dir):
        print(f"[WARN] decord source not found at {DECORD_SRC}; "
              f"cannot build binding for {env_name}")
        return
    print(f"[INFO] Building decord python binding for {env_name} "
          f"from {python_dir} ...")
    run_args(["conda", "run", "--no-capture-output", "-n", env_name,
              "pip", "install", "pybind11"])
    run(f"cd {python_dir} && conda run --no-capture-output -n {env_name} pip install .")


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
    with open(CI_VERSIONS_FILE, "r", encoding="utf-8") as f:
        versions = json.load(f)

    print(f"[INFO] Found {len(versions)} CI environment(s) to build: {list(versions.keys())}")

    # 1b. Determine the base env python so we can clone it when it matches.
    base_python = get_base_python()
    print(f"[INFO] Conda base python version: {base_python[0]}.{base_python[1]}")

    # 2. Build each CI environment one by one
    for env_name, cfg in versions.items():
        # 2a. Extract branch name from environment name: "ci_26.0.0" -> "26.0.0",
        #     "ci_master" -> "master".
        branch = env_name.removeprefix("ci_")
        print(f"\n{'=' * 60}")
        print(f"[INFO] Building environment: {env_name} (branch: {branch})")
        print(f"       config: {cfg}")
        print(f"{'=' * 60}")

        torch_ver = cfg.get("torch")
        torch_npu_ver = cfg.get("torch_npu")
        python_ver = cfg.get("python")
        extra_pkgs = {k: v for k, v in cfg.items() if k not in ("torch", "torch_npu", "python")}

        # 2b. Create the conda environment with the required python version.
        #     - If it matches the base env python, clone base to reuse already
        #       installed packages (torch 2.7.1, torch_npu, decord, ...) and save time/disk.
        #     - Otherwise create a fresh environment pinned to the required python.
        if python_ver:
            py_major, py_minor = (int(x) for x in python_ver.split("."))
            cloned = (py_major, py_minor) == base_python
            if cloned:
                print(f"[INFO] Cloning base env (python {python_ver}) ...")
                run(f"conda create -n {env_name} --clone base -y")
            else:
                print(f"[INFO] Creating fresh env with python {python_ver} ...")
                run(f"conda create -n {env_name} python={python_ver} -y")
                # A fresh env (e.g. ci_master on python 3.12) does NOT inherit
                # decord from base. The C++ core libs are shared in /usr/local/lib;
                # compile the per-python binding from source (independent of the
                # dev image, works for any python version).
                build_decord_binding(env_name)

        # 2c. Explicitly install the pinned torch / torch_npu versions first so
        #     that later dependency installs never downgrade them.
        if torch_ver:
            run_args(["conda", "run", "-n", env_name, "pip", "install", f"torch=={torch_ver}"])
        if torch_npu_ver:
            run_args(["conda", "run", "-n", env_name, "pip", "install", f"torch-npu=={torch_npu_ver}"])

        # 2d. Shallow-clone the corresponding branch to a temp directory and
        #     parse [project].dependencies from pyproject.toml.
        tmpdir = f"/tmp/mindspeed_mm_{branch}"
        if os.path.exists(tmpdir):
            run(f"rm -rf {tmpdir}")
        run(f"git clone {REPO_URL} -b {branch} {tmpdir} --depth 1")

        pyproject_path = os.path.join(tmpdir, "pyproject.toml")
        with open(pyproject_path, "rb") as f_toml:
            data = tomllib.load(f_toml)
        dependencies = data.get("project", {}).get("dependencies", [])

        if not dependencies:
            print(f"[WARN] No [project].dependencies found in pyproject.toml for branch {branch}")

        # 2e. Install base dependencies, EXCLUDING the packages this script
        #     manages explicitly (torch family). This keeps the pre-installed
        #     torch version untouched during the master 2.7.1 -> 2.10.0 transition.
        deps_to_install = filter_out_excluded(dependencies)
        print(f"[INFO] Installing {len(deps_to_install)} dependency(ies) "
              f"(excluded {len(dependencies) - len(deps_to_install)} managed pkg(s))...")
        if deps_to_install:
            run_args(["conda", "run", "-n", env_name, "pip", "install", *deps_to_install])

        # 2f. Install extra dependencies declared in the JSON (with pinned versions).
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
