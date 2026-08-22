import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

# Python 3.11+ has built-in tomllib; older versions use the tomli shim.
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CI_VERSIONS_FILE = os.path.join(SCRIPT_DIR, "ci_versions.json")

REPO_URL = "https://gitcode.com/Ascend/MindSpeed-MM.git"
FLA_REPO_URL = "https://github.com/flashserve/flash-linear-attention-npu.git"
TRITON_ASCEND_INDEX = "https://triton-ascend.osinfra.cn/pypi/simple"

DECORD_SRC = "/opt/decord_src"
PRIVATE_CANN_ROOT = "/workspace/cann"
CANN_INSTALLERS_DIR = "/tmp/cann_installers"
CANN_VERSION = os.environ.get("CANN_VERSION", "9.1.0")
CANN_DOWNLOAD_BASE = (
    "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/"
    f"CANN/CANN%20{CANN_VERSION}"
)

MANAGED_CONFIG_KEYS = {"torch", "torch_npu", "python", "fla"}
SAFE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# Packages installed explicitly by this script must not be reinstalled from a
# branch's pyproject.toml, which could silently replace the pinned versions.
EXCLUDED_DEPS = {
    "torch",
    "torchvision",
    "torchdata",
    "torchaudio",
    "torch_npu",
    "torch-npu",
}


def run(cmd: str, check: bool = True, cwd: str | None = None,
        env: dict | None = None) -> subprocess.CompletedProcess:
    """Print and execute a shell command."""
    print(f"[RUN] {cmd}")
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd, env=env)


def run_args(args: list[str], check: bool = True, cwd: str | None = None,
             env: dict | None = None) -> subprocess.CompletedProcess:
    """Print and execute an argument list without shell interpretation."""
    print(f"[RUN] {shlex.join(args)}")
    return subprocess.run(args, check=check, cwd=cwd, env=env)


def package_base_name(spec: str) -> str:
    """Return the normalized package name from a PEP 508 dependency string."""
    name = spec.strip().split("[", 1)[0]
    match = re.match(r"^[A-Za-z0-9_.-]+", name)
    return match.group(0).lower() if match else name.lower()


def filter_out_excluded(dependencies: list[str]) -> list[str]:
    """Drop dependencies whose versions are managed explicitly."""
    return [
        dependency
        for dependency in dependencies
        if package_base_name(dependency) not in EXCLUDED_DEPS
    ]


def require_safe_string(value: object, label: str) -> str:
    """Validate a version-like value before using it in paths or commands."""
    if not isinstance(value, str) or not value or not SAFE_VALUE_RE.fullmatch(value):
        raise ValueError(
            f"{label} must be a non-empty string containing only letters, "
            "digits, '.', '_' or '-'"
        )
    return value


def validate_fla_config(env_name: str, value: object) -> dict[str, str] | None:
    """Return a validated FLA config, or None when the environment opts out."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{env_name}.fla must be an object")

    expected_keys = {"version", "triton_ascend"}
    if set(value) != expected_keys:
        raise ValueError(
            f"{env_name}.fla must contain exactly: "
            f"{', '.join(sorted(expected_keys))}"
        )

    return {
        "version": require_safe_string(value["version"], f"{env_name}.fla.version"),
        "triton_ascend": require_safe_string(
            value["triton_ascend"], f"{env_name}.fla.triton_ascend"
        ),
    }


def validate_versions(versions: object) -> dict[str, dict]:
    """Validate the top-level CI configuration before making any changes."""
    if not isinstance(versions, dict) or not versions:
        raise ValueError("ci_versions.json must contain at least one environment")

    for env_name, cfg in versions.items():
        require_safe_string(env_name, "environment name")
        if not env_name.startswith("ci_"):
            raise ValueError(f"environment name must start with 'ci_': {env_name}")
        if not isinstance(cfg, dict):
            raise ValueError(f"{env_name} configuration must be an object")

        for key in ("torch", "torch_npu", "python"):
            require_safe_string(cfg.get(key), f"{env_name}.{key}")
        validate_fla_config(env_name, cfg.get("fla"))

        for package, version in cfg.items():
            if package in MANAGED_CONFIG_KEYS:
                continue
            require_safe_string(package, f"{env_name} package name")
            require_safe_string(version, f"{env_name}.{package}")

    return versions


def build_decord_binding(env_name: str) -> None:
    """Compile the Decord Python binding into one Conda environment."""
    python_dir = os.path.join(DECORD_SRC, "python")
    if not os.path.isdir(python_dir):
        raise FileNotFoundError(f"Decord source not found at {DECORD_SRC}")

    print(f"[INFO] Building Decord Python binding for {env_name} ...")
    run_args([
        "conda", "run", "--no-capture-output", "-n", env_name,
        "pip", "install", "--no-cache-dir", "pybind11",
    ])
    run_args([
        "conda", "run", "--no-capture-output", "-n", env_name,
        "pip", "install", "--no-cache-dir", ".",
    ], cwd=python_dir)


def ensure_cann_installers() -> tuple[str, str]:
    """Download the CANN Toolkit and 910b ops run packages once."""
    os.makedirs(CANN_INSTALLERS_DIR, exist_ok=True)
    filenames = (
        f"Ascend-cann-toolkit_{CANN_VERSION}_linux-aarch64.run",
        f"Ascend-cann-910b-ops_{CANN_VERSION}_linux-aarch64.run",
    )

    paths = []
    for filename in filenames:
        path = os.path.join(CANN_INSTALLERS_DIR, filename)
        if not os.path.isfile(path):
            run_args([
                "wget", "--tries=3", "--timeout=60", "-O", path,
                f"{CANN_DOWNLOAD_BASE}/{filename}",
            ])
        paths.append(path)
    return paths[0], paths[1]


def install_private_cann(env_name: str) -> str:
    """Install a private CANN Toolkit and 910b ops tree for one environment."""
    install_path = os.path.join(PRIVATE_CANN_ROOT, env_name)
    if os.path.lexists(install_path):
        shutil.rmtree(install_path)

    toolkit_installer, ops_installer = ensure_cann_installers()
    os.makedirs(PRIVATE_CANN_ROOT, exist_ok=True)
    run_args([
        "bash", toolkit_installer,
        f"--install-path={install_path}", "--full", "--quiet",
    ])
    run_args([
        "bash", ops_installer,
        f"--install-path={install_path}", "--install", "--quiet",
    ])

    set_env_path = os.path.join(install_path, "cann", "set_env.sh")
    if not os.path.isfile(set_env_path):
        raise FileNotFoundError(
            f"Private CANN installation for {env_name} did not create {set_env_path}"
        )
    print(f"[INFO] Private CANN for {env_name}: {set_env_path}")
    return set_env_path


def run_with_cann(env_name: str, set_env_path: str, args: list[str],
                  cwd: str | None = None,
                  extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run a command in a Conda environment after sourcing its private CANN."""
    command = [
        "conda", "run", "--no-capture-output", "-n", env_name, *args,
    ]
    shell_command = f"source {shlex.quote(set_env_path)} && {shlex.join(command)}"
    command_env = os.environ.copy()
    if extra_env:
        command_env.update(extra_env)
    print(f"[RUN] {shell_command}")
    return subprocess.run(
        ["bash", "-lc", shell_command],
        check=True,
        cwd=cwd,
        env=command_env,
    )


def install_fla(env_name: str, fla_config: dict[str, str],
                constraint_path: str) -> None:
    """Install Triton-Ascend and build FLA using the Qwen3.5 README flow."""
    fla_version = fla_config["version"]
    triton_version = fla_config["triton_ascend"]
    set_env_path = install_private_cann(env_name)
    source_dir = f"/tmp/flash_linear_attention_npu_{env_name}"

    if os.path.lexists(source_dir):
        shutil.rmtree(source_dir)

    try:
        run_with_cann(env_name, set_env_path, [
            "python", "-m", "pip", "install", "--no-cache-dir",
            "--constraint", constraint_path,
            f"triton-ascend=={triton_version}",
            f"--extra-index-url={TRITON_ASCEND_INDEX}",
        ])
        run_args([
            "git", "clone", "--depth", "1", "--branch", f"v{fla_version}",
            FLA_REPO_URL, source_dir,
        ])
        run_with_cann(env_name, set_env_path, [
            "python", "-m", "pip", "install", "--no-cache-dir",
            "--constraint", constraint_path,
            "-r", "requirements.txt",
        ], cwd=source_dir)
        run_with_cann(env_name, set_env_path, [
            "python", "scripts/check_npu_env.py", "--build-only",
        ], cwd=source_dir, extra_env={"FLA_NPU_SOC": "ascend910b"})
        run_with_cann(env_name, set_env_path, [
            "python", "-m", "pip", "wheel", "--no-build-isolation",
            "--no-deps", ".", "-w", "dist",
        ], cwd=source_dir, extra_env={"FLA_NPU_SOC": "ascend910b"})

        wheels = glob.glob(os.path.join(
            source_dir, "dist", "flash_linear_attention_npu-*.whl"
        ))
        if len(wheels) != 1:
            raise RuntimeError(
                f"Expected one FLA wheel for {env_name}, found {len(wheels)}"
            )
        run_with_cann(env_name, set_env_path, [
            "python", "-m", "pip", "install", "--force-reinstall",
            "--no-deps", wheels[0],
        ])
        run_with_cann(env_name, set_env_path, [
            "python", "scripts/check_packaged_wheel_api.py",
        ], cwd=source_dir)

        verification = (
            "import importlib.metadata as m; import fla_npu, triton; "
            f"assert m.version('triton-ascend') == '{triton_version}'; "
            "installed = m.version('flash-linear-attention-npu').split('+')[0]; "
            f"assert installed == '{fla_version}', installed; "
            "print('FLA verification passed')"
        )
        run_with_cann(env_name, set_env_path, ["python", "-c", verification])
    finally:
        shutil.rmtree(source_dir, ignore_errors=True)


def main() -> None:
    """Build every Conda environment declared in ci_versions.json."""
    for channel in (
        "https://repo.anaconda.com/pkgs/main",
        "https://repo.anaconda.com/pkgs/r",
    ):
        run_args([
            "conda", "tos", "accept", "--override-channels", "--channel", channel,
        ], check=False)

    with open(CI_VERSIONS_FILE, "r", encoding="utf-8") as config_file:
        versions = validate_versions(json.load(config_file))

    print(
        f"[INFO] Found {len(versions)} CI environment(s) to build: "
        f"{list(versions.keys())}"
    )

    try:
        for env_name, cfg in versions.items():
            branch = env_name.removeprefix("ci_")
            print(f"\n{'=' * 60}")
            print(f"[INFO] Building environment: {env_name} (branch: {branch})")
            print(f"       config: {cfg}")
            print(f"{'=' * 60}")

            python_ver = cfg["python"]
            constraint_path = f"/tmp/{env_name}-constraints.txt"
            with open(constraint_path, "w", encoding="utf-8") as constraints:
                constraints.write(
                    f"torch=={cfg['torch']}\n"
                    f"torch-npu=={cfg['torch_npu']}\n"
                )
            print(f"[INFO] Creating fresh environment with Python {python_ver} ...")
            run_args([
                "conda", "create", "-n", env_name, f"python={python_ver}", "-y",
            ])
            build_decord_binding(env_name)

            run_args([
                "conda", "run", "--no-capture-output", "-n", env_name,
                "pip", "install", "--no-cache-dir",
                "--constraint", constraint_path,
                f"torch=={cfg['torch']}", f"torch-npu=={cfg['torch_npu']}",
            ])

            source_dir = f"/tmp/mindspeed_mm_{branch}"
            shutil.rmtree(source_dir, ignore_errors=True)
            run_args([
                "git", "clone", "--depth", "1", "--branch", branch,
                REPO_URL, source_dir,
            ])
            try:
                pyproject_path = os.path.join(source_dir, "pyproject.toml")
                with open(pyproject_path, "rb") as pyproject_file:
                    data = tomllib.load(pyproject_file)
                dependencies = data.get("project", {}).get("dependencies", [])
                if not dependencies:
                    print(
                        "[WARN] No [project].dependencies found in "
                        f"pyproject.toml for branch {branch}"
                    )

                dependencies = filter_out_excluded(dependencies)
                if dependencies:
                    run_args([
                        "conda", "run", "--no-capture-output", "-n", env_name,
                        "pip", "install", "--no-cache-dir",
                        "--constraint", constraint_path, *dependencies,
                    ])

                extra_packages = {
                    package: version
                    for package, version in cfg.items()
                    if package not in MANAGED_CONFIG_KEYS
                }
                for package, version in extra_packages.items():
                    run_args([
                        "conda", "run", "--no-capture-output", "-n", env_name,
                        "pip", "install", "--no-cache-dir",
                        "--constraint", constraint_path, f"{package}=={version}",
                    ])

                fla_config = validate_fla_config(env_name, cfg.get("fla"))
                if fla_config:
                    install_fla(env_name, fla_config, constraint_path)

                managed_versions = (
                    "import importlib.metadata as m; "
                    f"assert m.version('torch') == '{cfg['torch']}'; "
                    f"assert m.version('torch-npu') == '{cfg['torch_npu']}'; "
                    "print('managed package versions verified')"
                )
                run_args([
                    "conda", "run", "--no-capture-output", "-n", env_name,
                    "python", "-c", managed_versions,
                ])
            finally:
                shutil.rmtree(source_dir, ignore_errors=True)
                if os.path.exists(constraint_path):
                    os.remove(constraint_path)

            print(f"[INFO] Environment {env_name} build completed.")
    finally:
        shutil.rmtree(CANN_INSTALLERS_DIR, ignore_errors=True)
        shutil.rmtree("/root/.cache/pip", ignore_errors=True)
        run_args(["conda", "clean", "-afy"], check=False)

    print("\n[INFO] All CI environments built successfully.")


if __name__ == "__main__":
    main()
