import importlib.util
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVS_PACKAGE_DIR = REPO_ROOT / "mindspeed_mm" / "fsdp" / "envs"
ENVS_PACKAGE_PATH = ENVS_PACKAGE_DIR / "__init__.py"


def _load_envs_package():
    module_name = "_mindspeed_mm_fsdp_envs_package_test"
    spec = importlib.util.spec_from_file_location(
        module_name,
        ENVS_PACKAGE_PATH,
        submodule_search_locations=[str(ENVS_PACKAGE_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


envs = _load_envs_package()
EnvironmentVariable = envs.EnvironmentVariable
EnvironmentVariableDeprecationWarning = envs.EnvironmentVariableDeprecationWarning
EnvironmentVariableManager = envs.EnvironmentVariableManager


def test_package_facade_preserves_public_interface(monkeypatch):
    envs_package = _load_envs_package()
    monkeypatch.setenv("LOCAL_RANK", "7")

    assert envs_package.LOCAL_RANK == 7
    assert envs_package.get("LOCAL_RANK") == 7
    assert envs_package.EnvironmentVariable is envs_package.envs.EnvironmentVariable
    assert envs_package.environment_variables is envs_package.envs.environment_variables
    assert "LOCAL_RANK" in dir(envs_package)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("true", True),
        ("", False),
        ("1", False),
        ("TRUE", False),
        ("yes", False),
    ],
)
def test_non_megatron_preserves_historical_parsing(monkeypatch, raw_value, expected):
    monkeypatch.setenv("NON_MEGATRON", raw_value)
    assert envs.NON_MEGATRON is expected


def test_non_megatron_defaults_to_false(monkeypatch):
    monkeypatch.delenv("NON_MEGATRON", raising=False)
    assert envs.NON_MEGATRON is False
    assert envs.is_set("NON_MEGATRON") is False


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("1", True),
        ("TRUE", True),
        ("yes", True),
        ("Y", True),
        (" on ", True),
        ("0", False),
        ("False", False),
        ("no", False),
        ("N", False),
        (" off ", False),
    ],
)
def test_bool_conversion(monkeypatch, raw_value, expected):
    monkeypatch.setenv("MM_FORCE_EP_BALANCE", raw_value)
    assert envs.MM_FORCE_EP_BALANCE is expected


def test_invalid_bool_reports_variable_name(monkeypatch):
    monkeypatch.setenv("MM_FORCE_EP_BALANCE", "invalid")
    with pytest.raises(ValueError, match="MM_FORCE_EP_BALANCE"):
        _ = envs.MM_FORCE_EP_BALANCE


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("true", True), ("1", True), ("false", False), ("0", False)],
)
def test_detect_anomaly_conversion(monkeypatch, raw_value, expected):
    monkeypatch.setenv("MM_DETECT_ANOMALY", raw_value)
    assert envs.MM_DETECT_ANOMALY is expected


def test_detect_anomaly_defaults_to_false(monkeypatch):
    monkeypatch.delenv("MM_DETECT_ANOMALY", raising=False)
    assert envs.MM_DETECT_ANOMALY is False


def test_integer_conversion_preserves_launcher_values(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "7")
    assert envs.LOCAL_RANK == 7

    monkeypatch.setenv("LOCAL_RANK", "-1")
    assert envs.LOCAL_RANK == -1


def test_custom_integer_validation(monkeypatch):
    def validate_non_negative(value):
        if value < 0:
            raise ValueError("must be non-negative")

    manager = EnvironmentVariableManager()
    manager.register(
        EnvironmentVariable(
            "MM_COUNT",
            0,
            int,
            "A non-negative count.",
            validator=validate_non_negative,
        )
    )

    monkeypatch.setenv("MM_COUNT", "-1")
    with pytest.raises(ValueError, match="MM_COUNT"):
        manager.get("MM_COUNT")


def test_invalid_integer_reports_variable_name(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "not-an-integer")
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        _ = envs.WORLD_SIZE


def test_float_and_custom_conversion(monkeypatch):
    manager = EnvironmentVariableManager()
    manager.register(
        EnvironmentVariable("MM_RATIO", 1.0, float, "A floating-point ratio.")
    )
    manager.register(
        EnvironmentVariable(
            "MM_RANKS",
            [],
            lambda value: [int(item) for item in value.split(",")],
            "A comma-separated rank list.",
        )
    )
    monkeypatch.setenv("MM_RATIO", "0.75")
    monkeypatch.setenv("MM_RANKS", "0,2,4")
    assert manager.get("MM_RATIO") == 0.75
    assert manager.get("MM_RANKS") == [0, 2, 4]


def test_hf_save_wait_mode_preserves_historical_values(monkeypatch):
    monkeypatch.setenv("HF_SAVE_WAIT_MODE", "custom-mode")
    assert envs.HF_SAVE_WAIT_MODE == "custom-mode"


def test_choice_validation(monkeypatch):
    manager = EnvironmentVariableManager()
    manager.register(
        EnvironmentVariable(
            "MM_MODE",
            "sleep",
            str,
            "A constrained mode.",
            choices=("sleep", "compute"),
        )
    )

    monkeypatch.setenv("MM_MODE", "compute")
    assert manager.get("MM_MODE") == "compute"

    monkeypatch.setenv("MM_MODE", "barrier")
    with pytest.raises(ValueError, match="MM_MODE"):
        manager.get("MM_MODE")


def test_values_are_read_lazily(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "2")
    assert envs.LOCAL_RANK == 2

    monkeypatch.setenv("LOCAL_RANK", "6")
    assert envs.LOCAL_RANK == 6


def test_call_site_default_preserves_legacy_fallback(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert envs.get("LOCAL_RANK") == 0
    assert envs.get("LOCAL_RANK", -1) == -1

    monkeypatch.setenv("LOCAL_RANK", "4")
    assert envs.get("LOCAL_RANK", -1) == 4


def test_required_read_rejects_missing_launcher_value(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.raises(KeyError, match="LOCAL_RANK"):
        envs.get("LOCAL_RANK", required=True)

    monkeypatch.setenv("LOCAL_RANK", "0")
    assert envs.get("LOCAL_RANK", required=True) == 0

    with pytest.raises(ValueError, match="default and required"):
        envs.get("LOCAL_RANK", -1, required=True)


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [("-1", 1), ("0", 1), ("1", 1), ("8", 8)],
)
def test_local_world_size_preserves_historical_clamping(
    monkeypatch, raw_value, expected
):
    monkeypatch.setenv("LOCAL_WORLD_SIZE", raw_value)
    assert max(1, envs.LOCAL_WORLD_SIZE) == expected


@pytest.mark.parametrize("name", ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
def test_launcher_boundaries_are_left_to_business_code(monkeypatch, name):
    monkeypatch.setenv(name, "-1")
    assert envs.get(name) == -1


def test_is_set_distinguishes_default_and_explicit_value(monkeypatch):
    monkeypatch.delenv("MM_FORCE_EP_BALANCE", raising=False)
    assert envs.MM_FORCE_EP_BALANCE is False
    assert envs.is_set("MM_FORCE_EP_BALANCE") is False

    monkeypatch.setenv("MM_FORCE_EP_BALANCE", "false")
    assert envs.MM_FORCE_EP_BALANCE is False
    assert envs.is_set("MM_FORCE_EP_BALANCE") is True


def test_definitions_and_descriptions_are_exposed():
    spec = envs.get_spec("LOCAL_RANK")
    assert spec.converter is int
    assert spec.default == 0
    assert "local rank" in envs.descriptions()["LOCAL_RANK"].lower()

    with pytest.raises(TypeError):
        envs.environment_variables["NEW_VARIABLE"] = spec


def test_unknown_variable_is_rejected():
    with pytest.raises(KeyError, match="UNKNOWN_VARIABLE"):
        envs.get("UNKNOWN_VARIABLE")
    with pytest.raises(AttributeError, match="UNKNOWN_VARIABLE"):
        _ = envs.UNKNOWN_VARIABLE


def test_duplicate_registration_is_rejected():
    manager = EnvironmentVariableManager()
    variable = EnvironmentVariable("MM_OPTION", False, bool, "An option.")
    manager.register(variable)
    with pytest.raises(ValueError, match="already registered"):
        manager.register(variable)


@pytest.mark.parametrize(
    "variable",
    [
        EnvironmentVariable("", False, bool, "An option."),
        EnvironmentVariable("MM_OPTION", False, bool, ""),
        EnvironmentVariable(
            "MM_OPTION",
            False,
            bool,
            "An option.",
            deprecated=True,
            replacement="MM_OPTION",
        ),
        EnvironmentVariable(
            "MM_OPTION",
            False,
            bool,
            "An option.",
            replacement="MM_NEW_OPTION",
        ),
        EnvironmentVariable("MM_OPTION", "false", bool, "An option."),
    ],
)
def test_invalid_definition_is_rejected(variable):
    manager = EnvironmentVariableManager()
    with pytest.raises(ValueError):
        manager.register(variable)


def test_missing_replacement_is_rejected():
    manager = EnvironmentVariableManager()
    manager.register(
        EnvironmentVariable(
            "MM_OLD_OPTION",
            False,
            bool,
            "Old option.",
            deprecated=True,
            replacement="MM_NEW_OPTION",
        )
    )
    with pytest.raises(ValueError, match="not registered"):
        manager.validate_registry()


def test_replacement_cycle_is_rejected():
    manager = EnvironmentVariableManager()
    manager.register(
        EnvironmentVariable(
            "MM_OPTION_A",
            False,
            bool,
            "Option A.",
            deprecated=True,
            replacement="MM_OPTION_B",
        )
    )
    manager.register(
        EnvironmentVariable(
            "MM_OPTION_B",
            False,
            bool,
            "Option B.",
            deprecated=True,
            replacement="MM_OPTION_A",
        )
    )
    with pytest.raises(ValueError, match="cycle"):
        manager.validate_registry()


def _manager_with_deprecated_alias():
    manager = EnvironmentVariableManager()
    manager.register(EnvironmentVariable("MM_NEW_OPTION", False, bool, "New option."))
    manager.register(
        EnvironmentVariable(
            "MM_OLD_OPTION",
            False,
            bool,
            "Old option.",
            deprecated=True,
            replacement="MM_NEW_OPTION",
        )
    )
    manager.validate_registry()
    return manager


def test_deprecated_name_warns_and_falls_back(monkeypatch):
    manager = _manager_with_deprecated_alias()
    monkeypatch.setenv("MM_OLD_OPTION", "true")
    monkeypatch.delenv("MM_NEW_OPTION", raising=False)

    with pytest.warns(
        EnvironmentVariableDeprecationWarning,
        match="MM_NEW_OPTION",
    ):
        assert manager.get("MM_NEW_OPTION") is True


def test_new_name_does_not_warn(monkeypatch):
    manager = _manager_with_deprecated_alias()
    monkeypatch.delenv("MM_OLD_OPTION", raising=False)
    monkeypatch.setenv("MM_NEW_OPTION", "true")

    with warnings.catch_warnings(record=True) as warning_records:
        warnings.simplefilter("always")
        assert manager.get("MM_NEW_OPTION") is True
    assert not warning_records


def test_new_name_wins_when_both_names_are_set(monkeypatch):
    manager = _manager_with_deprecated_alias()
    monkeypatch.setenv("MM_OLD_OPTION", "false")
    monkeypatch.setenv("MM_NEW_OPTION", "true")

    with pytest.warns(EnvironmentVariableDeprecationWarning):
        assert manager.get("MM_NEW_OPTION") is True


def test_default_is_used_when_neither_name_is_set(monkeypatch):
    manager = _manager_with_deprecated_alias()
    monkeypatch.delenv("MM_OLD_OPTION", raising=False)
    monkeypatch.delenv("MM_NEW_OPTION", raising=False)
    assert manager.get("MM_NEW_OPTION") is False


def test_exported_values_are_read_in_a_child_process():
    script = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location(
    'child_fsdp_envs',
    {str(ENVS_PACKAGE_PATH)!r},
    submodule_search_locations=[{str(ENVS_PACKAGE_DIR)!r}],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.LOCAL_RANK == 3
assert module.WORLD_SIZE == 8
assert module.MM_FORCE_EP_BALANCE is True
assert module.NON_MEGATRON is True
"""
    child_env = os.environ.copy()
    child_env.update(
        {
            "LOCAL_RANK": "3",
            "WORLD_SIZE": "8",
            "MM_FORCE_EP_BALANCE": "on",
            "NON_MEGATRON": "true",
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=child_env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "name",
    (
        "VIDEO_PLACEHOLDER",
        "IMAGE_PLACEHOLDER",
        "AUDIO_PLACEHOLDER",
        "FLA_CI_ENV",
        "FLA_NO_USE_TMA",
        "FLA_USE_FAST_OPS",
        "FLA_TRIL_PRECISION",
    ),
)
def test_unsupported_variables_remain_outside_fsdp_env_manager(name):
    assert name not in envs.environment_variables
    with pytest.raises(AttributeError, match=name):
        getattr(envs, name)


def test_fsdp_business_code_has_no_direct_environment_reads():
    import ast

    fsdp_root = REPO_ROOT / "mindspeed_mm" / "fsdp"
    allowed_direct_environment_readers = {
        fsdp_root / "data" / "data_utils" / "func_utils" / "mm_plugin.py",
        # Added upstream after this PR was opened; keep it as an explicit
        # compatibility exception until inference adopts the FSDP env manager.
        fsdp_root / "inference" / "inference_runner.py",
        fsdp_root / "ops" / "gdn" / "triton" / "solve_tril.py",
        fsdp_root / "ops" / "gdn" / "triton" / "utils.py",
        fsdp_root / "ops" / "gdn" / "triton_core" / "utils.py",
    }
    violations = []
    for path in fsdp_root.rglob("*.py"):
        if (
            path == fsdp_root / "__init__.py"
            or fsdp_root / "envs" in path.parents
            or path in allowed_direct_environment_readers
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            function = node.func
            is_getenv = (
                isinstance(function.value, ast.Name)
                and function.value.id == "os"
                and function.attr == "getenv"
            )
            is_environ_get = (
                isinstance(function.value, ast.Attribute)
                and isinstance(function.value.value, ast.Name)
                and function.value.value.id == "os"
                and function.value.attr == "environ"
                and function.attr == "get"
            )
            if is_getenv or is_environ_get:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not violations, "Direct environment reads found:\n" + "\n".join(violations)
