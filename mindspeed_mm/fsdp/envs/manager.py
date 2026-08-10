"""Reusable typed environment-variable definitions and management."""

import os
import warnings
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeVar

T = TypeVar("T")
Converter = type[T] | Callable[[str], T]
Validator = Callable[[Any], None]
_DEFAULT_NOT_PROVIDED = object()

TRUE_VALUES = frozenset({"1", "true", "yes", "y", "on"})
FALSE_VALUES = frozenset({"0", "false", "no", "n", "off"})


class EnvironmentVariableDeprecationWarning(FutureWarning):
    """Warning emitted when a deprecated environment variable is used."""


def parse_bool(value: str) -> bool:
    """Parse a shell string into a boolean using the project-wide convention."""
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(TRUE_VALUES | FALSE_VALUES))
    raise ValueError(f"expected one of: {accepted}")


def parse_exact_true(value: str) -> bool:
    """Preserve the historical ``NON_MEGATRON == 'true'`` behavior."""
    return value == "true"


def validate_positive(value: int) -> None:
    """Require a value greater than or equal to one."""
    if value < 1:
        raise ValueError("must be greater than or equal to 1")


def validate_non_negative(value: int) -> None:
    """Require a value greater than or equal to zero."""
    if value < 0:
        raise ValueError("must be greater than or equal to 0")


@dataclass(frozen=True)
class EnvironmentVariable:
    """Definition and documentation metadata for one environment variable."""

    name: str
    default: Any
    converter: Converter[Any] = str
    description: str = ""
    deprecated: bool = False
    replacement: str | None = None
    choices: Collection[Any] | None = None
    validator: Validator | None = None

    def convert(self, raw_value: str) -> Any:
        converter = parse_bool if self.converter is bool else self.converter
        try:
            value = converter(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid value for environment variable {self.name}: "
                f"{raw_value!r} ({error})"
            ) from error
        self.validate(value)
        return value

    def default_value(self) -> Any:
        value = self.default() if callable(self.default) else self.default
        return self.validate_default(value)

    def validate_default(self, value: Any) -> Any:
        """Validate and return a typed default supplied by code."""
        if isinstance(self.converter, type):
            type_matches = isinstance(value, self.converter)
            if self.converter is int and isinstance(value, bool):
                type_matches = False
            if not type_matches:
                raise ValueError(
                    f"Invalid default for environment variable {self.name}: "
                    f"{value!r}. Expected {self.converter.__name__}."
                )
        self.validate(value)
        return value

    def validate(self, value: Any) -> None:
        if self.choices is not None and value not in self.choices:
            raise ValueError(
                f"Invalid value for environment variable {self.name}: {value!r}. "
                f"Expected one of: {tuple(self.choices)!r}"
            )
        if self.validator is None:
            return
        try:
            self.validator(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid value for environment variable {self.name}: "
                f"{value!r} ({error})"
            ) from error


class EnvironmentVariableManager:
    """Registry providing lazy, typed reads from ``os.environ``."""

    def __init__(self) -> None:
        self._variables: dict[str, EnvironmentVariable] = {}

    @property
    def variables(self) -> Mapping[str, EnvironmentVariable]:
        """Return a read-only, live view of registered definitions."""
        return MappingProxyType(self._variables)

    def register(self, variable: EnvironmentVariable) -> None:
        if not variable.name:
            raise ValueError("Environment variable name cannot be empty")
        if not variable.description.strip():
            raise ValueError(
                f"Environment variable {variable.name} must have a description"
            )
        if variable.name in self._variables:
            raise ValueError(
                f"Environment variable {variable.name} is already registered"
            )
        if variable.replacement == variable.name:
            raise ValueError(
                f"Environment variable {variable.name} cannot replace itself"
            )
        if variable.replacement and not variable.deprecated:
            raise ValueError(
                f"Environment variable {variable.name} has a replacement "
                "but is not deprecated"
            )
        variable.default_value()
        self._variables[variable.name] = variable

    def validate_registry(self) -> None:
        """Validate replacement targets and reject replacement cycles."""
        for variable in self._variables.values():
            if variable.replacement and variable.replacement not in self._variables:
                raise ValueError(
                    f"Replacement {variable.replacement} for environment variable "
                    f"{variable.name} is not registered"
                )

        for name in self._variables:
            visited: set[str] = set()
            current = name
            while self._variables[current].replacement:
                if current in visited:
                    chain = " -> ".join((*visited, current))
                    raise ValueError(
                        f"Environment variable replacement cycle detected: {chain}"
                    )
                visited.add(current)
                replacement = self._variables[current].replacement
                assert replacement is not None
                current = replacement

    def get_spec(self, name: str) -> EnvironmentVariable:
        try:
            return self._variables[name]
        except KeyError as error:
            raise KeyError(f"Unknown environment variable: {name}") from error

    def is_set(self, name: str, include_deprecated: bool = True) -> bool:
        """Return whether a variable or its deprecated alias is explicitly set."""
        self.get_spec(name)
        if name in os.environ:
            return True
        return include_deprecated and any(
            variable.replacement == name and variable.name in os.environ
            for variable in self._variables.values()
        )

    def get(
        self,
        name: str,
        default: Any = _DEFAULT_NOT_PROVIDED,
        *,
        required: bool = False,
    ) -> Any:
        """Resolve a registered variable without caching its value.

        A call-site default preserves legacy readers that intentionally used a
        fallback different from the registered project-wide default. Required
        reads preserve launcher contracts that previously failed when unset.
        """
        if required and default is not _DEFAULT_NOT_PROVIDED:
            raise ValueError("default and required cannot be used together")

        variable = self.get_spec(name)

        if variable.deprecated:
            self._warn_deprecated(variable)
            if variable.replacement and variable.replacement in os.environ:
                return self._read(self.get_spec(variable.replacement))
            if variable.name not in os.environ:
                if required:
                    raise KeyError(f"Required environment variable is not set: {name}")
                if default is not _DEFAULT_NOT_PROVIDED:
                    return variable.validate_default(default)
            return self._read(variable)

        deprecated_sources = tuple(
            candidate
            for candidate in self._variables.values()
            if candidate.replacement == name and candidate.name in os.environ
        )
        for candidate in deprecated_sources:
            self._warn_deprecated(candidate)

        # An explicitly exported new name always wins over deprecated aliases.
        if name in os.environ:
            return self._read(variable)
        if deprecated_sources:
            return self._read(deprecated_sources[0])
        if required:
            raise KeyError(f"Required environment variable is not set: {name}")
        if default is not _DEFAULT_NOT_PROVIDED:
            return variable.validate_default(default)
        return variable.default_value()

    def descriptions(self) -> Mapping[str, str]:
        """Return descriptions for documentation and diagnostics."""
        return MappingProxyType(
            {name: variable.description for name, variable in self._variables.items()}
        )

    @staticmethod
    def _read(variable: EnvironmentVariable) -> Any:
        if variable.name not in os.environ:
            return variable.default_value()
        return variable.convert(os.environ[variable.name])

    @staticmethod
    def _warn_deprecated(variable: EnvironmentVariable) -> None:
        message = f"Environment variable {variable.name} is deprecated."
        if variable.replacement:
            message += f" Use {variable.replacement} instead."
        warnings.warn(
            message,
            EnvironmentVariableDeprecationWarning,
            stacklevel=4,
        )
