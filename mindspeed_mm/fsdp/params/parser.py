# Copyright 2025 Bytedance Ltd. and/or its affiliates
import argparse
import json
import math
import os
import sys
import types
from collections import defaultdict
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from enum import Enum
from inspect import isclass
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union, get_type_hints, get_origin, get_args
import logging

import yaml

T = TypeVar("T")

logger = logging.getLogger(__name__)


def _string_to_bool(value: Union[bool, str]) -> bool:

    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(
        f"Truthy value expected: got {value} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
    )


def _convert_str_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:

    for key, value in input_dict.items():
        if isinstance(value, dict):
            input_dict[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            if value.lower() in ("true", "false"):  # check for bool
                input_dict[key] = value.lower() == "true"
            elif value.isdigit():  # check for digit
                input_dict[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                input_dict[key] = float(value)

    return input_dict


def _make_choice_type_function(choices: List[Any]) -> Callable[[str], Any]:

    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


def _parse_nested_dataclass(
    parser: argparse.ArgumentParser,
    dataclass_type: type,
    parent_prefix: str = "",
    dict_fields: set = None
) -> None:
    """
    Recursively add arguments for nested dataclass fields.
    
    Args:
        parser: Argument parser
        dataclass_type: Dataclass type to parse
        parent_prefix: Prefix for the argument name (e.g., "model.encoder")
        dict_fields: Set to store dict field paths
    """
    if dict_fields is None:
        dict_fields = set()
    
    try:
        type_hints: Dict[str, type] = get_type_hints(dataclass_type)
    except Exception as e:
        raise RuntimeError(f"Type resolution failed for {dataclass_type}: {e}") from e


    for attr in fields(dataclass_type):
        if not attr.init:
            continue

        attr_name = attr.name
        attr_type = type_hints[attr_name]
        origin_type = get_origin(attr_type) or attr_type

        # Check if this is a nested dataclass
        if is_dataclass(attr_type) or (origin_type is None and is_dataclass(attr_type)):
            # Recursively process nested dataclass
            nested_prefix = f"{parent_prefix}.{attr_name}" if parent_prefix else attr_name
            _parse_nested_dataclass(parser, attr_type, nested_prefix, dict_fields)
            continue

        if origin_type is Union or (hasattr(types, "UnionType") and isinstance(origin_type, types.UnionType)):
            if len(attr_type.__args__) != 2 or type(None) not in attr_type.__args__:  # only allows Optional[X]
                raise RuntimeError(f"Cannot resolve type {attr.type} of {attr.name}.")

            if bool not in attr_type.__args__:  # except for `Union[bool, NoneType]`
                attr_type = (
                    attr_type.__args__[0] if isinstance(None, attr_type.__args__[1]) else attr_type.__args__[1]
                )
                origin_type = getattr(attr_type, "__origin__", attr_type)

        # Build argument name
        arg_name = f"{parent_prefix}.{attr_name}" if parent_prefix else attr_name
        parser_kwargs = attr.metadata.copy()

        # Handle different types
        if origin_type is Literal or (isinstance(attr_type, type) and issubclass(attr_type, Enum)):
            if origin_type is Literal:
                parser_kwargs["choices"] = attr_type.__args__
            else:
                parser_kwargs["choices"] = [x.value for x in attr_type]

            parser_kwargs["type"] = _make_choice_type_function(parser_kwargs["choices"])

            if attr.default is not MISSING:
                parser_kwargs["default"] = attr.default
            else:
                parser_kwargs["required"] = True

        elif attr_type is bool or attr_type == Optional[bool]:
            parser_kwargs["type"] = _string_to_bool
            if attr_type is bool or (attr.default is not None and attr.default is not MISSING):
                parser_kwargs["default"] = False if attr.default is MISSING else attr.default
                parser_kwargs["nargs"] = "?"
                parser_kwargs["const"] = True

        elif isclass(origin_type) and issubclass(origin_type, list):
            parser_kwargs["type"] = attr_type.__args__[0]
            parser_kwargs["nargs"] = "+"
            if attr.default_factory is not MISSING:
                parser_kwargs["default"] = attr.default_factory()
            elif attr.default is MISSING:
                parser_kwargs["required"] = True

        elif isclass(origin_type) and issubclass(origin_type, dict):
            parser_kwargs["type"] = str  # parse dict inputs with json string
            dict_fields.add(f"{parent_prefix}.{attr_name}" if parent_prefix else attr_name)
            if attr.default_factory is not MISSING:
                parser_kwargs["default"] = str(attr.default_factory())
            elif attr.default is MISSING:
                parser_kwargs["required"] = True

        else:
            parser_kwargs["type"] = attr_type
            if attr.default is not MISSING:
                parser_kwargs["default"] = attr.default
            elif attr.default_factory is not MISSING:
                parser_kwargs["default"] = attr.default_factory()
            else:
                parser_kwargs["required"] = True

        # Add argument to parser
        parser.add_argument(f"--{arg_name}", **parser_kwargs)


def _build_nested_structure(parse_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert flat dictionary with dot notation to nested structure.
    
    Args:
        parse_result: Flat dictionary with keys like "model.encoder.layers"
    
    Returns:
        Nested dictionary structure
    """
    result = {}
    
    for key, value in parse_result.items():
        parts = key.split(".")
        current = result
        
        # Build nested structure
        for _, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Set the value
        current[parts[-1]] = value
    
    return result


def _instantiate_dataclass(dataclass_type: type, data: Dict[str, Any]) -> Any:
    """
    Recursively instantiate dataclass from dictionary data.
    
    Args:
        dataclass_type: Dataclass type to instantiate
        data: Dictionary data
    
    Returns:
        Instantiated dataclass object
    """
    if not is_dataclass(dataclass_type):
        return data
    
    # Get type hints for the dataclass
    try:
        type_hints = get_type_hints(dataclass_type)
    except Exception as e:
        raise RuntimeError(f"Failed to get type hints for {dataclass_type}: {e}") from e
    
    kwargs = {}
    
    for field_obj in fields(dataclass_type):
        if not field_obj.init:
            continue
        
        field_name = field_obj.name
        field_type = type_hints[field_name]
        
        # Get the value from data (or use default)
        if field_name in data:
            value = data[field_name]
            
            # Check if this is a nested dataclass
            origin_type = get_origin(field_type) or field_type
            
            # Handle nested dataclass
            if is_dataclass(field_type):
                if isinstance(value, dict):
                    kwargs[field_name] = _instantiate_dataclass(field_type, value)
                else:
                    # If value is not a dict, use default factory
                    if field_obj.default_factory is not MISSING:
                        kwargs[field_name] = field_obj.default_factory()
                    elif field_obj.default is not MISSING:
                        kwargs[field_name] = field_obj.default
            # Handle Optional[nested dataclass]
            elif origin_type is Union:
                args = get_args(field_type)
                # Find the dataclass type in Union args
                dataclass_arg = next((arg for arg in args if is_dataclass(arg)), None)
                if dataclass_arg and isinstance(value, dict):
                    kwargs[field_name] = _instantiate_dataclass(dataclass_arg, value)
                else:
                    kwargs[field_name] = value
            else:
                kwargs[field_name] = value
        else:
            # Use default value if not provided
            if field_obj.default_factory is not MISSING:
                kwargs[field_name] = field_obj.default_factory()
            elif field_obj.default is not MISSING:
                kwargs[field_name] = field_obj.default
    
    return dataclass_type(**kwargs)


def parse_args(rootclass: T) -> T:
    """
    Parses the root argument class using the CLI inputs or yaml inputs.

    Based on: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/hf_argparser.py#L266
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Set to store dict field paths
    dict_fields = set()

    # First, check if rootclass itself is a dataclass
    if not is_dataclass(rootclass):
        raise ValueError(f"rootclass must be a dataclass, got {type(rootclass)}")
    
    # Recursively add arguments for all nested dataclass fields
    _parse_nested_dataclass(parser, rootclass, "", dict_fields)

    # Parse command line arguments
    cmd_args = sys.argv[1:]

    cmd_args_string = "=".join(cmd_args)  # use `=` to mark the end of arg name
    # Handle config file input
    input_data = {}
    if cmd_args[0].endswith(".yaml") or cmd_args[0].endswith(".yml"):
        input_path = cmd_args.pop(0)
        with open(os.path.abspath(input_path), encoding="utf-8") as f:
            input_data: Dict[str, Dict[str, Any]] = yaml.safe_load(f)

    elif cmd_args[0].endswith(".json"):
        input_path = cmd_args.pop(0)
        with open(os.path.abspath(input_path), encoding="utf-8") as f:
            input_data: Dict[str, Dict[str, Any]] = json.load(f)

    # Convert nested config data to flat arguments
    def _flatten_config(data: Dict[str, Any], prefix: str = "") -> List[str]:
        """Flatten nested dictionary to command line arguments."""
        args = []
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                args.extend(_flatten_config(value, full_key))
            else:
                # Add the argument if not already in cmd_args
                arg_string = f"--{full_key}"
                if not any(arg.startswith(arg_string + "=") or arg == arg_string for arg in cmd_args):
                    args.append(f"--{full_key}")
                    if isinstance(value, list):
                        args.extend(str(x) for x in value)
                    elif isinstance(value, dict):
                        args.append(json.dumps(value))
                    else:
                        args.append(str(value))
        return args

    # Add config file arguments to cmd_args (lower priority than CLI)
    if input_data:
        cmd_args.extend(_flatten_config(input_data))

    # Parse arguments
    args, remaining_args = parser.parse_known_args(cmd_args)
    if remaining_args:
        logger.warning(f"Some specified arguments are not used by the ArgumentParser: {remaining_args}")

    # Convert parsed args to dictionary and handle dict fields
    parse_result = defaultdict(dict)
    for key, value in vars(args).items():
        if key in dict_fields:
            if isinstance(value, str) and value.startswith("{"):
                value = _convert_str_dict(json.loads(value))
            else:
                raise ValueError(f"Expect a json string for dict argument, but got {value}")


        parse_result[key] = value

    # Convert flat structure to nested structure
    nested_result = _build_nested_structure(parse_result)

    # Recursively instantiate dataclass
    return _instantiate_dataclass(rootclass, nested_result)
