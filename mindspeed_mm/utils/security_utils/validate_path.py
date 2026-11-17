# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Union, Tuple


def normalize_path(
    path: str,
    resolve_symlinks: bool = True,
    check_symlink: bool = True
) -> Union[str, Tuple[str, bool]]:
    """
    Path normalization and symbolic link detection

    Args:
        path (str): Path to be processed
        resolve_symlinks (bool): Whether to resolve symbolic links
        check_symlink (bool): Whether to check symbolic link status

    Returns:
        Union[str, Tuple[str, bool]]:
            - check_symlink=False: Normalized path string
            - check_symlink=True: (normalized path, is symbolic link)
    """
    # Handle empty path
    if not path:
        path = Path.cwd()
    else:
        path = Path(path).expanduser()  # Expand user directory first

    # Symbolic link detection (prior to path resolution)
    is_link = False
    if check_symlink:
        try:
            is_link = path.is_symlink()
        except OSError:
            pass  # Ignore permission errors and other exceptions

    # Path normalization processing
    try:
        if resolve_symlinks:
            normalized_path = path.resolve()
        else:
            normalized_path = path.absolute()
    except OSError:
        # Fallback handling: Attempt absolute path resolution
        try:
            normalized_path = path.absolute()
        except OSError:
            # Final fallback: Concatenate with current working directory
            normalized_path = Path.cwd() / path

    # Return result
    return (str(normalized_path), is_link) if check_symlink else str(normalized_path)
