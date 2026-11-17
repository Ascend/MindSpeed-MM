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

from pandas import DataFrame
import pandas as pd


def sanitize_dataframe(
        df: pd.DataFrame,
        escape_char: str = '"'
) -> DataFrame:
    """
    Wrap all cell values in the DataFrame with specified quotation marks

    Parameters:
        df: Original DataFrame
        escape_char: Quotation mark to use for wrapping (default: double quote)

    Returns:
        New DataFrame with all cells wrapped in quotation marks
    """
    result_df = df.copy()

    # Add quotes to column names
    result_df.columns = [f"{escape_char}{col}{escape_char}" for col in result_df.columns]

    # Apply wrapping to all columns
    for col in result_df.columns:
        result_df[col] = result_df[col].apply(
            lambda x: f"{escape_char}{x}{escape_char}"
        )

    return result_df
