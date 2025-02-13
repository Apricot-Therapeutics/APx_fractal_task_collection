# Original authors:
# Adrian Tschan <atschan@apricotx.com>
#
# This file is part of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Initializes the parallelization list for normalize feature table task.
"""
import logging
from typing import Any, List
from enum import Enum
import pandas as pd

from pydantic import validate_call
from pathlib import Path
import zarr
from anndata.experimental import read_elem

logger = logging.getLogger(__name__)

class NormalizationLayout(Enum):
    """
    Enum for the normalization layout options.
    """

    full_plate = "full plate"
    row_and_column = "row and column"

@validate_call
def init_normalize_feature_table(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    condition_column: str = "condition",
    control_condition: str,
    feature_table_name: str,
    normalization_layout: NormalizationLayout = NormalizationLayout.full_plate,
    additional_control_filters: dict[str, str] = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Initializes normalize feature table task


    This task prepares a parallelization list of all zarr_urls that need to be
    used to perform measurement normalization.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
        condition_column: Name of the column in the feature table that contains
            the condition information.
        control_condition: Name of the control condition to be used for
            normalization.
        feature_table_name: Name of the feature table that contains the
            measurements to be normalized.
        normalization_layout: Layout of the normalization. Options are:
            - full plate: Use all control wells for normalization.
            - row and column: Use all control wells in the same row and column
                as the well to be normalized.
        additional_control_filters: Dictionary of additional filters to be
            applied to filter for control conditions. The dictionary should be
            formatted as: { "column_name": "value",}.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    logger.info(
        f"Running `init_normalize_feature_table.py` for {zarr_urls=}"
    )
    
    # filter zarr-urls to only include zarrs that have the feature table
    # (for example, aggregated tables are not present in all zarr files)
    zarr_urls = [zarr_url for zarr_url in zarr_urls if
                    Path(f"{zarr_url}/tables/{feature_table_name}").exists()]

    # for each zarr_url in zarr_urls, load a sample and get the condition
    condition_cycle_dict = {'zarr_url': [],
                            'condition': [],
                            'cycle': [],
                            'row': [],
                            'col': []}

    # add columns from additional_control_filters to the dict
    for col in additional_control_filters:
        condition_cycle_dict[col] = []

    for zarr_url in zarr_urls:
        # get the condition
        zarr_store = zarr.open(f"{zarr_url}/tables/{feature_table_name}",
                               mode="r")
        condition = read_elem(zarr_store[f"obs/{condition_column}"])[0]

        # get the cycle
        cycle_path = Path(zarr_url).name
        row = Path(zarr_url).parents[1].name
        col = Path(zarr_url).parent.name

        condition_cycle_dict['zarr_url'].append(zarr_url)
        condition_cycle_dict['condition'].append(condition)
        condition_cycle_dict['cycle'].append(cycle_path)
        condition_cycle_dict['row'].append(row)
        condition_cycle_dict['col'].append(col)

        # get additional control filters
        for col, value in additional_control_filters.items():
            column_value = read_elem(zarr_store[f"obs/{col}"])[0]
            condition_cycle_dict[col].append(column_value)


    condition_cycle_df = pd.DataFrame(condition_cycle_dict)

    # filter df to only include control conditions
    ctrl_df = condition_cycle_df.loc[
        condition_cycle_df['condition'] == control_condition]

    # if additional control filters are provided, filter the control df
    if additional_control_filters:
        for col, value in additional_control_filters.items():
            ctrl_df = ctrl_df.loc[ctrl_df[col] == value]

    # Create the parallelization list
    parallelization_list = []

    for i, row in condition_cycle_df.iterrows():
        # get the control zarr urls in the same cycle
        filtered_ctrl_df = ctrl_df.loc[ctrl_df['cycle'] == row['cycle']]
        if normalization_layout == NormalizationLayout.row_and_column:
            # get the control zarr urls in the same row and column
            filtered_ctrl_df = filtered_ctrl_df.loc[
                (filtered_ctrl_df['row'] == row['row']) |
                (filtered_ctrl_df['col'] == row['col'])]

        ctrl_zarr_urls = filtered_ctrl_df['zarr_url'].tolist()

        logger.info(f"Normalizing zarr at {row['zarr_url']} with "
                    f"control wells at: {ctrl_zarr_urls}")

        parallelization_list.append(
            dict(
                zarr_url=row.zarr_url,
                init_args=dict(
                    ctrl_zarr_urls=ctrl_zarr_urls,
                    feature_table_name=feature_table_name,
                ),
            )
        )

    return dict(parallelization_list=parallelization_list)

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=init_normalize_feature_table,
        logger_name=logger.name,
    )