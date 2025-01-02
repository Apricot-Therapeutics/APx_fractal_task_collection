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
Initializes the parallelization list for normalize measurements task.
"""
import logging
from typing import Any, Optional
from apx_fractal_task_collection.init_utils import (group_by_well,
                                                    get_label_zarr_url,
                                                    get_channel_zarr_url)
from pydantic import validate_call
from pathlib import Path
import zarr
from anndata.experimental import read_elem

logger = logging.getLogger(__name__)

@validate_call
def init_expand_labels(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    control_condition: str,
    feature_table_name: str,
#    normalization_layout: # should be drop-down with options available
) -> dict[str, list[dict[str, Any]]]:
    """
    Initialized normalize measurement task


    This task prepares a parallelization list of all zarr_urls that need to be
    used to perform measurement normalization.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        control_condition: Name of the control condition to be used for
            normalization.
        feature_table_name: Name of the feature table that contains the
            measurements to be normalized.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    logger.info(
        f"Running `init_normalize_measurements.py` for {zarr_urls=}"
    )

    # for each zarr-url in zarr-urls, load a sample and get the condition
    condition_dict = {}
    for zarr_url in zarr_urls:
        zarr_store = zarr.open(f"{zarr_url}/tables/{feature_table_name}",
                               mode="r")
        condition = read_elem(zarr_store["obs/condition"])[0]
        condition_dict[zarr_url] = condition

    # simplest way: use all zarr-urls that have control_condition as condition
    ctrl_zarr_urls = [zarr_url for zarr_url, condition in condition_dict.items()
                      if condition == control_condition]

    # Create the parallelization list
    parallelization_list = []

    for zarr_url in zarr_urls:
        parallelization_list.append(
            dict(
                zarr_url=zarr_url,
                init_args=dict(
                    ctrl_zarr_urls=ctrl_zarr_urls,
                ),
            )
        )

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=init_expand_labels,
        logger_name=logger.name,
    )