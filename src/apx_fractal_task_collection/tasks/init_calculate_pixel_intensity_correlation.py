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
Initializes the parallelization list for calculate pixel intensity correlation.
"""
import logging
from typing import Any, Optional
from apx_fractal_task_collection.init_utils import (group_by_well,
                                                    get_label_zarr_url,
                                                    get_channel_zarr_url)
from pydantic import validate_call
from pathlib import Path

logger = logging.getLogger(__name__)

@validate_call
def init_calculate_pixel_intensity_correlation(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    label_name: str,
    correlation_pairs:list[dict[str, str]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Initializes the Calculate Pixel Intensity Correlation task
    
    This task prepares a parallelization list of all zarr_urls that need to be
    used to perform Calculate Pixel Intensity Correlation task based on
    two channel images and a label image.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Name of the label image that contains the seeds.
            Needs to exist in OME-Zarr file.
        correlation_pairs: List of dictionaries. Key and value represent
        the label names of two channels that should be correlated.
        CAREFUL: add a new dictionary for each correlation.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    logger.info(
        f"Running `init_calculate_pixel_intensity_correlation` for {zarr_urls=}"
    )
    well_dict = group_by_well(zarr_urls)
    # Create the parallelization list
    parallelization_list = []

    for well, well_list in well_dict.items():
        label_zarr_url = get_label_zarr_url(well_list, label_name)
        zarr_url = well_list[0]

        corr_channel_urls = []
        corr_channel_labels = []
        for channel_pair in correlation_pairs:
            channel_zarr_url_1 = get_channel_zarr_url(
                well_list,
                list(channel_pair.keys())[0])
            channel_zarr_url_2 = get_channel_zarr_url(
                well_list,
                list(channel_pair.values())[0])

            corr_channel_urls.append(
                {channel_zarr_url_1: channel_zarr_url_2})
            corr_channel_labels.append(
                {list(channel_pair.keys())[0]: list(channel_pair.values())[0]})


        parallelization_list.append(
            dict(
                zarr_url=zarr_url,
                init_args=dict(
                    corr_channel_urls=corr_channel_urls,
                    corr_channel_labels=corr_channel_labels,
                    label_name=label_name,
                    label_zarr_url=label_zarr_url,
                ),
            )
        )
    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=init_calculate_pixel_intensity_correlation,
        logger_name=logger.name,
    )