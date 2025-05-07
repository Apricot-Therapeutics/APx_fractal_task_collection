# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
# Adrian Tschan <atschan@apricotx.com>
#
# This file is part of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Initializes the parallelization list for BaSiCPy illumination correction.
"""
import logging
from typing import Any, Optional

from apx_fractal_task_collection.io_models import CorrectBy
from apx_fractal_task_collection.init_utils import (group_by_channel, 
                                                    group_by_well_and_channel, 
                                                    group_by_wavelength, 
                                                    group_by_well_and_wavelength)
import random
from pydantic import validate_call
import pandas as pd
import anndata as ad

logger = logging.getLogger(__name__)

@validate_call
def init_calculate_basicpy_illumination_models(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    n_images: int = 150,
    correct_by: CorrectBy = CorrectBy.channel_label,
    compute_per_well: bool = False,
    exclude_border_FOVs: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """
    Initialized BaSiCPy illumination correction task
    
    This task prepares a parallelization list of all zarr_urls that need to be
    used to perform illumination correction with BaSiCPy.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        n_images: Number of images to use to calculate BaSiCPy model.
        correct_by: Defines how illumination correction will be calculated. 
            - channel label: illumination correction will be calculated per channel label
            - wavelength id: illumination correction will be calculated per wavelength id
        exclude_border_FOVs: If True, exclude border FOVs from the calculation.
            Useful if the whole well was imaged and the border FOVs have some
            artifacts.
        compute_per_well: If True, calculate illumination profiles per well.
            This can be useful if your experiment contains different stainings
            in each well (e.g., different antibodies with varying intensity
            ranges). Defaults to False.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    logger.info(
        f"Running `init_calculate_basicpy_illumination_models` "
        f"for {zarr_urls=}"
    )

    logger.info(
        f"Calculating illumination profiles based on {n_images} "
        f"randomly sampled images.")
    
    logger.info(
        f"Illumination profiles will be calculated based on {correct_by}."
    )
        
    if correct_by == CorrectBy.channel_label:
        if compute_per_well:
            channel_dict = group_by_well_and_channel(zarr_urls)
        else:
            channel_dict = group_by_channel(zarr_urls)

            FOV_ROI_table = ad.read_zarr(f"{zarr_urls[0]}/tables/FOV_ROI_table")
            FOV_ROI_df = FOV_ROI_table.to_df()

            # exclude FOVs where x_micrometer or y_micrometer is equal to 0 or
            # the max of x_micrometer or y_micrometer
            if exclude_border_FOVs:
                FOV_ROI_table = FOV_ROI_table[
                    (FOV_ROI_df['x_micrometer'] != 0)
                    & (FOV_ROI_df['y_micrometer'] != 0)
                    & (FOV_ROI_df['x_micrometer'] != FOV_ROI_df['x_micrometer'].max())
                    & (FOV_ROI_df['y_micrometer'] != FOV_ROI_df['y_micrometer'].max())
                , :]

            n_FOVs = len(FOV_ROI_table)

            # repeat each entry in the channel_dict n_FOVs times
            for channel, channel_list in channel_dict.items():
                channel_dict[channel] = channel_list * n_FOVs

    # Correct by wavelength id
    else:
        if compute_per_well:
            channel_dict = group_by_well_and_wavelength(zarr_urls)
        else:
            channel_dict = group_by_wavelength(zarr_urls)

            FOV_ROI_table = ad.read_zarr(f"{zarr_urls[0]}/tables/FOV_ROI_table")
            FOV_ROI_df = FOV_ROI_table.to_df()

            # exclude FOVs where x_micrometer or y_micrometer is equal to 0 or
            # the max of x_micrometer or y_micrometer
            if exclude_border_FOVs:
                FOV_ROI_table = FOV_ROI_table[
                    (FOV_ROI_df['x_micrometer'] != 0)
                    & (FOV_ROI_df['y_micrometer'] != 0)
                    & (FOV_ROI_df['x_micrometer'] != FOV_ROI_df['x_micrometer'].max())
                    & (FOV_ROI_df['y_micrometer'] != FOV_ROI_df['y_micrometer'].max())
                , :]

            n_FOVs = len(FOV_ROI_table)

            # repeat each entry in the channel_dict n_FOVs times
            for channel, channel_list in channel_dict.items():
                channel_dict[channel] = channel_list * n_FOVs

    # Create the parallelization list
    parallelization_list = []

    for channel, channel_list in channel_dict.items():

        if compute_per_well:
            # sample n_images times from the zarr_urls
            channel_zarr_urls = pd.DataFrame(
                channel_list*n_images,
                columns=['zarr_url'])
        else:
            # sample n_images times from the zarr_urls
            channel_zarr_urls = pd.DataFrame(
                random.sample(channel_list, k=n_images),
                columns=['zarr_url'])

        # get a dictionary of how often each zarr_url occurs in channel_zarr_urls
        channel_zarr_dict = channel_zarr_urls['zarr_url'].value_counts().to_dict()
        channel_zarr_urls = list(channel_zarr_urls['zarr_url'].unique())

        # generate zarr_url by taking the first entry of well_list and
        # replacing the last part of the path with the output_label_image_name
        zarr_url = zarr_urls[0] # never really used

        parallelization_list.append(
            dict(
                zarr_url=zarr_url,
                init_args=dict(
                    channel_name=channel,
                    # convert to str to avoid JSON serialization error
                    correct_by=correct_by.name,
                    channel_zarr_urls=channel_zarr_urls,
                    channel_zarr_dict=channel_zarr_dict,
                    compute_per_well=compute_per_well,
                    exclude_border_FOVs=exclude_border_FOVs,
                ),
            )
        )
    return dict(parallelization_list=parallelization_list)
    

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=init_calculate_basicpy_illumination_models,
        logger_name=logger.name,
    )