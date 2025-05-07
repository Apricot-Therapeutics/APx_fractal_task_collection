# Original authors:
# Adrian Tschan <adrian.tschan@uzh.ch>
#
# This file is part of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.

import logging
import random
from pathlib import Path

import anndata as ad
from basicpy import BaSiC
import dask.array as da
import numpy as np

from pydantic import Field
from pydantic import validate_call

from apx_fractal_task_collection.io_models import InitArgsBaSiCPyCalculate
from apx_fractal_task_collection.utils import BaSiCPyModelParams

from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)

logger = logging.getLogger(__name__)


@validate_call
def calculate_basicpy_illumination_models(
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsBaSiCPyCalculate,
    # Task-specific arguments
    illumination_profiles_folder: str,
    advanced_basicpy_model_params: BaSiCPyModelParams = Field(
        default_factory=BaSiCPyModelParams),
    overwrite: bool = False,
) -> None:

    """
    Calculates illumination correction profiles based on a random sample
    of images for each channel_label or wavelength.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_calculate_basicpy_illumination_models`.
        illumination_profiles_folder: Path to folder where illumination
            profiles will be saved.
        advanced_basicpy_model_params: Advanced parameters for the BaSiC model.
        overwrite: If True, overwrite existing illumination profiles.
    """

    # take care of channel label init arg (has well name in it if
    # compute_per_well)
    if init_args.compute_per_well:
        channel_name = init_args.channel_name.split("_ch_")[1]
    else:
        channel_name = init_args.channel_name

    # Check if illumination profiles are calculated per channel or wavelength
    correct_by = init_args.correct_by
    logger.info(
        f"Calculating illumination profile for channel "
        f"{channel_name} based on {correct_by=}.")


    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(init_args.channel_zarr_urls[0])
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(
        level=0)

    ROI_data = []

    i_img = 0
    n_images = np.sum(list(init_args.channel_zarr_dict.values()))

    # collect all images for the channel
    for zarr_url in init_args.channel_zarr_urls:
        # Read FOV ROIs
        FOV_ROI_table = ad.read_zarr(
            f"{zarr_url}/tables/FOV_ROI_table")

        # Create list of indices for 3D FOVs spanning the entire Z direction
        list_indices = convert_ROI_table_to_indices(
            FOV_ROI_table,
            level=0,
            coarsening_xy=coarsening_xy,
            full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
        )
        check_valid_ROI_indices(list_indices, "FOV_ROI_table")

        # Exclude border FOVs if requested
        if init_args.exclude_border_FOVs:
            max_y = np.max([x[2] for x in list_indices])
            max_x = np.max([x[4] for x in list_indices])
            list_indices = [x for x in list_indices if
                             (x[2] != 0) & (x[4] != 0) & (x[2] != max_y) & (
                                     x[4] != max_x)]
        # Extract image size from FOV-ROI indices. Note: this works at level=0,
        # where FOVs should all be of the exact same size (in pixels)
        ref_img_size = None
        for indices in list_indices:
            img_size = (
            indices[3] - indices[2], indices[5] - indices[4])
            if ref_img_size is None:
                ref_img_size = img_size
            else:
                if img_size != ref_img_size:
                    raise ValueError(
                        "ERROR: inconsistent image sizes in list_indices"
                    )
        if correct_by == "channel_label":
            tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=None,
                label=channel_name,
            )
        else:
            tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=channel_name,
                label=None,
            )
        ind_channel = tmp_channel.index
        data_zyx = \
        da.from_zarr(f"{zarr_url}/0")[
            ind_channel]

        list_indices = random.sample(list_indices,
                                     init_args.channel_zarr_dict[zarr_url])

        for i_ROI, indices in enumerate(list_indices):
            # Define region
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            i_img += 1
            logger.info(
                f"Now collecting data from ROI {i_img}/{n_images}."
            )
            # collect ROI data
            ROI_data.append(data_zyx[region].compute())

    ROI_data = np.stack(ROI_data, axis=0)

    # calculate illumination correction profile
    logger.info(f"Now calculating illumination correction for channel"
                f" {channel_name}.")
    basic = BaSiC(
        autosegment=advanced_basicpy_model_params.autosegment,
        autosegment_margin=advanced_basicpy_model_params.autosegment_margin,
        epsilon=advanced_basicpy_model_params.epsilon,
        fitting_mode=advanced_basicpy_model_params.fitting_mode,
        get_darkfield=advanced_basicpy_model_params.get_darkfield,
        max_iterations=advanced_basicpy_model_params.max_iterations,
        max_mu_coef=advanced_basicpy_model_params.max_mu_coef,
        max_reweight_iterations=advanced_basicpy_model_params.max_reweight_iterations,
        max_reweight_iterations_baseline=advanced_basicpy_model_params.max_reweight_iterations_baseline,
        max_workers=advanced_basicpy_model_params.max_workers,
        mu_coef=advanced_basicpy_model_params.mu_coef,
        optimization_tol=advanced_basicpy_model_params.optimization_tol,
        optimization_tol_diff=advanced_basicpy_model_params.optimization_tol_diff,
        resize_mode=advanced_basicpy_model_params.resize_mode,
        #resize_params=advanced_basicpy_model_params.resize_params,
        reweighting_tol=advanced_basicpy_model_params.reweighting_tol,
        rho=advanced_basicpy_model_params.rho,
        smoothness_darkfield=advanced_basicpy_model_params.smoothness_darkfield,
        smoothness_flatfield=advanced_basicpy_model_params.smoothness_flatfield,
        sort_intensity=advanced_basicpy_model_params.sort_intensity,
        sparse_cost_darkfield=advanced_basicpy_model_params.sparse_cost_darkfield,
        working_size=advanced_basicpy_model_params.working_size,
    )


    if np.shape(ROI_data)[0] == 1:
        logger.info(f"ROI data shape is {ROI_data[0, :, :, :].shape}.")
        basic.fit(ROI_data[0, :, :, :])
    else:
        logger.info(f"ROI data shape is {np.squeeze(ROI_data).shape}.")
        basic.fit(np.squeeze(ROI_data))

    logger.info(
        f"Finished calculating illumination correction for channel"
        f" {channel_name}.")

    # save illumination correction model
    logger.info(f"Now saving illumination correction model for channel"
                f" {channel_name}.")
    illum_path = Path(illumination_profiles_folder)
    illum_path.mkdir(parents=True, exist_ok=True)
    filename = illum_path.joinpath(init_args.channel_name)
    basic.save_model(model_dir=filename, overwrite=overwrite)

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=calculate_basicpy_illumination_models,
        logger_name=logger.name,
    )
