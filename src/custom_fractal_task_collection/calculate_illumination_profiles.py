"""
Calculcates illumination correction profiles based on a random sample of images
"""
import logging
import random
import time
import warnings
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Sequence

import anndata as ad
from basicpy import BaSiC
import dask.array as da
import numpy as np
import pandas as pd
import zarr
from pydantic.decorator import validate_arguments

from fractal_tasks_core.lib_channels import get_omero_channel_list
from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_ngff import load_NgffImageMeta
from fractal_tasks_core.lib_channels import get_channel_from_image_zarr
from fractal_tasks_core.lib_regions_of_interest import check_valid_ROI_indices
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)

logger = logging.getLogger(__name__)


@validate_arguments
def calculate_illumination_profiles(
    *,
    # Standard arguments
    input_paths: Sequence[str],
    output_path: str,
    metadata: dict[str, Any],
    # Task-specific arguments
    illumination_profiles_folder: str,
    n_images: int = 150,
    overwrite: bool = False,
) -> dict[str, Any]:

    """
    Calculates illumination correction profiles based on a random sample
    of images of for each channel.

    Args:
        input_paths: List of input paths where the image data is stored as
            OME-Zarrs. Should point to the parent folder containing one or many
            OME-Zarr files, not the actual OME-Zarr file. Example:
            `["/some/path/"]`. This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: Path were the output of this task is stored. Examples:
            `"/some/path/"` => puts the new OME-Zarr file in the same folder as
            the input OME-Zarr file; `"/some/new_path"` => puts the new
            OME-Zarr file into a new folder at `/some/new_path`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles_folder: Path to folder where illumination
            profiles will be saved.
        n_images: Number of images to sample for illumination correction.
        overwrite: If True, overwrite existing illumination profiles.
    """


    logger.info(f"Calculating illumination profiles based on {n_images} randomly sampled images.")
    in_path = Path(input_paths[0])

    list_plates = list(Path(in_path).glob("*.zarr"))
    logger.info(f"{list_plates=}")
    zarrurl = list_plates[0].as_posix()
    group = zarr.open_group(zarrurl, mode="r+")

    # get list of all channels
    well_path = group.attrs['plate']["wells"][0]["path"]
    well_group = zarr.open_group(f"{zarrurl}/{well_path}", mode="r+")

    channel_acquisition_dict = {}
    for acquisition in well_group.attrs['well']["images"]:
        omero_channels = get_omero_channel_list(
            image_zarr_path=f"{zarrurl}/{well_path}/{acquisition['path']}")
        channel_acquisition_dict.update(
            {channel.label: acquisition["path"] for channel in omero_channels})

    for channel, acquisition in channel_acquisition_dict.items():
        logger.info(
            f"Calculating illumination profile for channel {channel}.")
        wells_to_process = pd.DataFrame(
            random.choices(group.attrs['plate']["wells"], k=n_images))
        wells_to_process = wells_to_process.groupby("path").count()

        image_zarr_path = f"{zarrurl}/{wells_to_process.index[0]}/{acquisition}"

        # Read attributes from NGFF metadata
        ngff_image_meta = load_NgffImageMeta(image_zarr_path)
        coarsening_xy = ngff_image_meta.coarsening_xy
        full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(
            level=0)

        ROI_data = []
        # collect all images for on channel
        i_img = 0
        for well in wells_to_process.iterrows():
            well_n_images = well[1][0]
            logger.info(f"Now collecting {well_n_images} images for channel {channel} from well {well[0]}")

            image_zarr_path = f"{zarrurl}/{well[0]}/{acquisition}"
            # Read FOV ROIs
            FOV_ROI_table = ad.read_zarr(
                f"{image_zarr_path}/tables/FOV_ROI_table")

            # Create list of indices for 3D FOVs spanning the entire Z direction
            list_indices = convert_ROI_table_to_indices(
                FOV_ROI_table,
                level=0,
                coarsening_xy=coarsening_xy,
                full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
            )
            check_valid_ROI_indices(list_indices, "FOV_ROI_table")
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

            tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=image_zarr_path,
                wavelength_id=None,
                label=channel,
            )
            ind_channel = tmp_channel.index
            data_zyx = \
            da.from_zarr(f"{image_zarr_path}/0")[
                ind_channel]

            list_indices = random.sample(list_indices, well_n_images)

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
        logger.info(f"Now calculating illumination correction for channel {channel}.")
        basic = BaSiC(get_darkfield=True, smoothness_flatfield=1)
        basic.fit(np.squeeze(ROI_data))
        logger.info(
            f"Finished calculating illumination correction for channel {channel}.")

        # save illumination correction model
        logger.info(f"Now saving illumination correction model for channel {channel}.")
        filename = illumination_profiles_folder + f"/{channel}"
        basic.save_model(model_dir=filename, overwrite=overwrite)

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=calculate_illumination_profiles,
        logger_name=logger.name,
    )
