# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
#
# Adapted by:
# Adrian Tschan <adrian.tschan@uzh.ch>
#
# This file is based on Fractal code originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
import logging
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from basicpy import BaSiC
from pydantic import validate_call
from scipy.ndimage import zoom

from apx_fractal_task_collection.io_models import CorrectBy
from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)

from fractal_tasks_core.tasks._zarr_utils import _copy_hcs_ome_zarr_metadata
from fractal_tasks_core.tasks._zarr_utils import _copy_tables_from_zarr_url

logger = logging.getLogger(__name__)


def correct(
    img_stack: np.ndarray,
    flatfield: np.ndarray,
    darkfield: Optional[np.ndarray],
    baseline: Optional[int],
):
    """
    Apply illumination correction to all fields of view.

    Corrects a stack of images, using a given illumination profile (e.g. bright
    in the center of the image, dim outside).

    Args:
        img_stack: 4D numpy array (czyx), with dummy size along c.
        flatfield: 2D numpy array (yx)
        darkfield: Optional 2D numpy array (yx)
        baseline: Optional baseline value to be subtracted from the image
    """

    logger.info(f"Start correct, {img_stack.shape}")

    # Check shapes
    if img_stack.shape[0] != 1:
        raise ValueError(
            "Error in illumination_correction:\n"
            f"{img_stack.shape=}\n"
        )
    
    # Resampling flatfield and darkfield if necessary
    if flatfield.shape[-2:] != img_stack.shape[-2:]:
        logger.warning(
            f"Flatfield correction matrix shape does not match image shape in"
            f" x and y. {img_stack[-2:].shape=}\n{flatfield.shape=}. "
            "Resampling ...")
        flatfield = resample_to_shape(flatfield, img_stack.shape[-2:])
        
    if darkfield is not None:
        if darkfield.shape[-2:] != img_stack.shape[-2:]:
            logger.warning(
                "Darkfield correction matrix shape does not match image shape"
                " in x and y. "
                f"{img_stack[2:].shape=}\n{darkfield.shape=} "
                "Resampling ...")
            darkfield = resample_to_shape(darkfield, img_stack.shape[-2:])

    # Store info about dtype
    dtype = img_stack.dtype
    dtype_max = np.iinfo(dtype).max
    
    # Apply the correction matrices
    if darkfield is not None:
        new_img_stack = (img_stack - darkfield) / flatfield
    else:
        new_img_stack = img_stack / flatfield

    # Background subtraction
    if baseline is not None:
        new_img_stack = np.where(new_img_stack > baseline,
                                new_img_stack - baseline,
                                0)

    # Handle edge case: corrected image may have values beyond the limit of
    # the encoding, e.g. beyond 65535 for 16bit images. This clips values
    # that surpass this limit and triggers a warning
    if np.sum(new_img_stack > dtype_max) > 0:
        warnings.warn(
            "Illumination correction created values beyond the max range of "
            f"the current image type. These have been clipped to {dtype_max=}."
        )
        new_img_stack[new_img_stack > dtype_max] = dtype_max
        
    logger.info("End correct")

    # Cast back to original dtype and return
    return new_img_stack.astype(dtype)


def resample_to_shape(img, output_shape, order=3, mode='constant',
                      cval=0.0, prefilter=True):
    '''
    Function resamples image to the desired shape.

    Typically used to up or downscale a pyramid image by
    a potency of 2 (e.g. 0.5, 1, 2 etc.)
    '''
    zoom_values = [o / i for i, o in zip(img.shape, output_shape)]
    return zoom(img, zoom_values, order=order, mode=mode, cval=cval,
                prefilter=prefilter)


@validate_call
def apply_basicpy_illumination_models(
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    # Task-specific arguments
    illumination_profiles_folder: str,
    correct_by: CorrectBy = CorrectBy.channel_label,
    illumination_exceptions: Optional[list[str]] = None,
    darkfield: bool = True,
    subtract_baseline: bool = True,
    fixed_baseline: Optional[int] = None,
    input_ROI_table: str = "FOV_ROI_table",
    overwrite_input: bool = True,
    suffix: str = "_illum_corr",
) -> dict[str, Any]:

    """
    Applies illumination correction to the images in the OME-Zarr.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        illumination_profiles_folder: Path of folder of illumination profiles.
        correct_by: Defines how illumination correction has been calculated. 
            - channel label: illumination correction has been calculated per channel label
            - wavelength id: illumination correction has been calculated per wavelength id
        illumination_exceptions: List of channel labels or wavelength ids that 
            should not be corrected.
        darkfield: If `True`, darkfield correction will be performed.
        subtract_baseline: If `True`, baseline subtraction will be performed.
        fixed_baseline: Set a value if you want to subtract a fixed baseline
            (e.g., 100). If `None` and subtract baseline is 'True', the median
             of the basicpy baseline will be used.
        input_ROI_table: Name of the ROI table that contains the information
            about the location of the individual field of views (FOVs) to
            which the illumination correction shall be applied. Defaults to
            "FOV_ROI_table", the default name Fractal converters give the ROI
            tables that list all FOVs separately. If you generated your
            OME-Zarr with a different converter and used Import OME-Zarr to
            generate the ROI tables, `image_ROI_table` is the right choice if
            you only have 1 FOV per Zarr image and `grid_ROI_table` if you
            have multiple FOVs per Zarr image and set the right grid options
            during import.
        overwrite_input:
            If `True`, the results of this task will overwrite the input image
            data. If false, a new image is generated and the illumination
            corrected data is saved there.
        suffix: What suffix to append to the illumination corrected images.
            Only relevant if `overwrite_input=False`.
    """

    # Define old/new zarrurls
    if overwrite_input:
        zarr_url_new = zarr_url.rstrip("/")
    else:
        zarr_url_new = zarr_url.rstrip("/") + suffix

    # check whether illumination profile folder contains well+channel profiles
    # or only channel profiles

    # get all folders in illumination_profiles_folder
    profile_names = [f for f in Path(illumination_profiles_folder).iterdir()
                     if f.is_dir()]
    sample = str(profile_names[0])

    if "well_" in sample and "_ch_" in sample:
        compute_per_well = True
    else:
        compute_per_well = False

    t_start = time.perf_counter()
    logger.info("Start illumination_correction")
    logger.info(f"  {darkfield=}")
    logger.info(f"  {subtract_baseline=}")
    logger.info(f"  {overwrite_input=}")
    logger.info(f"  {zarr_url=}")
    logger.info(f"  {zarr_url_new=}")

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    logger.info(f"NGFF image has {num_levels=}")
    logger.info(f"NGFF image has {coarsening_xy=}")
    logger.info(
        f"NGFF image has full-res pixel sizes {full_res_pxl_sizes_zyx}"
    )

    # Read channels from .zattrs
    channels: list[OmeroChannel] = get_omero_channel_list(
        image_zarr_path=zarr_url
    )

    if illumination_exceptions is not None:
        # Filter out channels that should not be corrected
        if correct_by == CorrectBy.channel_label:
            channels = [c for c in channels if c.label not in illumination_exceptions]
        else:
            channels = [c for c in channels if c.wavelength_id not in illumination_exceptions]

    num_channels = len(channels)

    # Read FOV ROIs
    FOV_ROI_table = ad.read_zarr(f"{zarr_url}/tables/{input_ROI_table}")

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        FOV_ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, input_ROI_table)

    # Extract image size from FOV-ROI indices. Note: this works at level=0,
    # where FOVs should all be of the exact same size (in pixels)
    ref_img_size = None
    for indices in list_indices:
        img_size = (indices[3] - indices[2], indices[5] - indices[4])
        if ref_img_size is None:
            ref_img_size = img_size
        else:
            if img_size != ref_img_size:
                raise ValueError(
                    "ERROR: inconsistent image sizes in list_indices"
                )
    img_size_y, img_size_x = img_size[:]

    # Lazily load highest-res level from original zarr array
    data_czyx = da.from_zarr(f"{zarr_url}/0")

    # Create zarr for output
    if overwrite_input:
        new_zarr = zarr.open(f"{zarr_url_new}/0")
    else:
        new_zarr = zarr.create(
            shape=data_czyx.shape,
            chunks=data_czyx.chunksize,
            dtype=data_czyx.dtype,
            store=zarr.storage.FSStore(f"{zarr_url_new}/0"),
            overwrite=False,
            dimension_separator="/",
        )
        _copy_hcs_ome_zarr_metadata(zarr_url, zarr_url_new)
        # Copy ROI tables from the old zarr_url to keep ROI tables and other
        # tables available in the new Zarr
        _copy_tables_from_zarr_url(zarr_url, zarr_url_new)

    # Iterate over FOV ROIs
    num_ROIs = len(list_indices)
    for i_c, channel in enumerate(channels):
        # load illumination model
        if correct_by == CorrectBy.channel_label:
            logger.info(
                f"loading illumination model for channel {channel.label}"
            )
            basic = BaSiC()
            if compute_per_well:
                well_id = zarr_url.rsplit("/", 3)[1] + zarr_url.rsplit("/", 3)[2]
                basic = basic.load_model(
                    illumination_profiles_folder +
                    f"/well_{well_id}_ch_{channel.label}")
            else:
                basic = basic.load_model(
                    illumination_profiles_folder + f"/{channel.label}")
        else:
            logger.info(
                f"loading illumination model for channel {channel.wavelength_id}"
            )
            basic = BaSiC()
            if compute_per_well:
                well_id = zarr_url.rsplit("/", 3)[1] + zarr_url.rsplit("/", 3)[2]
                basic = basic.load_model(
                    illumination_profiles_folder +
                    f"/well_{well_id}_ch_{channel.wavelength_id}")
            else:
                basic = basic.load_model(
                    illumination_profiles_folder + f"/{channel.wavelength_id}")
            
        for i_ROI, indices in enumerate(list_indices):
            # Define region
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(i_c, i_c + 1),
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            logger.info(
                f"Now processing ROI {i_ROI + 1}/{num_ROIs} "
                f"for channel {i_c + 1}/{num_channels}"
            )

            if subtract_baseline:
                if fixed_baseline is not None:
                    logger.info(f"Using fixed baseline value {fixed_baseline}")
                    baseline = fixed_baseline
                else:
                    logger.info(f"Using median of basicpy model baseline")
                    baseline = int(np.median(basic.baseline))
            else:
                logger.info("No baseline correction applied")
                baseline = None

            # Execute illumination correction with appropriate darkfield setting
            corrected_fov = correct(
                img_stack=data_czyx[region].compute(),
                flatfield=basic.flatfield,
                darkfield=basic.darkfield if darkfield else None,
                baseline=baseline
            )
            
            # Write to disk
            da.array(corrected_fov).to_zarr(
                url=new_zarr,
                region=region,
                compute=True,
            )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=zarr_url_new,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_czyx.chunksize,
    )

    t_end = time.perf_counter()
    logger.info(f"End illumination_correction, elapsed: {t_end - t_start}")

    if overwrite_input:
        image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=zarr_url_new, origin=zarr_url)]
        )
    return image_list_updates


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=apply_basicpy_illumination_models,
        logger_name=logger.name,
    )
