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
Initializes the parallelization list Correct Chromatic Shift task.
"""

import logging
from typing import Any
import dask.array as da
import SimpleITK as sitk
import anndata as ad
from skimage import exposure
import numpy as np
from scipy.ndimage import gaussian_filter

from fractal_tasks_core.channels import (get_omero_channel_list, 
                                         get_channel_from_image_zarr)
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
    check_valid_ROI_indices
)

from pydantic import validate_call

logger = logging.getLogger(__name__)


def get_channel_image_from_zarr(zarr_urls, channel_label, channel_image_name):
    '''
    Get the image data for a specific channel from an OME-Zarr file. This
    function collects all images across all wells and returns them as a single
    stack.

    Args:
        zarrurl: Path to the OME-Zarr file.
        channel_label: Label of the channel to extract.
        channel_image_name: Name of the zarr image containing the channel

    Returns:
        The image data for the specified channel as dask array
    '''

    img = []
    channel_zarr_urls = [zarr_url for zarr_url in zarr_urls if
                         zarr_url.endswith(f"/{channel_image_name}")]

    for zarr_url in channel_zarr_urls:

        tmp_channel: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=zarr_url,
            wavelength_id=None,
            label=channel_label
        )
        ind_channel = tmp_channel.index
        data_zyx = \
            da.from_zarr(f"{zarr_url}/0")[ind_channel]

        img.append(data_zyx)
    return np.stack(img), tmp_channel.wavelength_id


def correct_background(image_stack):

    background = gaussian_filter(np.mean(image_stack, axis=0, keepdims=True),
                                 sigma=10).astype('uint16')
    corrected_stack = np.where(background < image_stack,
                               image_stack - background, 0)
    corrected_img = np.sum(
        corrected_stack, axis=0, keepdims=True).astype('uint16')

    return corrected_img


def register_channel(channel_image, ref_image):

    ref = correct_background(ref_image)
    img = correct_background(channel_image)
    img = exposure.match_histograms(img, ref).astype('uint16')

    if len(np.squeeze(ref).shape) == 2:
        ref = sitk.GetImageFromArray(np.squeeze(ref))
        ref.SetOrigin([0, 0])

        img = sitk.GetImageFromArray(np.squeeze(img))
        img.SetOrigin([0, 0])

    elif len(np.squeeze(ref).shape) == 3:
        ref = sitk.GetImageFromArray(np.squeeze(ref))
        ref.SetOrigin([0, 0, 0])

        img = sitk.GetImageFromArray(np.squeeze(img))
        img.SetOrigin([0, 0, 0])

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(ref)
    elastixImageFilter.SetMovingImage(img)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.Execute()

    map = elastixImageFilter.GetTransformParameterMap()[0]

    return map

@validate_call
def init_correct_chromatic_shift(
    *,
    # Fractal parameters
    zarr_urls: list[str],
    zarr_dir: str,
    # Core parameters
    reference_zarr_image: str,
    reference_channel_label: str,
) -> dict[str, list[dict[str, Any]]]:
    """
    Initialized Correct Chromatic Shift task
    
    This task prepares a parallelization list of all zarr_urls that need to be
    used to perform chromatic shift correction based on reference images of
    (for example) fluorescent beads.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        reference_zarr_image: Path to the reference zarr image used for
            chromatic shift correction. Needs to exist in OME-Zarr file.
        reference_channel_label: Label of the channel in the reference zarr
            image to which the other channels will be corrected.

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    logger.info(
        f"Running `init_correct_chromatic_shift` for {zarr_urls=}"
    )

    ref_channel_img, ref_wavelength_id = get_channel_image_from_zarr(
        zarr_urls=zarr_urls,
        channel_label=reference_channel_label,
        channel_image_name=reference_zarr_image
    )

    # get the reference zarr image and calculate the transformation map
    ref_zarr_url = f"{zarr_urls[0].rsplit('/', 1)[0]}/{reference_zarr_image}"

    # collect the correction channels
    correction_channels = get_omero_channel_list(image_zarr_path=ref_zarr_url)
    # make sure that reference channel is not in correction channels
    correction_channels = [channel for channel in
                           correction_channels if channel.label !=
                           reference_channel_label]

    # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(ref_zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)

    # Read FOV ROIs
    FOV_ROI_table = ad.read_zarr(f"{ref_zarr_url}/tables/FOV_ROI_table")

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
        img_size = (indices[3] - indices[2], indices[5] - indices[4])
        if ref_img_size is None:
            ref_img_size = img_size
        else:
            if img_size != ref_img_size:
                raise ValueError(
                    "ERROR: inconsistent image sizes in list_indices"
                )

    # Iterate over FOV ROIs
    num_ROIs = len(list_indices)

    # get transformation maps
    transformation_maps = {}
    for corr_channel in correction_channels:
        ROI_maps = {}

        channel_image, wavelength_id = get_channel_image_from_zarr(
            zarr_urls=zarr_urls,
            channel_label=corr_channel.label,
            channel_image_name=reference_zarr_image
        )

        for i_ROI, indices in enumerate(list_indices):
            # Define region
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(0, channel_image.shape[0]),
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            logger.info(
                f'calculating correction map for channel {corr_channel} '
                f'for ROI {i_ROI + 1}/{num_ROIs}')

            transformation_map = register_channel(
                channel_image[region].compute(),
                ref_channel_img[region].compute())
            ROI_maps[i_ROI] = transformation_map
        transformation_maps[corr_channel.wavelength_id] = ROI_maps

    # Create the parallelization list
    parallelization_list = []

    for zarr_url in zarr_urls:
        parallelization_list.append(
            dict(
                zarr_url=zarr_url,
                init_args=dict(
                    zarr_urls=[],
                    transformation_maps=transformation_maps,
                ),
            )
        )

    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=init_correct_chromatic_shift,
        logger_name=logger.name,
    )