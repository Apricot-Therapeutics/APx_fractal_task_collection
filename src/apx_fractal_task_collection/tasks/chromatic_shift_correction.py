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
import zarr
import numpy as np
import SimpleITK as sitk
import dask.array as da
import anndata as ad

from typing import Any
from typing import Sequence
from skimage import exposure
from pathlib import Path
from scipy.ndimage import gaussian_filter
from pydantic.decorator import validate_arguments

from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)

logger = logging.getLogger(__name__)


def get_channel_image_from_zarr(zarrurl, channel_label):
    '''
    Get the image data for a specific channel from an OME-Zarr file. This
    function collects all images across all wells and returns them as a single
    stack.

    Args:
        zarrurl: Path to the OME-Zarr file.
        channel_label: Label of the channel to extract.

    Returns:
        The image data for the specified channel as dask array
    '''

    img = []
    plate_group = zarr.open(zarrurl, mode='r')
    for well in plate_group.attrs['plate']['wells']:
        well_group = plate_group[well['path']]
        for image in well_group.attrs['well']['images']:
            img_zarr_path = zarrurl.joinpath(well['path'], image['path'])
            channel_list = get_omero_channel_list(
                image_zarr_path=img_zarr_path)

            if channel_label in [c.label for c in channel_list]:
                tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                    image_zarr_path=img_zarr_path,
                    wavelength_id=None,
                    label=channel_label
                )

                ind_channel = tmp_channel.index
                data_zyx = \
                    da.from_zarr(img_zarr_path.joinpath('0'))[ind_channel]

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


def register_channel(channel_images, ref_images):

    ref = correct_background(ref_images)
    img = correct_background(channel_images)
    img = exposure.match_histograms(img, ref).astype('uint16')

    ref = sitk.GetImageFromArray(ref[0, 0, :, :])
    ref.SetOrigin([0, 0])

    img = sitk.GetImageFromArray(img[0, 0, :, :])
    img.SetOrigin([0, 0])

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(ref)
    elastixImageFilter.SetMovingImage(img)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))
    elastixImageFilter.Execute()

    map = elastixImageFilter.GetTransformParameterMap()[0]

    return map


def register_image(img, transformation_map):

    img = sitk.GetImageFromArray(np.squeeze(img))
    img.SetOrigin([0, 0])

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transformation_map)

    transformixImageFilter.SetMovingImage(img)
    transformixImageFilter.Execute()

    result = transformixImageFilter.GetResultImage()
    result = sitk.GetArrayFromImage(result).astype('uint16')

    return result


@validate_arguments
def chromatic_shift_correction(
    *,
    # Standard arguments
    input_paths: Sequence[str],
    output_path: str,
    metadata: dict[str, Any],
    component: str,
    # Task-specific arguments
    correction_channel_labels: Sequence[str],
    reference_channel_label: str
) -> None:

    """
    Correct chromatic shift based on reference images (for example fluorescnet
    beads) and apply it to all images.

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
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/0"`
            (standard argument for Fractal tasks, managed by Fractal server).
        correction_channel_labels: List of channel labels that contain
            images used for chromatic shift correction. (This can include
            or exclude the reference channel.)
        reference_channel_label: Label of the channel that is used as
            reference. (This channel is not corrected.)
    """
    logger.info("Correcting chromatic shift based on reference images.")
    in_path = Path(input_paths[0])
    zarrurl = in_path.joinpath(in_path.joinpath(component))

    img_group = zarr.open(zarrurl.joinpath('0'))
    # make sure that reference channel is not in correction channels
    correction_channel_labels = [c for c in correction_channel_labels if
                           reference_channel_label not in c]

     # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarrurl)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)

    # Read FOV ROIs
    FOV_ROI_table = ad.read_zarr(f"{zarrurl}/tables/FOV_ROI_table")

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
    img_size_y, img_size_x = img_size[:]

    # Iterate over FOV ROIs
    num_ROIs = len(list_indices)

    # get reference images
    ref_images, ref_wavelength_id = \
        get_channel_image_from_zarr(zarrurl.parents[2], reference_channel_label)

    # get transformation maps
    transformation_maps = {}
    for corr_channel in correction_channel_labels:
        ROI_maps = {}
        channel_images, wavelength_id = \
            get_channel_image_from_zarr(zarrurl.parents[2], corr_channel)
        for i_ROI, indices in enumerate(list_indices):
            # Define region
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(0, channel_images.shape[0]),
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            logger.info(f'calculating correction map for channel {corr_channel} '
                        f'for ROI {i_ROI+1}/{num_ROIs}')

            transformation_map = register_channel(
                channel_images[region].compute(),
                ref_images[region].compute())
            ROI_maps[i_ROI] = transformation_map
        transformation_maps[wavelength_id] = ROI_maps

    # apply transformation maps to all images
    channel_list = get_omero_channel_list(image_zarr_path=zarrurl)
    channel_list = [c for c in channel_list if
                    c.wavelength_id != ref_wavelength_id]
    for channel in channel_list:
        tmp_channel: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=zarrurl,
            wavelength_id=None,
            label=channel.label
        )
        data_czyx = \
            da.from_zarr(zarrurl.joinpath('0'))

        for i_ROI, indices in enumerate(list_indices):
            # Define region
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(tmp_channel.index, tmp_channel.index + 1),
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            logger.info(f'applying correction map for channel {channel.label} '
                        f'and ROI {i_ROI+1}/{num_ROIs}')

            corrected_fov = da.zeros(data_czyx[region].shape,
                                     dtype=data_czyx.dtype)
            corrected_fov[0, 0, :, :] = register_image(
                data_czyx[region],
                transformation_maps[tmp_channel.wavelength_id][i_ROI])

            # Write to disk
            da.array(corrected_fov).to_zarr(
                url=img_group,
                region=region,
                compute=True,
            )

        # Starting from on-disk highest-resolution data, build and write
        # to disk a pyramid of coarser levels
        build_pyramid(
            zarrurl=zarrurl,
            overwrite=True,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            chunksize=data_czyx.chunksize,
        )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=chromatic_shift_correction,
        logger_name=logger.name,
    )

