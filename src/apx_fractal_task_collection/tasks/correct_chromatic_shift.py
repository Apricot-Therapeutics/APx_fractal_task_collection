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

from pydantic import validate_call

from apx_fractal_task_collection.io_models import InitArgsCorrectChromaticShift

from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.tasks._zarr_utils import _copy_hcs_ome_zarr_metadata
from fractal_tasks_core.tasks._zarr_utils import _copy_tables_from_zarr_url

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


def register_image(img, transformation_map):

    if len(np.squeeze(img).shape) == 2:
        img = sitk.GetImageFromArray(np.squeeze(img))
        img.SetOrigin([0, 0])

    elif len(np.squeeze(img).shape) == 3:
        img = sitk.GetImageFromArray(np.squeeze(img))
        img.SetOrigin([0, 0, 0])

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(transformation_map)

    transformixImageFilter.SetMovingImage(img)
    transformixImageFilter.Execute()

    result = transformixImageFilter.GetResultImage()
    result = sitk.GetArrayFromImage(result).astype('uint16')

    return result


@validate_call
def correct_chromatic_shift(
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsCorrectChromaticShift,
    # Task-specific arguments
    overwrite_input: bool = True,
    suffix: str = "_chromatic_shift_corr",
) -> None:

    """
    Correct chromatic shift based on reference images (for example fluorescent
    beads) and apply it to all images.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_correct_chromatic_shift.py`.
        overwrite_input:
            If `True`, the results of this task will overwrite the input image
            data. If false, a new image is generated and the chromatic shift
            corrected data is saved there.
        suffix: What suffix to append to the illumination corrected images.
            Only relevant if `overwrite_input=False`.
    """

    # Define old/new zarrurls
    if overwrite_input:
        zarr_url_new = zarr_url.rstrip("/")
    else:
        zarr_url_new = zarr_url.rstrip("/") + suffix

    logger.info("Correcting chromatic shift based on reference images.")

     # Read attributes from NGFF metadata
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)

    # Read FOV ROIs
    FOV_ROI_table = ad.read_zarr(f"{zarr_url}/tables/FOV_ROI_table")

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

    # load image data
    data_czyx = \
        da.from_zarr(f"{zarr_url}/0")

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

    # apply transformation maps to all images
    channel_list = get_omero_channel_list(image_zarr_path=zarr_url)
    channel_list = [c for c in channel_list if
                    c.wavelength_id in init_args.transformation_maps.keys()]
    for channel in channel_list:
        tmp_channel: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=zarr_url,
            wavelength_id=None,
            label=channel.label
        )

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

            print(f"data_czyx shape: {data_czyx[region].shape}")
            corrected_fov[0, :, :, :] = register_image(
                data_czyx[region],
                init_args.transformation_maps[tmp_channel.wavelength_id][i_ROI])

            # Write to disk
            da.array(corrected_fov).to_zarr(
                url=new_zarr,
                region=region,
                compute=True,
            )

        # Starting from on-disk highest-resolution data, build and write
        # to disk a pyramid of coarser levels
        build_pyramid(
            zarrurl=zarr_url_new,
            overwrite=True,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            chunksize=data_czyx.chunksize,
        )

        if overwrite_input:
            image_list_updates = dict(
                image_list_updates=[dict(zarr_url=zarr_url)])
        else:
            image_list_updates = dict(
                image_list_updates=[
                    dict(zarr_url=zarr_url_new, origin=zarr_url)]
            )
        return image_list_updates

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=correct_chromatic_shift,
        logger_name=logger.name,
    )

