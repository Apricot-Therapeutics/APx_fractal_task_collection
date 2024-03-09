"""
# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
# Joel Lüthi  <joel.luethi@fmi.ch>
#
# Adapted by:
# Adrian Tschan <adrian.tschan@uzh.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import dask.array as da
import fractal_tasks_core
import numpy as np
import zarr
import anndata as ad
import mahotas as mh
from skimage.filters import gaussian
from skimage.morphology import area_closing
from typing import Optional

from apx_fractal_task_collection.utils import (
    get_channel_image_from_well,
    get_label_image_from_well,
)

from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.utils import rescale_datasets
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.channels import ChannelInputModel
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)

from pydantic.decorator import validate_arguments


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

def watershed(intensity_image, label_image,
              min_threshold, max_threshold,
              gaussian_blur, contrast_threshold,
              mask):

    # if there are no labels, return a zero array
    if np.max(label_image) == 0:
        return np.zeros(label_image.shape, dtype='uint32')

    else:
        # only retain intensity in masked region
        intensity_image[mask == 0] = 0

        # apply gaussian blur
        if gaussian_blur is not None:
            intensity_image = gaussian(intensity_image, gaussian_blur,
                                       preserve_range=True).astype('uint16')

        # get the maximum label value in the primary label image
        max_label = np.max(np.unique(label_image[np.nonzero(label_image)]))

        # get the background mask and label its regions
        # (will later be used as background seeds)
        background_mask = mh.thresholding.bernsen(
            intensity_image, 5, contrast_threshold
        )

        if min_threshold is not None:
            background_mask[intensity_image < min_threshold] = True

        if max_threshold is not None:
            background_mask[intensity_image > max_threshold] = False

        background_label_image = mh.label(background_mask)[0]
        background_label_image[background_mask] += max_label

        # add background seeds to primary label image
        labels = label_image + background_label_image

        # perform watershed
        regions = mh.cwatershed(np.invert(intensity_image), labels)
        # remove regions that are not expansions of primary objects
        regions[regions > max_label] = 0

        return regions.astype('uint32')

@validate_arguments
def segment_secondary_objects(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments:
    label_image_name: str,
    channel: ChannelInputModel,
    ROI_table_name: str,
    min_threshold: Optional[int] = None,
    max_threshold: Optional[int] = None,
    gaussian_blur: Optional[int] = None,
    fill_holes_area: Optional[int] = None,
    contrast_threshold: int = 5,
    mask: Optional[str] = None,
    output_label_cycle: int,
    output_label_name: str,
    level: int = 0,
    overwrite: bool = True,
) -> None:
    """
    Segments secondary objects based on primary labels and intensity image.

    Takes a primary label image and an intensity image and calculates secondary
    labels based on watershed segmentation.

    Args:
        input_paths: Path to the parent folder of the NGFF image.
            This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This argument is not used in this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path of the NGFF image, relative to `input_paths[0]`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This argument is not used in this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_image_name: Name of the label image that contains the seeds.
            Needs to exist in OME-Zarr file.
        channel: Name of the intensity image used for watershed.
            Needs to exist in OME-Zarr file.
        ROI_table_name: Name of the table containing the ROIs.
        min_threshold: Minimum threshold for the background definition.
        max_threshold: Maximum threshold for the background definition.
        gaussian_blur: Sigma for gaussian blur.
        fill_holes_area: Area threshold for filling holes after watershed.
        contrast_threshold: Contrast threshold for background definition.
        mask: label image to use as mask. Only areas where the mask is
            non-zero will be considered for the watershed.
        output_label_cycle: indicates in which acquisition to store the result.
        output_label_name: Name of the output label image.
        level: Resolution of the label image to calculate overlap.
            Only tested for level 0.
        overwrite: If True, overwrite existing label image.
    """

    in_path = Path(input_paths[0])
    well_url = in_path.joinpath(component)

    # update components
    output_component = component + "/" + str(output_label_cycle)

    label_image, label_image_path = get_label_image_from_well(
        well_url, label_image_name, level)
    data_zyx, intensity_image_path = get_channel_image_from_well(
        well_url, channel.label, level)

    if mask is not None:
        mask_label, mask_image_path = get_label_image_from_well(
            well_url, mask, level)
    else:
        mask_label = da.ones(label_image.shape, dtype='uint32')

        # prepare label image
    ngff_image_meta = load_NgffImageMeta(intensity_image_path)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    actual_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)


    # Rescale datasets (only relevant for level>0)
    if ngff_image_meta.axes_names[0] != "c":
        raise ValueError(
            "Cannot set `remove_channel_axis=True` for multiscale "
            f"metadata with axes={ngff_image_meta.axes_names}. "
            'First axis should have name "c".'
        )
    new_datasets = rescale_datasets(
        datasets=[ds.dict() for ds in ngff_image_meta.datasets],
        coarsening_xy=coarsening_xy,
        reference_level=level,
        remove_channel_axis=True,
    )

    label_attrs = {
        "image-label": {
            "version": __OME_NGFF_VERSION__,
            "source": {"image": "../../"},
        },
        "multiscales": [
            {
                "name": output_label_name,
                "version": __OME_NGFF_VERSION__,
                "axes": [
                    ax.dict()
                    for ax in ngff_image_meta.multiscale.axes
                    if ax.type != "channel"
                ],
                "datasets": new_datasets,
            }
        ],
    }

    image_group = zarr.group(in_path.joinpath(output_component))
    label_group = prepare_label_group(
        image_group,
        output_label_name,
        overwrite=overwrite,
        label_attrs=label_attrs,
        logger=logger,
    )

    logger.info(
        f"Helper function `prepare_label_group` returned {label_group=}"
    )
    out = f"{output_path}/{output_component}/labels/{output_label_name}/0"
    logger.info(f"Output label path: {out}")
    store = zarr.storage.FSStore(str(out))
    label_dtype = np.uint32

    shape = data_zyx.shape
    if len(shape) == 2:
        shape = (1, *shape)
    chunks = data_zyx.chunksize
    if len(chunks) == 2:
        chunks = (1, *chunks)
    mask_zarr = zarr.create(
        shape=shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=False,
        dimension_separator="/",
    )

    logger.info(
        f"mask will have shape {data_zyx.shape} "
        f"and chunks {data_zyx.chunks}"
    )

    # load ROI table
    ROI_table = ad.read_zarr(label_image_path.joinpath("tables",
                                              ROI_table_name))
    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, "registered_well_ROI_table")
    num_ROIs = len(list_indices)

    # Loop over the list of indices and perform the secondary segmentation
    for i_ROI, indices in enumerate(list_indices):

        # Define region
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )
        logger.info(
            f"Now processing ROI {i_ROI + 1}/{num_ROIs} from ROI table"
            f" {ROI_table_name}."
        )

        # perform watershed
        new_label_image = watershed(
            np.squeeze(data_zyx[region].compute()),
            np.squeeze(label_image[region].compute()),
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            gaussian_blur=gaussian_blur,
            contrast_threshold=contrast_threshold,
            mask=np.squeeze(mask_label[region].compute()),
        )

        # fill holes in label image
        if fill_holes_area is not None:
            new_label_image = area_closing(new_label_image,
                                           area_threshold=fill_holes_area)

        new_label_image = np.expand_dims(new_label_image, axis=0)

        # Compute and store 0-th level to disk
        da.array(new_label_image).to_zarr(
            url=mask_zarr,
            region=region,
            compute=True,
        )


    logger.info(
        f"Secondary segmentation done for {out}."
        "now building pyramids."
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{output_path}/{output_component}/labels/{output_label_name}",
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.max,
    )



if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=segment_secondary_objects,
        logger_name=logger.name,
    )
