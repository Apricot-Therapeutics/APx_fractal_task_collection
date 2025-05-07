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

import dask.array as da
import fractal_tasks_core
import numpy as np
import zarr
import anndata as ad
from skimage.segmentation import expand_labels


from apx_fractal_task_collection.io_models import InitArgsExpandLabels

from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.utils import rescale_datasets
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)

from pydantic import validate_call


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


@validate_call
def expand_labels_skimage(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsExpandLabels,
    # Task-specific arguments:
    ROI_table_name: str,
    distance: int = 30,
    #spacing: int = 1,
    output_label_name: str,
    level: int = 0,
    overwrite: bool = True,
) -> None:
    """
    Expands labels in a label image by "distance" pixels without overlapping.

    Takes a primary label image and expands it by a certain distance. Direct
    implementation of skimage.segmentation.expand_labels.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_segment_secondary_objects`.
        ROI_table_name: Name of the table containing the ROIs.
        distance: Euclidean distance in pixels by which to grow the labels.
            Default is one.
        spacing: Spacing of elements along each dimension. If a sequence, must
            be of length equal to the input rank; if a single number,
             this is used for all axes. If not specified, a grid spacing
             of unity is implied.
        output_label_name: Name of the output label image.
        level: Resolution of the label image to calculate overlap.
            Only tested for level 0.
        overwrite: If True, overwrite existing label image.
    """

    # load label image
    label_image = da.from_zarr(
        f"{init_args.label_zarr_url}/labels/{init_args.label_name}/{level}")


    # prepare label image
    ngff_image_meta = load_NgffImageMeta(init_args.label_zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)


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

    image_group = zarr.group(zarr_url)
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

    out = f"{zarr_url}/labels/{output_label_name}/0"
    logger.info(f"Output label path: {out}")
    store = zarr.storage.FSStore(str(out))
    label_dtype = np.uint32

    shape = label_image.shape
    if len(shape) == 2:
        shape = (1, *shape)
    chunks = label_image.chunksize
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
        f"mask will have shape {label_image.shape} "
        f"and chunks {label_image.chunksize}"
    )

    # load ROI table
    ROI_table = ad.read_zarr(
        f"{init_args.label_zarr_url}/tables/{ROI_table_name}")

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

        # perform expansion
        new_label_image = expand_labels(
            np.squeeze(label_image[region].compute()),
            distance=distance,
            #spacing=spacing,
        )

        logger.info(f"Finished expansion for ROI {i_ROI + 1}/{num_ROIs}.")
        logger.info(f"New label image has shape {new_label_image.shape}.")

        if len(new_label_image.shape) == 2:
            new_label_image = np.expand_dims(new_label_image, axis=0)

        # Compute and store 0-th level to disk
        da.array(new_label_image).to_zarr(
            url=mask_zarr,
            region=region,
            compute=True,
        )

    logger.info(
        f"Label expansion done for {out}."
        "now building pyramids."
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{zarr_url}/labels/{output_label_name}",
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.max,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=expand_labels_skimage,
        logger_name=logger.name,
    )

