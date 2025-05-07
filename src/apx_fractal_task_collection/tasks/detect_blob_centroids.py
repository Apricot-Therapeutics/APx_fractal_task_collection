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
from skimage.feature import blob_log
from skimage.morphology import label

from apx_fractal_task_collection.io_models import InitArgsDetectBlobCentroids

from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.utils import rescale_datasets
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.channels import get_channel_from_image_zarr

from pydantic import validate_call


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

def blob_detection(intensity_image,
                   min_sigma=1,
                   max_sigma=10,
                   num_sigma=3,
                   threshold=0.002):
    """
    Detect blobs in an intensity image.
    Args:
        intensity_image:

    Returns:
        labels: 3D numpy array with the same shape as the input image,
        where each pixel is assigned to a blob centroid.
    """

    labels = np.zeros(intensity_image.shape, dtype='uint32')
    blobs = blob_log(image=intensity_image,
                     min_sigma=min_sigma,
                     max_sigma=max_sigma,
                     num_sigma=num_sigma,
                     threshold=threshold).astype('uint16')

    labels[blobs[:, 0], blobs[:, 1], blobs[:, 2]] = 1
    labels = label(labels)

    return labels

@validate_call
def detect_blob_centroids(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsDetectBlobCentroids,
    # Task-specific arguments:
    ROI_table_name: str,
    min_sigma: int = 1,
    max_sigma: int = 10,
    num_sigma: int = 3,
    threshold: float = 0.002,
    output_label_name: str,
    level: int = 0,
    relabeling: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Detects blob centroids in an intensity image and stores the result as a
    label image..

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_detect_blob_centroids`.
        ROI_table_name: Name of the table containing the ROIs.
        output_label_name: Name of the output label image.
        level: Resolution of the label image to calculate overlap.
            Only tested for level 0.
        relabeling: If True, relabel the label image to keep unique label
            across all ROIs
        overwrite: If True, overwrite existing label image.
    """

    tmp_channel: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=init_args.channel_zarr_url,
        wavelength_id=None,
        label=init_args.channel_label
    )

    ind_channel = tmp_channel.index
    data_zyx = \
        da.from_zarr(f"{init_args.channel_zarr_url}/0")[ind_channel]


    # prepare label image
    ngff_image_meta = load_NgffImageMeta(zarr_url)
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
    ROI_table = ad.read_zarr(f"{init_args.channel_zarr_url}/"
                             f"tables/{ROI_table_name}")
    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, "registered_well_ROI_table")
    num_ROIs = len(list_indices)

    # Counters for relabeling
    if relabeling:
        num_labels_tot = 0

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
        new_label_image = blob_detection(
            data_zyx[region].compute(),
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
        )

        # Shift labels and update relabeling counters
        if relabeling:
            num_labels_roi = np.max(new_label_image)
            new_label_image[new_label_image > 0] += num_labels_tot
            num_labels_tot += num_labels_roi

            # Write some logs
            logger.info(f"ROI {indices}, {num_labels_roi=}, {num_labels_tot=}")

            # Check that total number of labels is under control
            if num_labels_tot > np.iinfo(label_dtype).max:
                raise ValueError(
                    "ERROR in re-labeling:"
                    f"Reached {num_labels_tot} labels, "
                    f"but dtype={label_dtype}"
                )


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
        zarrurl=out.rsplit("/", 1)[0],
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.max,
    )



if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=detect_blob_centroids,
        logger_name=logger.name,
    )

