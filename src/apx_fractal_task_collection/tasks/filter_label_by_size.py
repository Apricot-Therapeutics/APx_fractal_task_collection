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
import dask.array as da

from typing import Optional
from skimage.morphology import remove_small_objects
from skimage.morphology import label
from pydantic import validate_call


from apx_fractal_task_collection.io_models import InitArgsFilterLabelBySize

import fractal_tasks_core
from fractal_tasks_core.utils import rescale_datasets
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.ngff import load_NgffImageMeta


logger = logging.getLogger(__name__)

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

def remove_large_objects(img, max_size):
    '''
    Remove objects larger than a specified size from a label image.

    Args:
        img: The label image as a numpy array.
        max_size: Maximum size of objects to keep.

    Returns:
        The label image with objects larger than max_size removed.
    '''

    img_temp = remove_small_objects(img, min_size=max_size)
    img = np.where(img_temp == 0, img, 0)

    return img


@validate_call
def filter_label_by_size(
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsFilterLabelBySize,
    # Task-specific arguments
    output_label_name: str,
    output_label_image_name: str = "0",
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    level: int = 0,
    overwrite: bool = True,
) -> None:

    """
    Filter objects in a label image by size.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_filter_label_by_size`.
        output_label_name: Name of the output label image.
        output_label_image_name: Name of the zarr image that will
            contain the size filtered label image. Defaults to "0".
        min_size: Minimum size of objects to keep. If None, no minimum size
            filter is applied.
        max_size: Maximum size of objects to keep. If None, no maximum size
            filter is applied.
        level: Resolution of the label image.
            Only tested for level 0.
        overwrite: If True, overwrite existing label image.
    """
    logger.info(f"Filtering label image '{init_args.label_name}' by size.")
    label_image = da.from_zarr(f"{init_args.label_zarr_url}/labels/"
                               f"{init_args.label_name}/{level}")

    output_zarr_url = f"{init_args.label_zarr_url.rsplit('/', 1)[0]}/" \
                      f"{output_label_image_name}"

    ngff_image_meta = load_NgffImageMeta(init_args.label_zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy

    label_img = label_image.compute()
    if min_size is not None:
        logger.info(f"Removing objects smaller than {min_size} pixels")
        label_img = remove_small_objects(label_img, min_size=min_size)
    if max_size is not None:
        logger.info(f"Removing objects larger than {max_size} pixels")
        label_img = remove_large_objects(label_img, max_size=max_size)

    # relabel to preserve consecutive labels
    new_label_image = label(label_img)

    if output_label_name == init_args.label_name:

        logger.info(f"Overwriting label image '{label_name}' "
                    f"with size filtered image.")

    else:

        logger.info(f"Creating new label image '{output_label_name}'")

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
        reference_level=0,
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

    image_group = zarr.group(output_zarr_url)
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
    out = f"{output_zarr_url}/labels/{output_label_name}/0"
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
        overwrite=overwrite,
        dimension_separator="/",
    )

    logger.info(
        f"mask will have shape {label_image.shape} "
        f"and chunks {label_image.chunks}"
    )

    # Compute and store 0-th level to disk
    da.array(new_label_image).to_zarr(
        url=mask_zarr,
        compute=True,
    )

    logger.info(
        f"Size filtering done for {out}."
        "now building pyramids."
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=f"{output_zarr_url}/labels/{output_label_name}",
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.max,
    )

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=filter_label_by_size,
        logger_name=logger.name,
    )

