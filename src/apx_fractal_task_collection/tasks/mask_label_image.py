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
# This file is based on Fractal code originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.

import logging

import dask.array as da
import fractal_tasks_core
import numpy as np
import zarr

from apx_fractal_task_collection.io_models import InitArgsMaskLabelImage

from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.utils import rescale_datasets
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid

from pydantic import validate_call


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

@validate_call
def mask_label_image(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsMaskLabelImage,
    # Task-specific arguments:
    output_label_name: str,
    level: int = 0,
    overwrite: bool = True,
) -> None:
    """
    Applies a mask to a label image.

    Takes two label images (or a label image and a binary mask) and replaces
    all values in the first label image with 0 where the second label image has
    values = 0.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by `init_clip_label_image`.
        output_label_name: Name of the output label image.
        level: Resolution of the label image.
            Only tested for level 0.
        overwrite: If True, overwrite existing label image.
    """
    # load the label image and the mask label image
    label_image = da.from_zarr(f"{init_args.label_zarr_url}/"
                               f"labels/{init_args.label_name}/{level}")
    mask = da.from_zarr(f"{init_args.mask_zarr_url}/"
                        f"labels/{init_args.mask_name}/{level}")

    # prepare label image
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy


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

    out = f"{zarr_url}/labels/{output_label_name}/{level}"
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

    # mask the label image
    new_label_image = np.where(mask > 0, label_image, 0)

    # Compute and store 0-th level to disk
    da.array(new_label_image).to_zarr(
        url=mask_zarr,
        compute=True,
    )

    logger.info(
        f"Masking done for {out}."
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
        task_function=mask_label_image,
        logger_name=logger.name,
    )

