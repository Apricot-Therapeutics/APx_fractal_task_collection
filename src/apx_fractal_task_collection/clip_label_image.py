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
from pathlib import Path
from typing import Any, Dict, Sequence

import dask.array as da
import fractal_tasks_core
import numpy as np
import zarr
from typing import Optional
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.utils import rescale_datasets
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid

from pydantic.decorator import validate_arguments


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

@validate_arguments
def clip_label_image(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: Dict[str, Any],
    # Task-specific arguments:
    label_image_name: str,
    clipping_mask_name: str,
    label_image_cycle: Optional[int] = None,
    clipping_mask_cycle: Optional[int] = None,
    output_label_cycle: Optional[int] = None,
    output_label_name: str,
    level: int = 0,
    overwrite: bool = True,
) -> None:
    """
    Clips a label image with a mask.

    Takes two label images (or a label image and a binary mask) and replaces
    all values in the first label image with 0 where the second label image has
    values > 0.

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
        label_image_name: Name of the label image to be clipped.
            Needs to exist in OME-Zarr file.
        clipping_mask_name: Name of the label image used as mask for clipping. This
            image will be binarized. Needs to exist in OME-Zarr file.
        label_image_cycle: indicates which cycle contains the label image (only needed if multiplexed).
        clipping_mask_cycle: indicates which cycle contains the clipping mask image (only needed if multiplexed).
        output_label_cycle:  indicates in which cycle to store the result (only needed if multiplexed).
        output_label_name: Name of the output label image.
        level: Resolution of the label image to calculate overlap.
            Only tested for level 0.
        overwrite: If True, overwrite existing label image.
    """

    # update the component for the label image if multiplexed experiment

    in_path = Path(input_paths[0])

    # load images
    label_image = da.from_zarr(
        in_path.joinpath(
            component,
            str(label_image_cycle),
            'labels',
            label_image_name,
            str(level)
        )
    )

    clipping_mask = da.from_zarr(
        in_path.joinpath(
            component,
            str(clipping_mask_cycle),
            'labels',
            clipping_mask_name,
            str(level)
        )
    )

    # prepare label image
    ngff_image_meta = load_NgffImageMeta(
        in_path.joinpath( component, str(label_image_cycle)))
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

    image_group = zarr.group(in_path.joinpath(component,
                                              str(output_label_cycle)))
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
    out = in_path.joinpath(component,
                                 str(output_label_cycle),
                                 "labels",
                                 output_label_name, '0')
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

    # clip the label image
    new_label_image = np.where(clipping_mask > 0, 0, label_image)

    # Compute and store 0-th level to disk
    da.array(new_label_image).to_zarr(
        url=mask_zarr,
        compute=True,
    )

    logger.info(
        f"Clipping done for {out}."
        "now building pyramids."
    )

    # Starting from on-disk highest-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=out.parent,
        overwrite=overwrite,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        aggregation_function=np.max,
    )



if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=clip_label_image,
        logger_name=logger.name,
    )

