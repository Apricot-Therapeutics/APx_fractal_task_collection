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

from skimage.morphology import label
from pydantic import validate_call

from apx_fractal_task_collection.io_models import InitArgsConvertChannelToLabel

import fractal_tasks_core
from fractal_tasks_core.utils import rescale_datasets
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.channels import get_channel_from_image_zarr


logger = logging.getLogger(__name__)


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


@validate_call
def convert_channel_to_label(
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsConvertChannelToLabel,
    # Task-specific arguments
    output_label_name: str,
    overwrite: bool = False,
) -> None:

    """
    Convert a channel of an OME-Zarr image to a label image.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
        `init_convert_channel_to_label`.
        output_label_name: Name of the label to be created.
        overwrite: If True, overwrite existing label image with same name.
    """

    ngff_image_meta = load_NgffImageMeta(zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy

    tmp_channel: OmeroChannel = get_channel_from_image_zarr(
        image_zarr_path=init_args.channel_zarr_url,
        wavelength_id=None,
        label=init_args.channel_label
    )

    ind_channel = tmp_channel.index
    img = \
        da.from_zarr(f"{init_args.channel_zarr_url}/0")[ind_channel]

    # relabel in case the segmentation was created by FOV
    relabeled_img = label(img)

    logger.info(f"Converting channel '{init_args.channel_label}' to "
                f"label image '{output_label_name}'.")

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

    image_group = zarr.group(zarr_url)
    label_group = prepare_label_group(
        image_group,
        output_label_name,
        overwrite=overwrite,
        label_attrs=label_attrs,
        logger=logger,
    )

    out = f"{zarr_url}/labels/{output_label_name}/0"
    store = zarr.storage.FSStore(out)
    label_dtype = np.uint32

    label_zarr = zarr.create(
        shape=img.shape,
        chunks=img.chunksize,
        dtype=label_dtype,
        store=store,
        overwrite=overwrite,
        dimension_separator="/",
    )

    # Write to disk
    da.array(relabeled_img).to_zarr(
        url=label_zarr,
        compute=overwrite,
        overwrite=overwrite,
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
        task_function=convert_channel_to_label,
        logger_name=logger.name,
    )
