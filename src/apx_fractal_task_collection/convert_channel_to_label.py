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


from typing import Any
from typing import Sequence
from typing import Optional
from pathlib import Path
from skimage.morphology import label
from pydantic.decorator import validate_arguments

import fractal_tasks_core
from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.utils import rescale_datasets
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.channels import get_channel_from_image_zarr


logger = logging.getLogger(__name__)


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

def get_channel_image_from_zarr(zarrurl, channel_label):
    '''
    Get the image data for a specific channel from an OME-Zarr file.

    Args:
        zarrurl: Path to the OME-Zarr file.
        channel_label: Label of the channel to extract.

    Returns:
        The image data for the specified channel as dask array
    '''


    well_group = zarr.open(zarrurl, mode='r')
    for image in well_group.attrs['well']['images']:
        img_zarr_path = zarrurl.joinpath(zarrurl, image['path'])
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

            return data_zyx


@validate_arguments
def convert_channel_to_label(
    *,
    # Standard arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    # Task-specific arguments
    channel_label: str,
    output_label_name: str,
    output_cycle: Optional[int] = None,
    overwrite: bool = False,
) -> None:

    """
    Convert a channel of an OME-Zarr image to a label image.

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
        channel_label: Label of the channel to convert to a label image.
        output_label_name: Name of the label to be created.
        output_cycle: Cycle in which to store the new label image. If not
            provided, the label image is saved in the first cycle.
        overwrite: If True, overwrite existing label image with same name.
    """
    input_path = Path(input_paths[0])
    well_url = input_path.joinpath(component)

    if output_cycle is None:
        output_cycle = 0

    ngff_image_meta = load_NgffImageMeta(well_url.joinpath(str(output_cycle)))
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy

    img = get_channel_image_from_zarr(well_url, channel_label)

    # relabel in case the segmentation was created by FOV
    relabeled_img = label(img)

    label_url = well_url.joinpath(str(output_cycle),
                                  'labels',
                                  output_label_name)

    logger.info(f"Converting channel '{channel_label}' to "
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

    image_group = zarr.group(well_url.joinpath(str(output_cycle)))
    label_group = prepare_label_group(
        image_group,
        output_label_name,
        overwrite=overwrite,
        label_attrs=label_attrs,
        logger=logger,
    )

    store = zarr.storage.FSStore(label_url.joinpath('0').as_posix())
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
        zarrurl=label_url,
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
