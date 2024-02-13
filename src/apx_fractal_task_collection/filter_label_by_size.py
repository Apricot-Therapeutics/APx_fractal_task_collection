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
from skimage.morphology import remove_small_objects
from pathlib import Path
from pydantic.decorator import validate_arguments

import fractal_tasks_core
from fractal_tasks_core.lib_zattrs_utils import rescale_datasets
from fractal_tasks_core.lib_write import prepare_label_group
from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from fractal_tasks_core.lib_ngff import load_NgffImageMeta


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


def get_label_image_from_zarr(zarrurl, label_name):
    '''
    Get the image data for a specific channel from an OME-Zarr file.

    Args:
        zarrurl: Path to the OME-Zarr file.
        channel_label: Label of the channel to extract.

    Returns:
        The image data for the specified channel as dask array
    '''

    well_group = zarr.open_group(zarrurl, mode="r+")
    for image in well_group.attrs['well']['images']:
        try:
            img_zarr_path = zarrurl.joinpath(zarrurl, image['path'])
            data_zyx = da.from_zarr(
                img_zarr_path.joinpath(f'labels/{label_name}/0'))
            break
        except:
            continue

    return data_zyx, img_zarr_path



@validate_arguments
def filter_label_by_size(
    *,
    # Standard arguments
    input_paths: Sequence[str],
    output_path: str,
    metadata: dict[str, Any],
    component: str,
    # Task-specific arguments
    label_name: str,
    output_label_name: str,
    min_size: Optional[int] = None,
    max_size: Optional[int] = None,
    overwrite: bool = False,
) -> None:

    """
    Filter objects in a label image by size.

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
        label_name: Name of the label image in the OME-Zarr file.
        output_label_name: Name of the output label image. If None, the input
            label image is overwritten.
        min_size: Minimum size of objects to keep. If None, no minimum size
            filter is applied.
        max_size: Maximum size of objects to keep. If None, no maximum size
            filter is applied.
    """
    logger.info(f"Filtering label image '{label_name}' by size.")
    in_path = Path(input_paths[0])
    zarrurl = in_path.joinpath(component)

    data_zyx, img_zarr_path = get_label_image_from_zarr(
        zarrurl, label_name)

    ngff_image_meta = load_NgffImageMeta(img_zarr_path)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy

    label_img = data_zyx.compute()
    if min_size is not None:
        logger.info(f"Removing objects smaller than {min_size} pixels")
        label_img = remove_small_objects(label_img, min_size=min_size)
    if max_size is not None:
        logger.info(f"Removing objects larger than {max_size} pixels")
        label_img = remove_large_objects(label_img, max_size=max_size)

    if output_label_name == label_name or output_label_name is None:

        logger.info(f"Overwriting label image '{label_name}' "
                    f"with size filtered image.")

        out_path = img_zarr_path.joinpath(f'labels/{label_name}/0')

        out_zarr = zarr.create(
            shape=data_zyx.shape,
            chunks=data_zyx.chunksize,
            dtype=data_zyx.dtype,
            store=out_path,
            overwrite=overwrite,
            dimension_separator="/",
        )

        # Write to disk
        da.array(label_img).to_zarr(
            url=out_zarr,
            compute=True,
            overwrite=overwrite
        )

        # Starting from on-disk highest-resolution data, build and write
        # to disk a pyramid of coarser levels

        build_pyramid(
            zarrurl=out_path.parent,
            overwrite=overwrite,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
        )

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

        image_group = zarr.group(img_zarr_path)
        label_group = prepare_label_group(
            image_group,
            output_label_name,
            overwrite=overwrite,
            label_attrs=label_attrs,
            logger=logger,
        )

        label_url = img_zarr_path.joinpath(f'labels/{output_label_name}/0')
        store = zarr.storage.FSStore(img_zarr_path.joinpath(f'labels/{output_label_name}/0').as_posix())
        label_dtype = np.uint32

        label_zarr = zarr.create(
            shape=data_zyx.shape,
            chunks=data_zyx.chunksize,
            dtype=label_dtype,
            store=store,
            overwrite=overwrite,
            dimension_separator="/",
        )

        # Write to disk
        da.array(label_img).to_zarr(
            url=label_zarr,
            compute=True,
            overwrite=overwrite
        )

        # Starting from on-disk highest-resolution data, build and write to disk a
        # pyramid of coarser levels

        build_pyramid(
            zarrurl=label_url.parent,
            overwrite=overwrite,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy
        )

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=filter_label_by_size,
        logger_name=logger.name,
    )

