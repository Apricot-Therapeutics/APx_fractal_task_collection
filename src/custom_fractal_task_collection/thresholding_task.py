"""
This is the Python module for my_task
"""

import logging
from typing import Any
from pathlib import Path

import zarr
import dask.array as da

from pydantic.decorator import validate_arguments

from fractal_tasks_core.lib_ngff import load_NgffImageMeta
from fractal_tasks_core.lib_pyramid_creation import build_pyramid

@validate_arguments
def thresholding_task(
    *,
    input_paths: list[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
) -> None:
    """
    Short description of thresholding_task.

    Long description of thresholding_task.

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
    """

    # Use the first of input_paths
    input_path = (Path(input_paths[0]) / component).as_posix()
    logging.info(f"input_path set to {input_path}")

    # Parse and log several NGFF-image metadata attributes
    ngff_image_meta = load_NgffImageMeta(input_path)
    logging.info(f"  Axes: {ngff_image_meta.axes_names}")
    logging.info(f"  Number of pyramid levels: {ngff_image_meta.num_levels}")
    logging.info(f"  Linear coarsening factor for YX axes: {ngff_image_meta.coarsening_xy}")
    logging.info(f"  Full-resolution ZYX pixel sizes (micrometer):    {ngff_image_meta.get_pixel_sizes_zyx(level=0)}")
    logging.info(f"  Coarsening-level-1 ZYX pixel sizes (micrometer): {ngff_image_meta.get_pixel_sizes_zyx(level=1)}")

    # Load the highest-resolution multiscale array through dask.array
    array_czyx = da.from_zarr(f"{input_path}/0")
    logging.info(f"{array_czyx=}")

    # Set values below 100 to 0
    array_max = array_czyx.max().compute()
    array_min = array_czyx.min().compute()
    logging.info(f"Pre thresholding:  {array_min=}, {array_max=}")
    array_czyx[array_czyx < 99] = 99
    array_czyx[array_czyx > 1000] = 1000
    array_max = array_czyx.max().compute()
    array_min = array_czyx.min().compute()
    logging.info(f"Post thresholding: {array_min=}, {array_max=}")

    # Write the processed array back to the same full-resolution Zarr array
    array_czyx.to_zarr(f"{input_path}/0", overwrite=True)

    # Starting from on-disk full-resolution data, build and write to disk a
    # pyramid of coarser levels
    build_pyramid(
        zarrurl=input_path,
        overwrite=True,
        num_levels=ngff_image_meta.num_levels,
        coarsening_xy=ngff_image_meta.coarsening_xy,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(task_function=thresholding_task)
