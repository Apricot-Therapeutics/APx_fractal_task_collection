# Original authors:
# Adrian Tschan <atschan@apricotx.com>
#
# This file is part of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""
Initializes the parallelization list for the clip label image task.
"""
import logging
from typing import Any
from apx_fractal_task_collection.init_utils import (group_by_well,
                                                    get_label_zarr_url)
from pydantic import validate_call
from pathlib import Path

logger = logging.getLogger(__name__)


@validate_call
def init_clip_label_image(
        *,
        # Fractal parameters
        zarr_urls: list[str],
        zarr_dir: str,
        # Core parameters
        label_name: str,
        clipping_mask_name: str,
        output_label_image_name: str = "0",
) -> dict[str, list[dict[str, Any]]]:
    """
    Initialized clip label image task

    This task prepares a parallelization list of all zarr_urls that need to be
    used to perform clipping based on two label images.

    Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created. Not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Name of the label image to be clipped.
            Needs to exist in OME-Zarr file.
        clipping_mask_name: Name of the label image used as mask for clipping.
            This image will be binarized. Needs to exist in OME-Zarr file.
        output_label_image_name: Name of the zarr image that will
            contain the clipped label image. Defaults to "0".
            In case you saved, for example, illumination corrected images in
            the same zarr without overwriting the original images, you can
            specify a different name here (e.g. "0_illum_corrected").

    Returns:
        task_output: Dictionary for Fractal server that contains a
            parallelization list.
    """
    logger.info(
        f"Running `init_clip_label_image` for {zarr_urls=}"
    )
    well_dict = group_by_well(zarr_urls)
    # Create the parallelization list
    parallelization_list = []
    for well, well_list in well_dict.items():
        label_zarr_url = get_label_zarr_url(well_list,
                                            label_name)
        clipping_mask_zarr_url = get_label_zarr_url(well_list,
                                                    clipping_mask_name)

        # generate zarr_url by taking the first entry of well_list and
        # replacing the last part of the path with the output_label_image_name
        zarr_url = (f"{well_list[0].rsplit('/', 1)[0]}/"
                    f"{output_label_image_name}")

        parallelization_list.append(
            dict(
                zarr_url=zarr_url,
                init_args=dict(
                    label_name=label_name,
                    label_zarr_url=label_zarr_url,
                    clipping_mask_name=clipping_mask_name,
                    clipping_mask_zarr_url=clipping_mask_zarr_url,
                ),
            )
        )
    return dict(parallelization_list=parallelization_list)


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=init_clip_label_image,
        logger_name=logger.name,
    )