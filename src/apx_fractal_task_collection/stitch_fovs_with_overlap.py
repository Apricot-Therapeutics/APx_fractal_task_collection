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
import itertools
import time
import warnings
import subprocess
from pathlib import Path
import tempfile
from typing import Any
from typing import Optional
from typing import Sequence

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from basicpy import BaSiC
from skimage import io
from pydantic.decorator import validate_arguments

from fractal_tasks_core.lib_channels import get_omero_channel_list
from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_ngff import load_NgffImageMeta
from fractal_tasks_core.lib_pyramid_creation import build_pyramid
from fractal_tasks_core.lib_regions_of_interest import check_valid_ROI_indices
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)

logger = logging.getLogger(__name__)

@validate_arguments
def stitch_fovs_with_overlap(
    *,
    # Standard arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    # Task-specific arguments
    overlap: float = 0.1,
    filter_sigma: float = 10,
    safety_pad: int = 250,
) -> None:

    """
    Stitches FOVs with overlap using ASHLAR (https://github.com/labsyspharm/ashlar).

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
        component: Path to the OME-Zarr image in the OME-Zarr plate that is
            processed. Example: `"some_plate.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This parameter is not used by this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        overlap: The overlap between FOVs in percent (0-1).
        filter_sigma: The sigma of the Gaussian filter used to filter the
            FOVs for stitching. Can help to improve the stitching performance.
        safety_pad: The amount of padding in pixels to add to the stitched
            image additionally to the padding to match the original image size.
            (image can be slightly larger in each dimension after stitching)
    """
    in_path = Path(input_paths[0])
    zarrurl = in_path.joinpath(component)

    # get pixel size
    ngff_image_meta = load_NgffImageMeta(zarrurl)
    pixel_size_yx = ngff_image_meta.get_pixel_sizes_zyx(level=0)[-1]
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy

    data_czyx = da.from_zarr(zarrurl.joinpath('0'))

    height = data_czyx.blocks.shape[-2]
    width = data_czyx.blocks.shape[-1]

    data_czyx_out = []

    for data_zyx in data_czyx:
        with tempfile.TemporaryDirectory() as tmpdirname:

            logger.info(f'created temporary directory {tmpdirname}')
            tmpdir = Path(tmpdirname)

            logger.info("Saving FOVs to temporary directory")
            for i, inds in enumerate(
                    itertools.product(*map(range, data_zyx.blocks.shape))):
                chunk = data_zyx.blocks[inds]
                io.imsave(
                    tmpdir.joinpath(f'chunk_{i:03d}.tif'),
                    np.squeeze(chunk.compute()))
                logger.info(f"Saved chunk {i} of shape "
                            f"{np.squeeze(chunk).shape} to "
                            f"{tmpdir.joinpath(f'chunk_{i:03d}.tif')}")

            logger.info("Running ASHLAR to stitch FOVs")
            ashlar_path = f"fileseries|{tmpdir}|pattern=chunk_.tif|" \
                          f"overlap={overlap}|width={width}|" \
                          f"height={height}|pixel_size={pixel_size_yx}"
            ashlar_path = ashlar_path.replace("chunk_", "chunk_{series:3}")
            ashlar_args = f"--output={tmpdir.joinpath('ashlar_output.tif')} " \
                          f"--filter-sigma={filter_sigma}"

            logger.info(f"Running ASHLAR with path: {ashlar_path}")
            subprocess.call(f". /etc/profile.d/lmod.sh;"
                            f" module load openjdk/11.0.2/gcc;"
                            f" ashlar '{ashlar_path}' {ashlar_args}",
                            shell=True)

            logger.info("Reading stitched FOV")
            stitched_img = io.imread(tmpdir.joinpath('ashlar_output.tif'))
            logger.info(f"Stitched image has shape {stitched_img.shape}")

            logger.info("Padding stitched image")

            to_pad_x = data_zyx.shape[2] - stitched_img.shape[1] + safety_pad
            to_pad_y = data_zyx.shape[1] - stitched_img.shape[0] + safety_pad
            stitched_img = np.pad(stitched_img,
                                  ((0, to_pad_y), (0, to_pad_x)),
                                   mode='constant',
                                  constant_values=0)

            stitched_img = np.expand_dims(stitched_img, axis=0)

            logger.info(f"Shape of the stitched image after padding: "
                        f"{stitched_img.shape}")

            data_czyx_out.append(stitched_img)

    logger.info("Saving stitched image to disk")
    data_czyx_out = np.stack(data_czyx_out)
    # Write to disk
    da.array(data_czyx_out).to_zarr(
        url=zarrurl.joinpath('0'),
        compute=True,
        overwrite=True,
        chunksize=data_czyx.chunksize,
    )

    logger.info("build pyramids")

    build_pyramid(
        zarrurl=zarrurl,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
    )

    logger.info("Updating well_ROI_table")
    # Update the well_ROI_table
    well_ROI_table = ad.read_zarr(zarrurl.joinpath('tables/well_ROI_table'))
    data = well_ROI_table.to_df()
    new_len_x = data_czyx_out.shape[-1] * pixel_size_yx
    new_len_y = data_czyx_out.shape[-2] * pixel_size_yx

    logger.info(f"Updated len_x_micrometer "
                f"({int(data['len_x_micrometer'].values[0])} ->"
                f" {int(new_len_x)}) "
                f"and len_y_micrometer "
                f"({int(data['len_y_micrometer'].values[0])} ->"
                f" {int(new_len_y)})")

    data['len_x_micrometer'] = new_len_x
    data['len_y_micrometer'] = new_len_y

    well_ROI_table.X = data
    well_ROI_table.write_zarr(zarrurl.joinpath('tables/well_ROI_table'))


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=stitch_fovs_with_overlap,
        logger_name=logger.name,
    )
