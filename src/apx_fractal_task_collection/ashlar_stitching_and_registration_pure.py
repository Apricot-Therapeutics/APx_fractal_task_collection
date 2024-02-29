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
import subprocess
from pathlib import Path
import tempfile
from typing import Any
from typing import Sequence

import dask.array as da
import numpy as np
from skimage import io
from pydantic.decorator import validate_arguments
from ashlar.scripts import ashlar


from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid

logger = logging.getLogger(__name__)

@validate_arguments
def ashlar_stitching_and_registration(
    *,
    # Standard arguments
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    # Task-specific arguments
    overlap: float = 0.1,
    filter_sigma: float = 10,
    ref_channel_id: str,
    ref_cycle: int = 0
) -> None:

    """
    Stitches FOVs with overlap using ASHLAR (https://github.com/labsyspharm/ashlar)
    and register the stitched image to the reference cycle.

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
        ref_channel_id: The wavelength id of the channel to use as reference.
        ref_cycle: The cycle to which the stitched image will be registered to.
    """
    in_path = Path(input_paths[0])
    zarrurl = in_path.joinpath(component)
    current_cycle = zarrurl.name

    # get pixel size
    ngff_image_meta = load_NgffImageMeta(zarrurl)
    pixel_size_yx = ngff_image_meta.get_pixel_sizes_zyx(level=0)[-1]
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy

    cycle_data_czyx = []
    # add data from reference cycle
    cycle_data_czyx.append(
        da.from_zarr(zarrurl.parent.joinpath(f'{ref_cycle}/0')))
    # add data from current cycle
    cycle_data_czyx.append(da.from_zarr(zarrurl.joinpath('0')))

    cycle_data_czyx = np.stack(cycle_data_czyx)


    height = cycle_data_czyx.blocks.shape[-2]
    width = cycle_data_czyx.blocks.shape[-1]

    data_czyx_out = []
    with tempfile.TemporaryDirectory() as tmpdirname:

        logger.info(f'created temporary directory {tmpdirname}')
        tmpdir = Path(tmpdirname)

        for i_cycle, data_czyx in enumerate(cycle_data_czyx):
            # make a folder for the cycle
            cycle_tmpdir = tmpdir.joinpath(f'cycle_{i_cycle}')
            cycle_tmpdir.mkdir()
            for i_c, data_zyx in enumerate(data_czyx):
                logger.info(
                    f"Saving FOVs of channel index {i_c} from cycle {i_cycle} to "
                    f"temporary directory")
                for i, inds in enumerate(
                        itertools.product(*map(range, data_zyx.blocks.shape))):
                    chunk = data_zyx.blocks[inds]
                    io.imsave(
                        cycle_tmpdir.joinpath(f'chunk_F{i:03d}_C{i_c:02d}.tif'),
                        np.squeeze(chunk.compute()))
                    logger.info(
                        f"Saved chunk {i} of shape "
                        f"{np.squeeze(chunk).shape} to "
                        f"{cycle_tmpdir.joinpath(f'chunk_F{i:03d}_C{i_c:02d}.tif')}")

        logger.info("Running ASHLAR to stitch FOVs")
        ashlar_path1 = f"fileseries|{tmpdir.joinpath('cycle_0')}|pattern=chunk_.tif|" \
                      f"overlap={overlap}|width={width}|" \
                      f"height={height}|pixel_size={pixel_size_yx}"
        ashlar_path1 = ashlar_path1.replace("chunk_",
                                          "chunk_F{series:3}_C{channel:2}")

        ashlar_path2 = f"fileseries|{tmpdir.joinpath('cycle_1')}|pattern=chunk_.tif|" \
                       f"overlap={overlap}|width={width}|" \
                       f"height={height}|pixel_size={pixel_size_yx}"
        ashlar_path2 = ashlar_path2.replace("chunk_",
                                            "chunk_F{series:3}_C{channel:2}")

        #ashlar_args = \
        #    f"--output=" \
        #    f"{tmpdir.joinpath('ashlar_output_{cycle}_{channel}.tif')} " \
        #    f"--filter-sigma={filter_sigma}"

        ashlar_args = \
            [f"--output={tmpdir.joinpath('ashlar_output_{cycle}_{channel}.tif')}",
             f"--filter-sigma={filter_sigma}",
             "--align-channel=0"]

        logger.info(f"Running ASHLAR with path: {ashlar_path1} and {ashlar_path2}")

        #subprocess.call(f". /etc/profile.d/lmod.sh;"
        #                f" module load openjdk/11.0.2/gcc;"
        #                f" ashlar '{ashlar_path}' {ashlar_args}",
        #                shell=True)

        ashlar.main(["test", ashlar_path1, ashlar_path2] + ashlar_args)
        for i_c, data_zyx in enumerate(data_czyx):
            logger.info("Reading stitched FOV")
            if current_cycle == '0':
                stitched_img = io.imread(
                    tmpdir.joinpath(f'ashlar_output_0_{i_c}.tif'))
            else:
                stitched_img = io.imread(
                    tmpdir.joinpath(f'ashlar_output_1_{i_c}.tif'))
            logger.info(f"Stitched image has shape {stitched_img.shape}")

            logger.info(f"Padding stitched image to match original image size "
                        f"({data_zyx.shape})")

            to_pad_x = data_zyx.shape[2] - stitched_img.shape[1]
            to_pad_y = data_zyx.shape[1] - stitched_img.shape[0]
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
    data_czyx_out = da.from_array(data_czyx_out, chunks=data_czyx.chunksize)

    # Write to disk
    data_czyx_out.to_zarr(
        url=zarrurl.joinpath('0'),
        compute=True,
        overwrite=True,
        dimension_separator="/"
    )

    logger.info("build pyramids")

    build_pyramid(
        zarrurl=zarrurl,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=ashlar_stitching_and_registration,
        logger_name=logger.name,
    )