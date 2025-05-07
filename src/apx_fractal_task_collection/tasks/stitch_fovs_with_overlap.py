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
from pathlib import Path
import tempfile
import zarr

from typing import Optional

import dask.array as da
import numpy as np
from skimage import io
from pydantic import validate_call
from ashlar.scripts import ashlar


from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.tasks._zarr_utils import _copy_hcs_ome_zarr_metadata
from fractal_tasks_core.tasks._zarr_utils import _copy_tables_from_zarr_url

logger = logging.getLogger(__name__)

@validate_call
def stitch_fovs_with_overlap(
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    # Task-specific arguments
    overlap: float = 0.1,
    filter_sigma: float = 10,
    overwrite_input: bool = True,
    suffix: str = "_stitched",
    tmp_dir: Optional[str] = None,
) -> None:

    """
    Stitches FOVs with overlap using ASHLAR (https://github.com/labsyspharm/ashlar).

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        overlap: The overlap between FOVs in percent (0-1).
        filter_sigma: The sigma of the Gaussian filter used to filter the
            FOVs for stitching. Can help to improve the stitching performance.
        overwrite_input:
            If `True`, the results of this task will overwrite the input image
            data. If false, a new image is generated and the stitched
             data is saved there.
        suffix: What suffix to append to the stitched images.
            Only relevant if `overwrite_input=False`.
        tmp_dir: Optional path to where the temporary directory should be
            created. Might be necessary if the default temporary directory
            (/tmp/) on the cluster has too little space left.
    """

    # Define old/new zarrurls
    if overwrite_input:
        zarr_url_new = zarr_url.rstrip("/")
    else:
        zarr_url_new = zarr_url.rstrip("/") + suffix

    # get pixel size
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    pixel_size_yx = ngff_image_meta.get_pixel_sizes_zyx(level=0)[-1]
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy

    data_czyx = da.from_zarr(f"{zarr_url}/0")

    height = data_czyx.blocks.shape[-2]
    width = data_czyx.blocks.shape[-1]

    data_czyx_out = []

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpdirname:

        logger.info(f'created temporary directory {tmpdirname}')
        tmpdir = Path(tmpdirname)

        for i_c, data_zyx in enumerate(data_czyx):
            logger.info(f"Saving FOVs of channel index {i_c} to "
                        f"temporary directory")
            for i, inds in enumerate(
                    itertools.product(*map(range, data_zyx.blocks.shape))):
                chunk = data_zyx.blocks[inds]
                io.imsave(
                    tmpdir.joinpath(f'chunk_F{i:03d}_C{i_c:02d}.tif'),
                    np.squeeze(chunk.compute()))
                logger.info(
                    f"Saved chunk {i} of shape "
                    f"{np.squeeze(chunk).shape} to "
                    f"{tmpdir.joinpath(f'chunk_F{i:03d}_C{i_c:02d}.tif')}")

        logger.info("Running ASHLAR to stitch FOVs")

        # layout: The layout of the FOVs. Can be "raster" or "snake".
         #     See https://forum.image.sc/t/ashlar-how-to-pass-multiple-images-to-be-stitched/49864/24 for more information.
         #direction: The direction of the stitching. Can be "vertical" or
         #    "horizontal". See https://forum.image.sc/t/ashlar-how-to-pass-multiple-images-to-be-stitched/49864/24 for more information.

        ashlar_path = f"fileseries|{tmpdir}|pattern=chunk_.tif|" \
                      f"overlap={overlap}|width={width}|" \
                      f"height={height}|pixel_size={pixel_size_yx}"
        ashlar_path = ashlar_path.replace("chunk_",
                                          "chunk_F{series:3}_C{channel:2}")
        #ashlar_args = \
        #    f"--output=" \
        #    f"{tmpdir.joinpath('ashlar_output_{cycle}_{channel}.tif')} " \
        #    f"--filter-sigma={filter_sigma}"
        ashlar_args = \
            [f"--output={tmpdir.joinpath('ashlar_output_{cycle}_{channel}.tif')}",
             f"--filter-sigma={filter_sigma}"]

        logger.info(f"Running ASHLAR with path: {ashlar_path}")
        #subprocess.call(f". /etc/profile.d/lmod.sh;"
        #                f" module load openjdk/11.0.2/gcc;"
        #                f" ashlar '{ashlar_path}' {ashlar_args}",
        #                shell=True)
        ashlar.main(["test", ashlar_path] + ashlar_args)
        for i_c, data_zyx in enumerate(data_czyx):
            logger.info("Reading stitched FOV")
            stitched_img = io.imread(
                tmpdir.joinpath(f'ashlar_output_0_{i_c}.tif'))
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

    if overwrite_input:
        new_zarr = zarr.open(f"{zarr_url_new}/0")
    else:
        new_zarr = zarr.create(
            shape=data_czyx.shape,
            chunks=data_czyx.chunksize,
            dtype=data_czyx.dtype,
            store=zarr.storage.FSStore(f"{zarr_url_new}/0"),
            overwrite=False,
            dimension_separator="/",
        )
        _copy_hcs_ome_zarr_metadata(zarr_url, zarr_url_new)
        # Copy ROI tables from the old zarr_url to keep ROI tables and other
        # tables available in the new Zarr
        _copy_tables_from_zarr_url(zarr_url, zarr_url_new)

    # Write to disk
    data_czyx_out.to_zarr(
        url=new_zarr,
        compute=True,
        overwrite=True,
        dimension_separator="/"
    )

    logger.info("build pyramids")

    build_pyramid(
        zarrurl=zarr_url_new,
        overwrite=True,
        num_levels=num_levels,
        coarsening_xy=coarsening_xy,
        chunksize=data_czyx.chunksize,
    )

    if overwrite_input:
        image_list_updates = dict(image_list_updates=[dict(zarr_url=zarr_url)])
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=zarr_url_new, origin=zarr_url)]
        )
    return image_list_updates


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=stitch_fovs_with_overlap,
        logger_name=logger.name,
    )