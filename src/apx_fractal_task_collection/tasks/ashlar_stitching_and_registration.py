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

import dask.array as da
import numpy as np
from skimage import io
from pydantic import validate_call
from ashlar.scripts import ashlar
from typing import Optional


from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.channels import get_channel_from_image_zarr, get_omero_channel_list, OmeroChannel
from fractal_tasks_core.tasks._zarr_utils import _copy_hcs_ome_zarr_metadata
from fractal_tasks_core.tasks._zarr_utils import _copy_tables_from_zarr_url

from apx_fractal_task_collection.io_models import InitArgsAshlarStitchingAndRegistration

logger = logging.getLogger(__name__)

@validate_call
def ashlar_stitching_and_registration(
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsAshlarStitchingAndRegistration,
    # Task-specific arguments
    overlap: float = 0.1,
    filter_sigma: float = 10,
    ref_channel_id: str,
    ref_cycle: int = 0,
    overwrite_input: bool = False,
    suffix: str = "_stitched",
    tmp_dir: Optional[str] = None,
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
        overwrite_input: If `True`, the results of this task will overwrite the
            input image data. If false, a new image is generated and the
            stitched data is saved there.
        suffix: What suffix to append to the stitched images.
            Only relevant if `overwrite_input=False`.
    """

    # get pixel size
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    pixel_size_yx = ngff_image_meta.get_pixel_sizes_zyx(level=0)[-1]
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy

    sample_data = da.from_zarr(f"{zarr_url}/0")

    # get unique wavelendth ids
    unique_wavelength_ids = []
    for zarr_url in init_args.zarr_urls:
        wavelength_ids = get_omero_channel_list(image_zarr_path=zarr_url)
        unique_wavelength_ids.extend([w.wavelength_id for w in wavelength_ids])

    unique_wavelength_ids = list(set(unique_wavelength_ids))

    # get the shape of each chunk
    height = sample_data.blocks.shape[-2]
    width = sample_data.blocks.shape[-1]

    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpdirname:

        logger.info(f'created temporary directory {tmpdirname}')
        tmpdir = Path(tmpdirname)

        ashlar_paths = {}

        for zarr_url in init_args.zarr_urls:

            i_cycle = zarr_url.split("/")[-1]

            # get wavelength ids for the cycle
            wavelength_ids = get_omero_channel_list(image_zarr_path=zarr_url)
            wavelength_ids = [w.wavelength_id for w in wavelength_ids]

            # make a folder for the cycle
            cycle_tmpdir = tmpdir.joinpath(f'cycle_{i_cycle}')
            cycle_tmpdir.mkdir()

            # add ashlar command to ashlar_paths
            ashlar_paths[i_cycle] = f"fileseries|{tmpdir.joinpath(f'cycle_{i_cycle}')}|pattern=chunk_.tif|" \
                                    f"overlap={overlap}|width={width}|" \
                                    f"height={height}|pixel_size={pixel_size_yx}"
            ashlar_paths[i_cycle] = ashlar_paths[i_cycle].replace("chunk_",
                                                                  "chunk_F{series:3}_C{channel:2}")

            for channel in wavelength_ids:
                logger.info(
                    f"Saving FOVs of channel {channel} from cycle {i_cycle} to "
                    f"temporary directory")

                # load intensity image
                tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                    image_zarr_path=zarr_url,
                    wavelength_id=channel,
                    label=None,
                )
                ind_channel = tmp_channel.index

                print(f'loading data from {zarr_url}/0')
                data_zyx = da.from_zarr(
                    f"{zarr_url}/0")[ind_channel]

                # get the index of the channel that will be consistent
                # across all cycles
                i_c = unique_wavelength_ids.index(channel)

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

        ashlar_args = \
            [f"--output={tmpdir.joinpath('ashlar_output_{cycle}_{channel}.tif')}",
             f"--filter-sigma={filter_sigma}",
             f"--align-channel={unique_wavelength_ids.index(ref_channel_id)}"]

        logger.info(f"Running ASHLAR with paths: {ashlar_paths}")

        ashlar_input = ["test"]
        ashlar_input.extend([value for key, value in ashlar_paths.items() if key == str(ref_cycle)])
        ashlar_input.extend([value for key, value in ashlar_paths.items() if key != str(ref_cycle)])

        ashlar.main(ashlar_input + ashlar_args)

        image_list_updates = dict(image_list_updates=[])

        # loading stitched images and saving them back to zarr
        for zarr_url in init_args.zarr_urls:
            data_czyx_out = []
            # Define old/new zarrurls
            if overwrite_input:
                zarr_url_new = zarr_url.rstrip("/")
            else:
                zarr_url_new = zarr_url.rstrip("/") + suffix

            i_cycle = zarr_url.split("/")[-1]

            data_czyx = da.from_zarr(f"{zarr_url}/0")

            # get wavelength ids for the cycle
            wavelength_ids = get_omero_channel_list(image_zarr_path=zarr_url)
            wavelength_ids = [w.wavelength_id for w in wavelength_ids]

            '''
            Weird behaviour of Ashlar: even when channels are saved 
            with their correct index (e.g. C00, C01, C03), the stitched
            output will have channel names increasing from 0 (C00, C01, C02).
            '''

            actual_ashlar_ids = [unique_wavelength_ids.index(w) for w in wavelength_ids]
            # remove gaps in sequence (e.g. [0, 3, 1] should become [0, 2, 1]
            # to match the stitched image output)
            new_ids = np.arange(len(actual_ashlar_ids))
            # sort new_ids the same way as acutal_ashlar_ids
            new_ids = list(new_ids[np.argsort(actual_ashlar_ids)])

            for i, channel in enumerate(wavelength_ids):
                logger.info(f"Reading stitched FOV for cycle {i_cycle} "
                            f"and channel {channel}")

                #i_c = unique_wavelength_ids.index(channel)
                i_c = new_ids[i]

                stitched_img = io.imread(
                    tmpdir.joinpath(f'ashlar_output_{i_cycle}_{i_c}.tif'))
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
                image_list_updates['image_list_updates'].append(dict(zarr_url=zarr_url))
            else:
                image_list_updates['image_list_updates'].append(
                    dict(zarr_url=zarr_url_new, origin=zarr_url))

    return image_list_updates


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=ashlar_stitching_and_registration,
        logger_name=logger.name,
    )