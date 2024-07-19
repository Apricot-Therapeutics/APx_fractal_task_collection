# Original authors:
# Adrian Tschan <adrian.tschan@uzh.ch>
#
# This file is part of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.

import zarr
import logging
import fractal_tasks_core
import shutil

import numpy as np
import dask.array as da


from pathlib import Path
from typing import Any, Dict, Sequence
from typing import Callable
from typing import Optional
from typing import Union
import numcodecs

from fractal_tasks_core.ngff import load_NgffImageMeta
from pydantic import validate_call


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


def rescale_values(data, q=0.9999):
    data[data > np.quantile(data, q)] = np.quantile(data, q)
    data = (data - data.min()) / (np.quantile(data, q) - data.min())
    data = data * (2 ** 8 - 1)
    return data.astype('uint8')


def build_pyramid(
    *,
    zarrurl: Union[str, Path],
    overwrite: bool = False,
    num_levels: int = 2,
    coarsening_xy: int = 2,
    chunksize: Optional[Sequence[int]] = None,
    aggregation_function: Optional[Callable] = None,
    codec: str,
) -> None:

    """
    Starting from on-disk highest-resolution data, build and write to disk a
    pyramid with `(num_levels - 1)` coarsened levels.
    This function works for 2D, 3D or 4D arrays.

    Args:
        zarrurl: Path of the image zarr group, not including the
            multiscale-level path (e.g. `"some/path/plate.zarr/B/03/0"`).
        overwrite: Whether to overwrite existing pyramid levels.
        num_levels: Total number of pyramid levels (including 0).
        coarsening_xy: Linear coarsening factor between subsequent levels.
        chunksize: Shape of a single chunk.
        aggregation_function: Function to be used when downsampling.
    """

    # Clean up zarrurl
    zarrurl = str(Path(zarrurl))  # FIXME

    # Select full-resolution multiscale level
    zarrurl_highres = f"{zarrurl}/0"
    logger.info(f"[build_pyramid] High-resolution path: {zarrurl_highres}")

    # Lazily load highest-resolution data
    data_highres = da.from_zarr(zarrurl_highres)
    logger.info(f"[build_pyramid] High-resolution data: {str(data_highres)}")

    # Check the number of axes and identify YX dimensions
    ndims = len(data_highres.shape)
    if ndims not in [2, 3, 4]:
        raise ValueError(f"{data_highres.shape=}, ndims not in [2,3,4]")
    y_axis = ndims - 2
    x_axis = ndims - 1

    # Set aggregation_function
    if aggregation_function is None:
        aggregation_function = np.mean

    # Compute and write lower-resolution levels
    previous_level = data_highres
    for ind_level in range(1, num_levels):
        # Verify that coarsening is doable
        if min(previous_level.shape[-2:]) < coarsening_xy:
            raise ValueError(
                f"ERROR: at {ind_level}-th level, "
                f"coarsening_xy={coarsening_xy} "
                f"but previous level has shape {previous_level.shape}"
            )
        # Apply coarsening
        newlevel = da.coarsen(
            aggregation_function,
            previous_level,
            {y_axis: coarsening_xy, x_axis: coarsening_xy},
            trim_excess=True,
        ).astype(data_highres.dtype)

        # Apply rechunking
        if chunksize is None:
            newlevel_rechunked = newlevel
        else:
            if newlevel.shape[-1] > 2000:
                new_chunksize = [x/2 if i in [2, 3] else
                                1 for i, x in enumerate(newlevel.shape)]
                newlevel_rechunked = newlevel.rechunk(new_chunksize)
            else:
                new_chunksize = [x*4 if i in [2, 3] else
                                1 for i, x in enumerate(newlevel.chunksize)]
                newlevel_rechunked = newlevel.rechunk(new_chunksize)
        logger.info(
            f"[build_pyramid] Level {ind_level} data: "
            f"{str(newlevel_rechunked)}"
        )

        # Write zarr and store output (useful to construct next level)
        previous_level = newlevel_rechunked.to_zarr(
            zarrurl,
            component=f"{ind_level}",
            overwrite=overwrite,
            compute=True,
            return_stored=True,
            write_empty_chunks=False,
            dimension_separator="/",
            compressor=codec,
        )

@validate_call
def compress_zarr_for_visualization(  # noqa: C901
        *,
        # Default arguments for fractal tasks:
        input_paths: Sequence[str],
        output_path: str,
        component: str,
        metadata: Dict[str, Any],
        # Task-specific arguments:
        output_zarr_path: str,
        overwrite: bool = False
) -> None:
    """
    Convert all images in a zarr file to a lower bit depth and compress them
    to allow for smoother visualization in napari.

    Args:
        Args:
        input_paths: Path to the parent folder of the NGFF image.
            This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: This argument is not used in this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: This argument is not used in this task.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_zarr_path: Path to the output zarr file. Should have been
            previously created by the "Copy OME_Zarr" Fractal task.
        overwrite: Whether to overwrite existing pyramid levels.
    """

    zarr_path = Path(input_paths[0]).joinpath(
        component.replace(f"_{metadata['copy_ome_zarr']['suffix']}", ""))
    output_zarr_path = Path(output_zarr_path)
    well_path = f"{zarr_path.parent.name}/{zarr_path.name}"
    #compressor = numcodecs.Blosc(
    # cname='zstd', clevel=9, shuffle=Blosc.AUTOSHUFFLE)
    compressor = numcodecs.bz2.BZ2(level=9)

    # Load the original zarr file
    zarr_group = zarr.open(zarr_path, mode="r")

    # get all img_groups
    img_group_keys = [group for group in zarr_group.keys() if
                      "multiscales" in zarr_group[group].attrs]

    ## prepare meta data
    ngff_image_meta = load_NgffImageMeta(zarr_path.joinpath(img_group_keys[0]))
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    #num_levels = 2
    #coarsening_xy = 4


    # for each img_group, load the first pyramid level and compress it
    for img_group_key in img_group_keys:

        out_path = output_zarr_path.joinpath(well_path, img_group_key, '0')
        # use level 1 of the pyramid as highest resolution image to reduce size
        data = da.from_zarr(zarr_path.joinpath(img_group_key, '1'))
        # rescale the data to 8 bit range
        rescale_chunks = [x if i in [2, 3] else
                         1 for i, x in enumerate(data.shape)]
        original_chunks = data.chunksize
        data = data.rechunk(rescale_chunks)
        data = data.map_blocks(lambda x: rescale_values(x, q=0.9999),
                               dtype=np.uint8)
        data = data.rechunk(original_chunks)

        out_zarr = zarr.create(
            shape=data.shape,
            chunks=data.chunksize,
            dtype=np.uint8,
            store=out_path,
            overwrite=overwrite,
            dimension_separator=".",
            compressor=compressor
        )

        data.to_zarr(out_zarr, overwrite=True, compressor=compressor,
                     compute=True)

        # build pyramid
        build_pyramid(
            zarrurl=out_path.parent,
            overwrite=overwrite,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
            codec=compressor,
            chunksize=[1],
        )

        # update zattrs
        # copy zattrs from original plate
        shutil.copy(zarr_path.joinpath(img_group_key, '.zattrs'),
                    out_path.parent.joinpath('.zattrs'))

        out_group = zarr.open(
            output_zarr_path.joinpath(well_path, img_group_key), mode="a")

        new_multiscales = out_group.attrs['multiscales']
        new_datasets = new_multiscales[0]['datasets']
        n_orig_levels = len(new_datasets)
        new_datasets.pop(0)
        new_datasets = [
            {**d, 'path': str(int(d['path'])-1)} for d in new_datasets]
        old_scale = new_datasets[-1]['coordinateTransformations'][0]['scale']
        old_path = new_datasets[-1]['path']
        new_datasets.append({
            'coordinateTransformations': [{
                'scale':[x*coarsening_xy if i in [2, 3] else
                         x for i, x in enumerate(old_scale)],
                'type': 'scale'}],
            'path': str(int(old_path) + 1)})
        new_datasets = new_datasets[0:num_levels]
        new_multiscales[0]['datasets'] = new_datasets
        out_group.attrs.__setitem__('multiscales', new_multiscales)

        # copy the labels folder if it's present and not copied yet
        if 'labels' in zarr_group[img_group_key]:
            if out_path.parent.joinpath('labels').exists():
                logger.info("Labels folder already exists in the output zarr.")
                pass
            else:
                shutil.copytree(zarr_path.joinpath(img_group_key, 'labels'),
                                out_path.parent.joinpath('labels'),
                                dirs_exist_ok=True)

        # remove any levels that are not needed
        for i in range(num_levels, n_orig_levels):
                shutil.rmtree(out_path.parent.joinpath(str(i)),
                              ignore_errors=True)




if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=compress_zarr_for_visualization,
        logger_name=logger.name,
    )
