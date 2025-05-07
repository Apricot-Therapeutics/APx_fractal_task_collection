"""
# Original authors:
# Adrian Tschan <adrian.tschan@uzh.ch>
#
# This file is part of the Apricot Therapeutics Fractal Task Collection, which
# is developed by Apricot Therapeutics AG and intended to be used with the
# Fractal platform originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""

import logging

import dask.array as da
import fractal_tasks_core
import numpy as np
import zarr
import anndata as ad
from pathlib import Path
import pandas as pd

from apx_fractal_task_collection.io_models import InitArgsCalculatePixelIntensityCorrelation
from apx_fractal_task_collection.features.intensity import object_intensity_correlation
from apx_fractal_task_collection.utils import get_acquisition_from_label_name

from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)
from fractal_tasks_core.tables import write_table

from pydantic import validate_call


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

@validate_call
def calculate_pixel_intensity_correlation(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsCalculatePixelIntensityCorrelation,
    # Task-specific arguments:
    ROI_table_name: str,
    output_table_name: str,
    level: int = 0,
    overwrite: bool = True,
) -> None:
    """
    Calculate pixel intensity correlation between two channels.

    Takes a label image and two channel images and calculates the pixel
    intensity correlation between the two channels for each object in the label
    image.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_segment_secondary_objects`.
        ROI_table_name: Name of the table containing the ROIs.
        output_table_name: Name of the output feature table.
        level: Resolution of the label image to calculate correlation.
            Only tested for level 0.
        overwrite: If True, overwrite existing table.
    """
    well_url = Path(zarr_url).parent
    well_name = zarr_url.split("/")[-3] + zarr_url.split("/")[-2]

    # load label image
    label_image = da.from_zarr(
        f"{init_args.label_zarr_url}/labels/{init_args.label_name}/{level}")

    label_image_cycle = get_acquisition_from_label_name(well_url,
                                                        init_args.label_name)

    # prepare label image
    ngff_image_meta = load_NgffImageMeta(init_args.label_zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)

    # load ROI table
    ROI_table = ad.read_zarr(
        f"{init_args.label_zarr_url}/tables/{ROI_table_name}")

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=0,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, ROI_table_name)
    num_ROIs = len(list_indices)

    channel_pair_features = []
    for i, channel_pair in enumerate(init_args.corr_channel_urls):

        channel_labels = init_args.corr_channel_labels[i]

        # load intensity image 1
        tmp_channel: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=list(channel_pair.keys())[0],
            wavelength_id=None,
            label=list(channel_labels.keys())[0],
        )
        ind_channel = tmp_channel.index
        data_zyx_1 = da.from_zarr(
            f"{list(channel_pair.keys())[0]}/{level}")[ind_channel]

        # load intensity image 2
        tmp_channel: OmeroChannel = get_channel_from_image_zarr(
            image_zarr_path=list(channel_pair.values())[0],
            wavelength_id=None,
            label=list(channel_labels.values())[0],
        )
        ind_channel = tmp_channel.index
        data_zyx_2 = da.from_zarr(
            f"{list(channel_pair.values())[0]}/{level}")[ind_channel]

        feature_list=[]
        obs_list = []
        # Loop over the list of indices and perform the secondary segmentation
        for i_ROI, indices in enumerate(list_indices):

            # Define region
            s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
            region = (
                slice(s_z, e_z),
                slice(s_y, e_y),
                slice(s_x, e_x),
            )
            logger.info(
                f"Now processing ROI {i_ROI + 1}/{num_ROIs} from ROI table"
                f" {ROI_table_name}."
            )

            # perform measurements
            correlation = object_intensity_correlation(
                labels=label_image[region].compute(),
                ref_img=data_zyx_1[region].compute(),
                img=data_zyx_2[region].compute(),
            )

            correlation.set_index("label", inplace=True)
            correlation.columns = [init_args.label_name + \
                                  "_Correlation" + \
                                  "_" + list(channel_labels.keys())[0] + \
                                  "_" + list(channel_labels.values())[0]]

            correlation.reset_index(inplace=True)

            ROI_obs = pd.DataFrame(
                {"label": correlation['label'],
                 "well_name": well_name,
                 "ROI": ROI_table.obs.index[i_ROI]})

            feature_list.append(correlation)
            obs_list.append(ROI_obs)

            logger.info(f"Finished correlation calculation for ROI {i_ROI + 1}/{num_ROIs}.")

            if feature_list:
                obs = pd.concat(obs_list, axis=0)
                merged_features = pd.concat(feature_list, axis=0)

                merged_features.set_index('label', inplace=True)
                # obs.set_index('label', inplace=True)
                obs.index = np.arange(0, len(obs))

            else:
                logger.info(
                    f"No features calculated for {init_args.label_name}. Likely,"
                    f" there are no objects in the label image.")

        channel_pair_features.append(merged_features)

    channel_pair_features_merged = pd.concat(channel_pair_features, axis=1)
    # save features as AnnData table
    feature_table = ad.AnnData(
        X=channel_pair_features_merged.reset_index(drop=True),
        obs=obs,
        dtype='float32')

    # Write to zarr group
    image_group = zarr.group(init_args.label_zarr_url)
    write_table(
        image_group,
        output_table_name,
        feature_table,
        overwrite=overwrite,
        table_attrs={"type": "feature_table",
                     "region": {
                         "path": f"../../{label_image_cycle}/"
                                 f"labels/{init_args.label_name}"},
                     "instance_key": "label"}
    )



if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=calculate_pixel_intensity_correlation,
        logger_name=logger.name,
    )
