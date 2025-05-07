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
from pathlib import Path
from typing import Any, Dict, Sequence, Optional
import dask.array as da
import fractal_tasks_core
import pandas as pd
import numpy as np
import zarr
import anndata as ad

from apx_fractal_task_collection.utils import (
    get_acquisition_from_label_name,
    TextureFeatures)
from apx_fractal_task_collection.features.intensity import measure_intensity_features
from apx_fractal_task_collection.features.morphology import (
    measure_morphology_features,
    get_borders_internal,
    get_borders_external,
    get_well_coordinates)
from apx_fractal_task_collection.features.texture import measure_texture_features
from apx_fractal_task_collection.features.population import measure_population_features

from fractal_tasks_core.channels import get_omero_channel_list
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.channels import OmeroChannel, ChannelInputModel
from fractal_tasks_core.channels import get_channel_from_image_zarr
from pydantic import validate_call
from fractal_tasks_core.roi import check_valid_ROI_indices
from fractal_tasks_core.roi import (
    convert_ROI_table_to_indices,
)


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)


@validate_call
def measure_features(  # noqa: C901
        *,
        # Default arguments for fractal tasks:
        zarr_url: str,
        # Task-specific arguments:
        label_image_name: str,
		ROI_table_name: str,
		output_table_name: str,
        channels_to_include: Optional[list[ChannelInputModel]] = None,
        channels_to_exclude: Optional[list[ChannelInputModel]] = None,
        measure_intensity: bool = False,
        measure_morphology: bool = False,
        measure_texture: TextureFeatures = TextureFeatures(),
        measure_population: bool = False,
        calculate_internal_borders: bool = False,
        level: int = 0,
        overwrite: bool = True,
) -> None:
    """
    Calculate features based on label image and intensity image (optional).

    Takes a label image and an optional intensity image and calculates
    morphology, intensity and texture features in 2D.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_image_name: Name of the label image that contains the seeds.
            Needs to exist in OME-Zarr file.
		ROI_table_name: Name of the ROI table to process.
		output_table_name: Name of the feature table.
        channels_to_include: List of channels to include for intensity
            and texture measurements. Use the channel label to indicate
            single channels. If None, all channels are included.
        channels_to_exclude: List of channels to exclude for intensity
            and texture measurements. Use the channel label to indicate
            single channels. If None, no channels are excluded.
        measure_intensity: If True, calculate intensity features.
        measure_morphology: If True, calculate morphology features.
        measure_texture: Select which texture features should be calculated.
        measure_population: If True, calculate population features.
        calculate_internal_borders: For a typical experiment this should
            not be selected. If True, calculate internal borders (whether
            an object touches or overlaps with a FOV border). This
            is only useful if you registered by well and want to remove
            objects that are on the border of a FOV. IMPORTANT: This only
            catches objects that are on the border of FOVs in cycle 1 of a
            multiplexed experiment.
        level: Resolution of the label image to calculate features.
            Only tested for level 0.
        overwrite: If True, overwrite existing feature table.
    """

    well_url = Path(zarr_url).parent
    label_image_cycle = get_acquisition_from_label_name(well_url,
                                                        label_image_name)
    # update the component for the label image if multiplexed experiment
    label_image_url = well_url.joinpath(str(label_image_cycle))

    # get some meta data
    ngff_image_meta = load_NgffImageMeta(zarr_url)
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)
    coarsening_xy = ngff_image_meta.coarsening_xy

    # load ROI table
    ROI_table = ad.read_zarr(f"{zarr_url}/tables/{ROI_table_name}")
        
    well_name = zarr_url.split("/")[-3] + zarr_url.split("/")[-2]

    # get list of texture features to compute:
    texture_feature_list = [measure_texture.haralick,
                            measure_texture.laws_texture_energy]
    texture_feature_names = ['haralick', 'lte']
    texture_feature_list = [f for i, f in enumerate(texture_feature_names) if
                            texture_feature_list[i]]

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=level,
        coarsening_xy=coarsening_xy,
        full_res_pxl_sizes_zyx=full_res_pxl_sizes_zyx,
    )
    check_valid_ROI_indices(list_indices, "registered_well_ROI_table")
    num_ROIs = len(list_indices)

    obs_list = []
    feature_list = []
    for i_ROI, indices in enumerate(list_indices):

        # initialize features for this ROI
        roi_feature_list = []
        # Define region
        s_z, e_z, s_y, e_y, s_x, e_x = indices[:]
        region = (
            slice(0, None),
            slice(s_z, e_z),
            slice(s_y, e_y),
            slice(s_x, e_x),
        )
        logger.info(
            f"Now processing ROI {i_ROI + 1}/{num_ROIs} from ROI table"
            f" {ROI_table_name}."
        )

        # load label image
        logger.info(f"{label_image_url}")
        label_image = da.from_zarr(
            f"{label_image_url}/labels/"
            f"{label_image_name}/{level}")[region[1:]].compute()

        label_image = np.squeeze(label_image)

        # check whether there is objects in the label image and skip if not
        if np.max(label_image) == 0:
            logger.info(f"No objects found in {label_image_name}. "
                        f"Skipping ROI.")
            continue

        # make morphology measurements
        if measure_morphology:
            logger.info(f"Calculating morphology features for "
                        f"{label_image_name}.")
            morphology_features = measure_morphology_features(label_image)

            well_coords = get_well_coordinates(ROI_table,
                                               i_ROI,
                                               morphology_features,
                                               full_res_pxl_sizes_zyx)

            well_coords.rename(columns={
                'centroid-0': 'well_centroid-0',
                'centroid-1': 'well_centroid-1'}, inplace=True)

            morphology_features = pd.merge(morphology_features, well_coords,
                                           on='label')

            # move the well coords after the original coords
            # get index of centroid columns
            centroid_index = morphology_features.columns.get_loc('centroid-1')
            # move the well coords after the original coords
            morphology_features.insert(
                centroid_index + 1,
                'well_centroid-0',
                morphology_features.pop('well_centroid-0')
            )
            morphology_features.insert(
                centroid_index + 2,
                'well_centroid-1',
                morphology_features.pop('well_centroid-1'))

            morphology_features.set_index("label", inplace=True)
            morphology_features.columns = label_image_name +\
                                          "_Morphology_" +\
                                          morphology_features.columns
            roi_feature_list.append(morphology_features)

            # get borders
            borders_external = get_borders_external(ROI_table[i_ROI],
                                                    morphology_features,
                                                    full_res_pxl_sizes_zyx[-1])

            if calculate_internal_borders:
                fov_table = ad.read_zarr(f"{zarr_url}/tables/FOV_ROI_table")
                borders_internal = get_borders_internal(ROI_table[i_ROI],
                                                        fov_table,
                                                        morphology_features,
                                                        full_res_pxl_sizes_zyx[
                                                            -1])
            logger.info(f"Done calculating morphology features "
                        f"for {label_image_name}.")

        if measure_intensity or texture_feature_list:
            intensity_features = []
            texture_features = []
            # get all channels in the acquisition
            channels = get_omero_channel_list(
                image_zarr_path=zarr_url
            )
            if channels_to_include:
                channel_labels_to_include = [c.label for c in channels_to_include]
                channels = [c for c in channels if
                            c.label in channel_labels_to_include]
            if channels_to_exclude:
                channel_labels_to_exclude = [c.label for c in channels_to_exclude]
                channels = [c for c in channels if
                            c.label not in channel_labels_to_exclude]
            # loop over channels and measure intensity and texture features
            for channel in channels:
                tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                    image_zarr_path=zarr_url,
                    wavelength_id=channel.wavelength_id,
                    label=channel.label,
                )
                ind_channel = tmp_channel.index
                data_zyx = da.from_zarr(f"{zarr_url}/{level}")[region][
                    ind_channel].compute()

                # intensity features
                if measure_intensity:
                    logger.info(
                        f"Calculating intensity features for channel"
                        f" {channel.label}.")
                    current_features = measure_intensity_features(
                        np.squeeze(label_image),
                        np.squeeze(data_zyx))
                    current_features.set_index("label", inplace=True)
                    current_features.columns = label_image_name +\
                                               "_Intensity_" +\
                                               current_features.columns +\
                                               "_" + channel.label
                    intensity_features.append(current_features)
                    logger.info(
                        f"Done calculating intensity features for channel"
                        f" {channel.label}.")

                # texture features
                if texture_feature_list:
                    logger.info(
                        f"Calculating texture features for channel "
                        f"{channel.label}.")

                    if "haralick" in texture_feature_list:
                        if channel.label in measure_texture.clip_value_exceptions:
                            current_clip_value = \
                                measure_texture.clip_value_exceptions[channel.label]
                            logger.info(
                                f"Found clip value exception for channel "
                                f"{channel.label}. Clipping at:"
                                f" {current_clip_value}."
                            )
                        else:
                            current_clip_value = measure_texture.clip_value
                            logger.info(
                                f"No clip value exception found for channel "
                                f"{channel.label}. Clipping at:"
                                f" {current_clip_value}."
                            )

                    else:
                        current_clip_value = measure_texture.clip_value

                    current_features = measure_texture_features(
                        label_image=np.squeeze(label_image),
                        intensity_image=np.squeeze(data_zyx),
                        clip_value=current_clip_value,
                        feature_selection=texture_feature_list)

                    current_features.set_index("label", inplace=True)
                    current_features.columns = label_image_name +\
                                               "_Texture_" +\
                                               current_features.columns + \
                                               "_" + channel.label
                    texture_features.append(current_features)
                    logger.info(
                        f"Done calculating texture features for channel"
                        f" {channel.label}.")

            if intensity_features:
                intensity_features = pd.concat(intensity_features, axis=1)
                roi_feature_list.append(intensity_features)
            if texture_features:
                texture_features = pd.concat(texture_features, axis=1)
                roi_feature_list.append(texture_features)

        if measure_population:
            logger.info(f"Calculating population features for "
                        f"{label_image_name}.")

            population_features = measure_population_features(
                input=label_image)
            population_features.set_index("label", inplace=True)
            population_features.columns = label_image_name +\
                                         "_Population_" +\
                                         population_features.columns
            roi_feature_list.append(population_features)
            logger.info(f"Done calculating population features "
                        f"for {label_image_name}.")

        merged_roi_features = pd.concat(roi_feature_list, axis=1)
        merged_roi_features.reset_index(inplace=True)

        feature_list.append(merged_roi_features)

        ROI_obs = pd.DataFrame(
            {"label": merged_roi_features['label'],
             "well_name": well_name,
             "ROI": ROI_table.obs.index[i_ROI]})


        if measure_morphology:
            if calculate_internal_borders:

                ROI_obs['is_border_internal'] = borders_internal.values
                ROI_obs['is_border_external'] = borders_external.values

            else:
                ROI_obs['is_border'] = borders_external.values

        obs_list.append(ROI_obs)

    if feature_list:
        obs = pd.concat(obs_list, axis=0)
        merged_features = pd.concat(feature_list, axis=0)

        merged_features.set_index('label', inplace=True)
        #obs.set_index('label', inplace=True)
        obs.index = np.arange(0, len(obs))

        # save features as AnnData table
        feature_table = ad.AnnData(X=merged_features.reset_index(drop=True),
                                   obs=obs,
                                   dtype='float32')

        # Write to zarr group
        image_group = zarr.group(zarr_url)
        write_table(
            image_group,
            output_table_name,
            feature_table,
            overwrite=overwrite,
            table_attrs={"type": "feature_table",
                         "region": {
                             "path": f"../../{label_image_cycle}/"
                                     f"labels/{label_image_name}"},
                         "instance_key": "label"}
        )

    else:
        logger.info(f"No features calculated for {label_image_name}. Likely,"
                    f" there are no objects in the label image.")

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=measure_features,
        logger_name=logger.name,
    )

