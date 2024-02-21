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
from typing import Any, Dict, Sequence

import dask.array as da
import fractal_tasks_core
import pandas as pd
from skimage.measure import regionprops_table
import mahotas as mh
import numpy as np
import zarr
import anndata as ad
from typing import Optional
from fractal_tasks_core.lib_channels import get_omero_channel_list
from fractal_tasks_core.lib_ngff import load_NgffImageMeta
from fractal_tasks_core.lib_write import write_table
from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_channels import get_channel_from_image_zarr
from pydantic.decorator import validate_arguments
from fractal_tasks_core.lib_regions_of_interest import check_valid_ROI_indices
from fractal_tasks_core.lib_regions_of_interest import (
    convert_ROI_table_to_indices,
)


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

def sum_intensity(regionmask, intensity_image):
    return np.sum(intensity_image[regionmask])

def std_intensity(regionmask, intensity_image):
    return np.std(intensity_image[regionmask])

def measure_intensity_features(label_image, intensity_image):
    """
    Measure intensity features for label image.
    """
    intensity_features = pd.DataFrame(
        regionprops_table(
            np.squeeze(label_image),
            intensity_image,
            properties=[
                "label",
                "max_intensity",
                "mean_intensity",
                "min_intensity",
                "weighted_moments_hu",
            ],
            extra_properties=[
                sum_intensity,
                std_intensity
            ],
        )
    )
    return intensity_features


def roundness(regionmask):
    return mh.features.roundness(regionmask)


def get_borders_internal(well_table, fov_table, morphology_features,
                         pixel_size_xy):

    well_table = well_table.to_df()
    fov_table = fov_table.to_df()
    obj_name = morphology_features.columns[0].split("_")[0]

    def check_range(row, borders, col_start, col_end):
        start_value = row[col_start]
        end_value = row[col_end]
        range_values = np.arange(start_value, end_value + 1)
        is_in_range = np.isin(borders, range_values)
        return np.any(is_in_range)

    # get internal borders of FOVs
    borders_x = np.unique(
        (fov_table['x_micrometer'] / pixel_size_xy).astype('uint16'))[1:] - \
                np.unique((well_table['x_micrometer'] / pixel_size_xy).astype(
                    'uint16'))[0]
    borders_y = np.unique(
        (fov_table['y_micrometer'] / pixel_size_xy).astype('uint16'))[1:] - \
                np.unique((well_table['y_micrometer'] / pixel_size_xy).astype(
                    'uint16'))[0]

    #a = morphology_features[f'{obj_name}_Morphology_bbox-0'].isin(borders_y)
    #b = morphology_features[f'{obj_name}_Morphology_bbox-1'].isin(borders_x)
    #c = morphology_features[f'{obj_name}_Morphology_bbox-2'].isin(borders_y)
    #d = morphology_features[f'{obj_name}_Morphology_bbox-3'].isin(borders_x)

    e = morphology_features.apply(lambda x:
                              check_range(
                                  x,
                                  borders_y,
                                  f'{obj_name}_Morphology_bbox-0',
                                  f'{obj_name}_Morphology_bbox-2'),
                              axis=1)

    f = morphology_features.apply(lambda x:
                              check_range(
                                  x,
                                  borders_x,
                                  f'{obj_name}_Morphology_bbox-1',
                                  f'{obj_name}_Morphology_bbox-3'),
                              axis=1)


    #is_border_internal = a | b | c | d

    is_border_internal = e | f

    return is_border_internal



def get_borders_external(ROI_table, morphology_features, pixel_size_xy):

    safety_range = 5
    ROI_df = ROI_table.to_df()
    obj_name = morphology_features.columns[0].split("_")[0]

    borders_x = np.unique(
        (ROI_df['x_micrometer'] / pixel_size_xy).astype('uint16')) - np.min(
        ROI_df['x_micrometer'] / pixel_size_xy).astype('uint16')
    borders_x = np.append(borders_x, borders_x[-1] + np.round(
        ROI_df['len_x_micrometer'] / pixel_size_xy).astype('uint16')[0])
    borders_x_start = np.arange(borders_x[0], borders_x[0] + safety_range)
    borders_x_end = np.arange(borders_x[-1] - safety_range, borders_x[-1])

    borders_x = np.append(borders_x, borders_x_start)
    borders_x = np.append(borders_x, borders_x_end)

    borders_y = np.unique(
        (ROI_df['y_micrometer'] / pixel_size_xy).astype('uint16')) - np.min(
        ROI_df['y_micrometer'] / pixel_size_xy).astype('uint16')
    borders_y = np.append(borders_y, borders_y[-1] + np.round(
        ROI_df['len_y_micrometer'] / pixel_size_xy).astype('uint16')[0])
    borders_y_start = np.arange(borders_y[0], borders_y[0] + safety_range)
    borders_y_end = np.arange(borders_y[-1] - safety_range, borders_y[-1])

    borders_y = np.append(borders_y, borders_y_start)
    borders_y = np.append(borders_y, borders_y_end)

    a = morphology_features[f'{obj_name}_Morphology_bbox-0'].isin(borders_y)
    b = morphology_features[f'{obj_name}_Morphology_bbox-1'].isin(borders_x)
    c = morphology_features[f'{obj_name}_Morphology_bbox-2'].isin(borders_y)
    d = morphology_features[f'{obj_name}_Morphology_bbox-3'].isin(borders_x)

    is_border_external = a | b | c | d

    return is_border_external


def measure_morphology_features(label_image):
    """
    Measure morphology features for label image.
    """
    morphology_features = pd.DataFrame(regionprops_table(
        np.squeeze(label_image),
        properties=[
            "label",
            "area",
            "centroid",
            "bbox_area",
            "bbox",
            "convex_area",
            "eccentricity",
            "equivalent_diameter",
            "euler_number",
            "extent",
            "filled_area",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "perimeter",
            "solidity",
        ],
        extra_properties=[
            roundness,
        ],
    )
    )
    morphology_features["circularity"] = (4*np.pi*morphology_features.area) \
                                         / (morphology_features.perimeter**2)
    return morphology_features

# histomicstk version, couldn't get it to install in package as dependency
# def measure_texture_features(label_image, intensity_image):
#     """
#     Measure texture features for label image.
#     """
#
#     texture_features = compute_haralick_features(label_image, intensity_image)
#     texture_features['label'] = np.unique(label_image[np.nonzero(label_image)])
#
#     return texture_features


def haralick_features(regionmask, intensity_image):
    haralick_values_list = []
    masked_image = np.where(regionmask > 0, intensity_image, 0)
    for distance in [2, 5]:
        try:
            haralick_values = mh.features.haralick(
                masked_image.astype('uint8'),
                distance=distance,
                return_mean=True,
                ignore_zeros=True)
        except ValueError:
            haralick_values = np.full(13, np.NaN, dtype=float)

        haralick_values_list.extend(haralick_values)
    return haralick_values_list


def measure_texture_features(label_image, intensity_image):
    """
    Measure texture features for label image.
    """

    # NOTE: Haralick features are computed on 8-bit images.
    clip_value = np.percentile(intensity_image, 99.999)
    clipped_img = np.clip(intensity_image, 0, clip_value).astype('uint16')
    rescaled_img = mh.stretch(clipped_img)

    names = ['angular-second-moment', 'contrast', 'correlation',
             'sum-of-squares', 'inverse-diff-moment', 'sum-avg',
             'sum-var', 'sum-entropy', 'entropy', 'diff-var',
             'diff-entropy', 'info-measure-corr-1', 'info-measure-corr-2']

    names = [
        f"Haralick-{name}-{distance}" for distance in [2, 5] for name in names]

    texture_features = pd.DataFrame(
        regionprops_table(label_image,
                          rescaled_img,
                          properties=['label'],
                          extra_properties=[haralick_features]))

    texture_features.set_index('label', inplace=True)
    texture_features.columns = names
    texture_features.reset_index(inplace=True)
    return texture_features


@validate_arguments
def measure_features(  # noqa: C901
        *,
        # Default arguments for fractal tasks:
        input_paths: Sequence[str],
        output_path: str,
        component: str,
        metadata: Dict[str, Any],
        # Task-specific arguments:
        label_image_name: str,
        label_image_cycle: Optional[int] = None,
        measure_intensity: bool = False,
        measure_morphology: bool = False,
        measure_texture: bool = False,
        ROI_table_name: str = None,
        output_table_name: str = None,
        level: int = 0,
        overwrite: bool = True,
) -> None:
    """
    Calculate features based on label image and intensity image (optional).

    Takes a label image and an optional intensity image and calculates
    morphology, intensity and texture features in 2D.

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
        label_image_name: Name of the label image that contains the seeds.
            Needs to exist in OME-Zarr file.
        label_image_cycle: indicates which cycle contains the label image
            (only needed if multiplexed).
        measure_intensity: If True, calculate intensity features.
        measure_morphology: If True, calculate morphology features.
        measure_texture: If True, calculate texture features.
        ROI_table_name: Name of the ROI table to process.
        calculate_internal_borders: For a typical experiment this should
            not be selected. If True, calculate internal borders (whether
            an object touches or overlaps with a FOV border). This
            is only useful if you registered by well and want to remove
            objects that are on the border of a FOV. IMPORTANT: This only
            removes objects that are on the border of FOVs in cycle 1 of a
            multiplexed experiment.
        output_table_name: Name of the feature table.
        level: Resolution of the label image to calculate features.
            Only tested for level 0.
        overwrite: If True, overwrite existing label image.
    """

    # update the component for the label image if multiplexed experiment
    if label_image_cycle is not None:
        parts = component.rsplit("/", 1)
        label_image_component = parts[0] + "/" + str(label_image_cycle)
    else:
        label_image_component = component

    in_path = Path(input_paths[0])
    # get some meta data
    ngff_image_meta = load_NgffImageMeta(in_path.joinpath(component))
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    coarsening_xy = ngff_image_meta.coarsening_xy

    # load ROI table
    ROI_table = ad.read_zarr(in_path.joinpath(component, "tables",
                                               ROI_table_name))

    fov_table = ad.read_zarr(in_path.joinpath(component, "tables",
                                              'FOV_ROI_table'))

    well_name = component.split("/")[1] + component.split("/")[2]

    # Create list of indices for 3D FOVs spanning the entire Z direction
    list_indices = convert_ROI_table_to_indices(
        ROI_table,
        level=0,
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

        # make morphology measurements
        # load label image
        label_image = da.from_zarr(
            f"{in_path}/{label_image_component}/labels/"
            f"{label_image_name}/{level}")[region[1:]].compute()

        if measure_morphology:
            logger.info(f"Calculating morphology features for "
                        f"{label_image_name}.")
            morphology_features = measure_morphology_features(label_image)
            morphology_features.set_index("label", inplace=True)

            # add well centroid position:
            ROI_df = ROI_table.to_df()
            x_offset =\
                int((ROI_df.iloc[i_ROI]['x_micrometer'] -
                     ROI_df['x_micrometer'].min()) /full_res_pxl_sizes_zyx[-1])
            y_offset =\
                int((ROI_df.iloc[i_ROI]['y_micrometer'] -
                     ROI_df['y_micrometer'].min()) /full_res_pxl_sizes_zyx[-1])

            well_centroid_0 =\
                morphology_features['centroid-0'] + y_offset
            well_centroid_1 = \
                morphology_features['centroid-1'] + x_offset

            # get index of centroid columns
            centroid_index = morphology_features.columns.get_loc('centroid-1')

            # insert new centroid columns
            morphology_features.insert(centroid_index + 1,
                                       f'well_centroid-0',
                                       well_centroid_0)
            morphology_features.insert(centroid_index + 2,
                                       f'well_centroid-1',
                                       well_centroid_1)


            morphology_features.columns = label_image_name +\
                                          "_Morphology_" +\
                                          morphology_features.columns
            roi_feature_list.append(morphology_features)
            logger.info(f"Done calculating morphology features "
                        f"for {label_image_name}.")


        if measure_intensity or measure_texture:
            intensity_features = []
            texture_features = []
            # get all channels in the acquisition
            zarrurl = (in_path.resolve() / component).as_posix()
            channels = get_omero_channel_list(
                image_zarr_path=zarrurl
            )
            # loop over channels and measure intensity and texture features
            for channel in channels:
                tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                    image_zarr_path=zarrurl,
                    wavelength_id=channel.wavelength_id,
                    label=channel.label,
                )
                ind_channel = tmp_channel.index
                data_zyx = da.from_zarr(f"{zarrurl}/{level}")[region][
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
                if measure_texture:
                    logger.info(
                        f"Calculating texture features for channel "
                        f"{channel.label}.")
                    current_features = measure_texture_features(
                        np.squeeze(label_image),
                        np.squeeze(data_zyx))
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

        merged_roi_features = pd.concat(roi_feature_list, axis=1)
        merged_roi_features.reset_index(inplace=True)

        feature_list.append(merged_roi_features)

        borders_external = get_borders_external(ROI_table[i_ROI],
                                                morphology_features,
                                                full_res_pxl_sizes_zyx[-1])

        if calculate_internal_borders:
            borders_internal = get_borders_internal(ROI_table[i_ROI],
                                                    fov_table,
                                                    morphology_features,
                                                    full_res_pxl_sizes_zyx[-1])

            ROI_obs = pd.DataFrame(
                {"ROI": ROI_table.obs.index[i_ROI],
                 "is_border_internal": borders_internal.values,
                 "is_border_external": borders_external.values,
                 "label": merged_roi_features['label'],
                 "well_name": well_name})
        else:
            ROI_obs = pd.DataFrame({"ROI": ROI_table.obs.index[i_ROI],
                                    "is_border": borders_external.values,
                                    "label": merged_roi_features['label'],
                                    "well_name": well_name})

        obs_list.append(ROI_obs)

    obs = pd.concat(obs_list, axis=0)
    merged_features = pd.concat(feature_list, axis=0)

    merged_features.set_index('label', inplace=True)
    obs.set_index('label', inplace=True)

    # save features as AnnData table
    feature_table = ad.AnnData(X=merged_features.values,
                               obs=obs,
                               var=merged_features.columns.to_frame(),
                               dtype='float32')


    # Write to zarr group
    image_group = zarr.group(f"{in_path}/{component}")
    write_table(
        image_group,
        output_table_name,
        feature_table,
        overwrite=overwrite,
        logger=logger,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=measure_features,
        logger_name=logger.name,
    )

