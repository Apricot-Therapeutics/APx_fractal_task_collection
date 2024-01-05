"""
# Copyright 2022 (C) Friedrich Miescher Institute for Biomedical Research and
# University of Zurich
#
# Original authors:
# Tommaso Comparin <tommaso.comparin@exact-lab.it>
# Marco Franzon <marco.franzon@exact-lab.it>
# Joel Lüthi  <joel.luethi@fmi.ch>
#
# This file is part of Fractal and was originally developed by eXact lab S.r.l.
# <exact-lab.it> under contract with Liberali Lab from the Friedrich Miescher
# Institute for Biomedical Research and Pelkmans Lab from the University of
# Zurich.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Sequence

import dask.array as da
import fractal_tasks_core
import pandas as pd
import skimage
import mahotas as mh
import numpy as np
import zarr
import anndata as ad
from typing import Optional
from fractal_tasks_core.lib_channels import get_omero_channel_list
from fractal_tasks_core.lib_write import write_table
from fractal_tasks_core.lib_channels import OmeroChannel
from fractal_tasks_core.lib_channels import get_channel_from_image_zarr
from pydantic.decorator import validate_arguments


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
        skimage.measure.regionprops_table(
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


def measure_morphology_features(label_image):
    """
    Measure morphology features for label image.
    """
    morphology_features = pd.DataFrame(skimage.measure.regionprops_table(
        np.squeeze(label_image),
        properties=[
            "label",
            "area",
            "centroid",
            "bbox_area",
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
    morphology_features["circularity"] = (4 * np.pi * morphology_features.area / (morphology_features.perimeter ** 2))
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
    for distance in [2, 5]:
        try:
            haralick_values = mh.features.haralick(
                intensity_image.astype('uint16'),
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

    names = ['angular-second-moment', 'contrast', 'correlation',
             'sum-of-squares', 'inverse-diff-moment', 'sum-avg',
             'sum-var', 'sum-entropy', 'entropy', 'diff-var',
             'diff-entropy', 'info-measure-corr-1', 'info-measure-corr-2']

    names = [f"Haralick-{name}-{distance}" for distance in [2, 5] for name in names]

    texture_features = pd.DataFrame(regionprops_table(label_image, intensity_image,
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
        output_table_name: str = None,
        level: int = 0,
        overwrite: bool = True,
) -> None:
    """
    Calculate features based on label image and intensity image (optional).

    Takes a label image and an optional intensity image and calculates morphology,
    intensity and texture features in 2D.

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
        label_image_cycle: indicates which cycle contains the label image (only needed if multiplexed).
        measure_intensity: If True, calculate intensity features.
        measure_morphology: If True, calculate morphology features.
        measure_texture: If True, calculate texture features.
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

    # load label image
    label_image = da.from_zarr(
        f"{in_path}/{label_image_component}/labels/{label_image_name}/{level}"
    ).compute()

    feature_list = []
    # make morphology measurements
    if measure_morphology:
        logger.info(f"Calculating morphology features for {label_image_name}.")
        morphology_features = measure_morphology_features(label_image)
        morphology_features.set_index("label", inplace=True)
        morphology_features.columns = label_image_name +"_Morphology_" + morphology_features.columns
        feature_list.append(morphology_features)
        logger.info(f"Done calculating morphology features for {label_image_name}.")

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
            data_zyx = da.from_zarr(f"{zarrurl}/{level}")[
                ind_channel].compute()

            # intensity features
            if measure_intensity:
                logger.info(
                    f"Calculating intensity features for channel {channel.label}.")
                current_features = measure_intensity_features(
                    np.squeeze(label_image),
                    np.squeeze(data_zyx))
                current_features.set_index("label", inplace=True)
                current_features.columns = label_image_name + "_Intensity_" + current_features.columns + "_" + channel.label
                intensity_features.append(current_features)
                logger.info(
                    f"Done calculating intensity features for channel {channel.label}.")

            # texture features
            if measure_texture:
                logger.info(
                    f"Calculating texture features for channel {channel.label}.")
                current_features = measure_texture_features(
                    np.squeeze(label_image),
                    np.squeeze(data_zyx))
                current_features.set_index("label", inplace=True)
                current_features.columns = label_image_name + "_Texture_" + current_features.columns + "_" + channel.label
                texture_features.append(current_features)
                logger.info(
                    f"Done calculating texture features for channel {channel.label}.")

        if intensity_features:
            intensity_features = pd.concat(intensity_features, axis=1)
            feature_list.append(intensity_features)
        if texture_features:
            texture_features = pd.concat(texture_features, axis=1)
            feature_list.append(texture_features)

    merged_features = pd.concat(feature_list, axis=1)
    merged_features.reset_index(inplace=True)
    # save features as AnnData table
    well_name = component.split("/")[1] + component.split("/")[2]
    observation_info = pd.DataFrame({"label": merged_features.index,
                                     "well": well_name})
    feature_table = ad.AnnData(X=merged_features.drop("label", axis=1), obs=observation_info)

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

