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

import numpy as np
import dask.array as da
import pandas as pd
import anndata as ad

from natsort import natsorted
from phenograph import cluster
from minisom import MiniSom
from pathlib import Path
from typing import Any, Dict, Sequence

from sklearn.preprocessing import FunctionTransformer, StandardScaler
from fractal_tasks_core.lib_write import write_table
from fractal_tasks_core.lib_channels import get_channel_from_image_zarr
from fractal_tasks_core.lib_channels import get_omero_channel_list
from pydantic.decorator import validate_arguments


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


def get_image_from_zarr(zarr_path: str, well_name: str,
                        coords: list, level: int = 0):
    """
    Get image from zarr file.
    Args:
        image_zarr_path: path to zarr file to use (well folder).
        coords: Image coordinates to use.
                Format: [y_start, y_end, x_start, x_end].
        level: pyramid level to use.

    Returns: numpy array of shape
            (n_channels, coords[0]:coords[1], coords[2]:coords[3]).

    """

    zarr_path = Path(zarr_path)
    well_zarr_path = zarr_path.joinpath(f"{well_name[0]}/{well_name[1:]}")
    well_group = zarr.open_group(well_zarr_path, mode="r+")
    img = []
    img_paths = natsorted(
        [image['path'] for image in well_group.attrs['well']["images"]])

    channel_names = []
    for img_path in img_paths:
        omero_channels = get_omero_channel_list(
            image_zarr_path=well_zarr_path.joinpath(img_path))

        for channel in omero_channels:
            tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=well_zarr_path.joinpath(img_path),
                wavelength_id=None,
                label=channel.label
            )
            logger.info(
                f"loading channel: {channel.label} from \n"
                f"well {well_name} (coordinates {coords}) at level {level}")

            channel_names.append(channel.label)
            img.append(da.from_zarr(
                well_zarr_path.joinpath(f"{img_path}/{level}"))[
                       tmp_channel.index, 0, coords[0]:coords[1],
                       coords[2]:coords[3]].compute())


    return np.squeeze(np.stack(img, axis=0)), channel_names


def get_label_from_zarr(zarr_path: str, well_name: str, label_name: str,
                        coords: list, level: int = 0):
    """
    Get label from zarr file.
    Args:
        image_zarr_path: path to zarr file to use (well folder).
        label_name: name of label to use.
        coords: Image coordinates to use.
                Format: [y_start, y_end, x_start, x_end].
        level: pyramid level to use.

    Returns: numpy array of shape (coords[0]:coords[1], coords[2]:coords[3]).

    """

    zarr_path = Path(zarr_path)
    well_zarr_path = zarr_path.joinpath(f"{well_name[0]}/{well_name[1:]}")
    well_group = zarr.open_group(well_zarr_path, mode="r+")
    img_paths = natsorted(
        [image['path'] for image in well_group.attrs['well']["images"]])

    for img_path in img_paths:
        try:
            label = da.from_zarr(well_zarr_path.joinpath(
                f"{img_path}/labels/{label_name}/{level}"))[0,
                    coords[0]:coords[1], coords[2]:coords[3]].compute()
        except:
            continue

    return label


def get_mpps(intensity_image: np.array, labels: np.array, channel_names: list,
             well_name: str):
    """
    Get multiplexed pixel profiles of pixels inside labels from image.
    Args:
        intensity_image: numpy array of shape (n_channels, y, x).
        labels: numpy array of shape (y, x).
        channel_names: list of channel names.
        well_name: name of well.

    Returns: numpy array of shape (n_channels, n_labels).

    """
    # add labels to intensity image to retain their object association
    intensity_image = np.concatenate([intensity_image,
                                      np.expand_dims(labels, 0)], axis=0)
    columns = channel_names.copy()
    columns.extend(['label'])

    # move channel axis to last dimension
    intensity_image = np.moveaxis(intensity_image, 0, -1)
    mpps = intensity_image[labels > 0]

    indices = np.where(labels > 0)

    mpps = pd.DataFrame(mpps, columns=columns)
    mpps['y'] = indices[0]
    mpps['x'] = indices[1]
    mpps['well'] = well_name

    return mpps.set_index(['well', 'y', 'x', 'label'])


def preprocess_mpps(mpps: pd.DataFrame):
    """
    Preprocess multiplexed pixel profiles.
    Args:
        mpps: multiplexed pixel profiles as pandas dataframe.

    Returns: preprocessed multiplexed pixel profiles as pandas dataframe.

    """

    # rescale MPPS to [0, 1] where 1 corresponds to the 98th quantile
    def quantile_scale(x):
        return (x - x.min(axis=0)) / (
                    np.quantile(x, 0.999, axis=0) - x.min(axis=0))

    transformer = FunctionTransformer(quantile_scale)
    mpps_scaled = transformer.fit_transform(mpps)
    mpps_scaled[mpps_scaled > 1] = 1

    # remove rows where all values are > 0.1
    #mpps_scaled = mpps_scaled[~(mpps_scaled < 0.1).all(axis=1)]

    return mpps_scaled


def get_image_from_mpps(mpps: pd.DataFrame, well_name: str,
                        coords: list, column: str):
    """
    Get label map of multiplexed pixel profiles.

    Args:
        mpps: multiplexed pixel profiles as pandas dataframe.
        well_name: name of well.
        coords: Image coordinates to use.
                Format: [y_start, y_end, x_start, x_end].
        column: column to use for image.

    Returns: numpy array of shape (coords[0]:coords[1], coords[2]:coords[3]).

    """
    mpps = mpps.reset_index()
    mpps = mpps.loc[mpps.well == well_name]
    mpps_array = np.zeros((coords[1] - coords[0], coords[3] - coords[2]),
                          dtype='uint16')
    mpps_array[mpps['y'] - coords[0], mpps['x'] - coords[2]] = mpps[
        column].values

    return mpps_array


@validate_arguments
def multiplexed_pixel_clustering(  # noqa: C901
        *,
        # Default arguments for fractal tasks:
        input_paths: Sequence[str],
        output_path: str,
        component: str,
        metadata: Dict[str, Any],
        # Task-specific arguments:
        label_image_name: str,
        channels_to_use: Sequence[str] = None,
        channels_to_exclude: Sequence[str] = None,
        wells_names: Sequence[str] = None,
        coords: Sequence[int] = None,
        level: int = 0,
        output_table_name: str = None,
        overwrite: bool = True,
) -> None:
    """
    Perform multiplexed cell unit (MCU) analysis on a label image. Inspired by
    Gabriele Gut et al., Multiplexed protein maps link subcellular
    organization to cellular states. Science (2018)
    DOI: 10.1126/science.aar7042

    Args:
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
        label_image_name: Name of label image to use.
        channels_to_use: List of channel labels to use for clustering.
                If None, all channels are used.
        channels_to_exclude: List of channel labels to exclude from clustering.
                If None, no channels are excluded.
        wells_names: List of wells to use for pixel clustering.
        coords: Image coordinates to use. If None, the whole well will be used.
                Format: [y_start, y_end, x_start, x_end].
        level: pyramid level to use.
        output_table_name: Name of output table.
        overwrite: If True, overwrite existing output table.
    """

    zarr_path = Path(input_paths[0])
    mpps_list = []
    for well_name in well_names:
        img, channel_names = get_image_from_zarr(zarr_path=zarr_path,
                                                 well_name=well_name,
                                                 level=level,
                                                 coords=coords)

        label = get_label_from_zarr(zarr_path=zarr_path,
                                    well_name=well_name,
                                    label_name='nuclei',
                                    level=level,
                                    coords=coords)

        # calculate multiplexed pixel profiles
        mpps_list.append(get_mpps(img, label,
                                  channel_names=channel_names,
                                  well_name=well_name))

    mpps = pd.concat(mpps_list)

    # add well level coordinates
    new_index = list(mpps.index.names)
    new_index.extend(['well_x', 'well_y'])

    mpps.set_index(mpps.reset_index()['y'].values + coords[0],
                   inplace=True, append=True)
    mpps.set_index(mpps.reset_index()['x'].values + coords[2],
                   inplace=True, append=True)
    mpps.index.rename(names=new_index, inplace=True)

    # only use the specified channels for clustering
    if channels_to_use is not None:
        mpps_filtered = mpps[channels_to_use]
    elif channels_to_exclude is not None:
        mpps_filtered = mpps.drop(columns=channels_to_exclude)

    # preprocess multiplexed pixel profiles
    mpps_scaled = preprocess_mpps(mpps_filtered)

    # train SOM
    logger.info("Training SOM")
    som_data = np.array(mpps_scaled)
    som_shape = (50, 50)
    som = MiniSom(som_shape[0], som_shape[1], som_data.shape[1],
                  sigma=0.3, learning_rate=0.1,
                  neighborhood_function='gaussian', random_seed=10)
    som.random_weights_init(som_data)
    som.train_random(som_data, 100)
    winner_coordinates = np.array([som.winner(x) for x in som_data]).T
    som_cluster = np.ravel_multi_index(winner_coordinates, som_shape)

    # add som_cluster to index
    new_index = list(mpps.index.names)
    new_index.extend(['som_cluster'])

    mpps.set_index(som_cluster + 1, inplace=True, append=True)
    mpps.index.rename(names=new_index, inplace=True)
    mpps_scaled.set_index(som_cluster + 1, inplace=True, append=True)
    mpps_scaled.index.rename(names=new_index, inplace=True)

    # aggregate median values for each cluster
    mpps_scaled_agg = mpps_scaled.groupby('som_cluster').median()

    # cluster with PhenoGraph using jaccard distance
    logger.info("Clustering with PhenoGraph.")
    pheno_labels, graph, Q = cluster(mpps_scaled_agg, k=15, n_jobs=8)

    # assign cluster labels
    mpps_scaled_agg['pheno_cluster'] = pheno_labels + 1
    replace_dict = {i: j for i, j in zip(
        mpps_scaled_agg.reset_index().som_cluster,
        mpps_scaled_agg.reset_index().pheno_cluster)}

    pheno_cluster = mpps_scaled.reset_index().replace(
        {'som_cluster': replace_dict})['som_cluster'].values

    # add pheno_cluster to index
    new_index = list(mpps.index.names)
    new_index.extend(['pheno_cluster'])

    mpps.set_index(pheno_cluster, inplace=True, append=True)
    mpps.index.rename(names=new_index, inplace=True)
    mpps_scaled.set_index(pheno_cluster, inplace=True, append=True)
    mpps_scaled.index.rename(names=new_index, inplace=True)

    # convert to AnnData table and save
    logger.info("Save results to AnnData table.")

    # add z-scored layer
    ss = StandardScaler()
    mpps_normalized = pd.DataFrame(ss.fit_transform(mpps),
                                   columns=mpps.columns,
                                   index=mpps.index)

    obs = mpps.reset_index()
    obs = obs[mpps.index.names]
    #var = mpps.columns.to_frame(name='columns')
    used_for_clustering = [True if c in mpps_filtered.columns
                           else False for c in mpps.columns]
    varm = {'used_for_clustering': np.array(used_for_clustering)}

    mpps2 = ad.AnnData(X=mpps.reset_index(drop=True),
                       obs=obs,
                       varm=varm,
                       layers={
                           'z-scored': np.array(mpps_normalized.reset_index(
                               drop=True))
                       },
                       dtype='uint16')

    # save results
    image_group = zarr.group(f"{zarr_path}")
    write_table(
        image_group,
        output_table_name,
        mpps2,
        overwrite=overwrite,
        logger=logger,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=multiplexed_pixel_clustering,
        logger_name=logger.name,
    )