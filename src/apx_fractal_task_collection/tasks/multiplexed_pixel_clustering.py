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

from phenograph import cluster
from minisom import MiniSom
from typing import Sequence, Optional

from sklearn.preprocessing import (StandardScaler,
                                   FunctionTransformer)

from apx_fractal_task_collection.init_utils import (group_by_well,
                                                    get_label_zarr_url)

from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.channels import OmeroChannel
from fractal_tasks_core.utils import rescale_datasets
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.tables import write_table
from fractal_tasks_core.channels import get_channel_from_image_zarr
from fractal_tasks_core.channels import get_omero_channel_list
from pydantic import validate_call


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


def get_image_from_zarr(well_list: list[str], well_name: str,
                        coords: list, level: int = 0):
    """
    Get numpy array with all channels across multiplexing cycles
    from specified well.

    Args:
        well_list: list of urls of zarr-images in well.
        well_name: name of the well to use.
        coords: Image coordinates to use.
                Format: [y_start, y_end, x_start, x_end].
        level: pyramid level to use.

    Returns: numpy array of shape
            (n_channels, coords[0]:coords[1], coords[2]:coords[3]).

    """

    img = []
    channel_names = []
    for zarr_url in well_list:
        omero_channels = get_omero_channel_list(
            image_zarr_path=zarr_url)

        for channel in omero_channels:
            tmp_channel: OmeroChannel = get_channel_from_image_zarr(
                image_zarr_path=zarr_url,
                wavelength_id=None,
                label=channel.label
            )

            channel_names.append(channel.label)
            current_img = da.from_zarr(f"{zarr_url}/{level}")[
                       tmp_channel.index, 0, coords[0]:coords[1],
                       coords[2]:coords[3]]
            img.append(current_img.compute())

            # extremely clumsy way to get the actual shape of the image
            actual_shape = [coords[0],
                            current_img.shape[0]+coords[0],
                            coords[2],
                            current_img.shape[1]+coords[2]]
            logger.info(
                f"loaded channel: {channel.label} from well {well_name} \n"
                f"from coordinates {actual_shape} at level {level}")

    return np.squeeze(np.stack(img, axis=0)), channel_names



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


def filter_mpps(mpps: pd.DataFrame):
    """
    Filter multiplexed pixel profiles.
    Args:
        mpps: multiplexed pixel profiles as pandas dataframe.
        channels_to_use: list of channel labels to use
        channels_to_exclude: list of channel labels to exclude

    Returns: filtered multiplexed pixel profiles as pandas dataframe where
    pixels that are above 0.999 quantile in one channel or below 0.33 quantile
    across all channels are removed

    """

    mpps_filtered = mpps.copy()

    # identify pixels that have intensity < 0.1
    # quantile gray values across all channels that should be
    # used for clustering

    quantile_threshold = 0.1
    mpps_filtered['low_quantile_filter'] = mpps_filtered.apply(
        lambda col: col < col.quantile(quantile_threshold), axis=0).all(axis=1)
    mpps_filtered.set_index('low_quantile_filter', append=True, inplace=True)

    # identify pixels that have intensity < 0.999
    # quantile gray values across all channels that should be
    # used for clustering
    #quantile_threshold = 0.999
    #mpps['high_quantile_filter'] = mpps_filtered.apply(
    #    lambda col: col < col.quantile(quantile_threshold), axis=0).all(axis=1)
    #mpps.set_index('high_quantile_filter', append=True, inplace=True)

    # remove pixels for which one of the filters is true
    #mpps_filtered = mpps.loc[
    #    (mpps.index.get_level_values('high_quantile_filter').values) &
    #    (mpps.index.get_level_values('low_quantile_filter').values)]

    mpps_filtered = mpps_filtered.loc[
        ~mpps_filtered.index.get_level_values('low_quantile_filter').values]

    #mpps_filtered.reset_index(level=(-2, -1), drop=True, inplace=True)
    mpps_filtered.reset_index(level=(-1), drop=True, inplace=True)

    return mpps_filtered


def scale_mpps(mpps: pd.DataFrame):
    """
    Preprocess multiplexed pixel profiles.
    Args:
        mpps: multiplexed pixel profiles as pandas dataframe.

    Returns: preprocessed multiplexed pixel profiles as pandas dataframe.

    """

    def quantile_scale(x):
        return (x - x.min(axis=0)) / (
                np.quantile(x, 0.99, axis=0) - x.min(axis=0))

    transformer = FunctionTransformer(quantile_scale)
    mpps_scaled = pd.DataFrame(transformer.fit_transform(mpps),
                               columns=mpps.columns,
                               index=mpps.index)


    return mpps_scaled


def get_image_from_mpps(mpps: pd.DataFrame, well_name: str,
                        shape: list, column: str):
    """
    Get label map of multiplexed pixel profiles.

    Args:
        mpps: multiplexed pixel profiles as pandas dataframe.
        well_name: name of well.
        coords: Image coordinates to use.
                Format: [y_start, y_end, x_start, x_end].
        shape: shape of label map.
        column: column to use for image.

    Returns: numpy array of shape (coords[0]:coords[1], coords[2]:coords[3]).

    """
    mpps = mpps.reset_index()
    mpps = mpps.loc[mpps.well == well_name]
    mpps_array = np.zeros(shape, dtype='uint32')
    mpps_array[
        0,
        mpps['well_x'],
        mpps['well_y']
    ] = mpps[column].values

    return mpps_array


@validate_call
def multiplexed_pixel_clustering(  # noqa: C901
        *,
        # Default arguments for fractal tasks:
        zarr_urls: list[str],
        zarr_dir: str,
        # Task-specific arguments:
        label_image_name: str,
        channels_to_use: Optional[list[str]] = None,
        channels_to_exclude: Optional[list[str]] = None,
        well_names: Sequence[str],
        som_shape: Sequence[int] = (20, 20),
        phenograph_neighbours: int = 15,
        enforce_equal_object_count: bool = False,
        seed: Optional[int] = None,
        coords: Optional[list[int]] = None,
        level: int = 0,
        output_table_name: str,
        output_label_name: str,
        overwrite: bool = True,
) -> None:
    """
    Perform multiplexed cell unit (MCU) analysis on a label image. Inspired by
    Gabriele Gut et al., Multiplexed protein maps link subcellular
    organization to cellular states. Science (2018)
    DOI: 10.1126/science.aar7042

    Args:
        Args:
        zarr_urls: List of paths or urls to the individual OME-Zarr image to
            be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        zarr_dir: path of the directory where the new OME-Zarrs will be
            created.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_image_name: Name of label image to use. Only pixels that are part
            of a label will be considered for clustering.
        channels_to_use: List of channel labels to use for clustering.
                If None, all channels are used.
        channels_to_exclude: List of channel labels to exclude from clustering.
                If None, no channels are excluded.
        wells_names: List of wells to use for pixel clustering. For example
            ["C03", "C04"].
        som_shape: Shape of the self-organizing map (SOM) to use for clustering.
        phenograph_neighbours: Number of nearest neighbors to use in first
                step of graph construction.
        enforce_equal_object_count: If True, the same number of objects from
            the label images will be used to extract pixels for clustering.
        seed: Random seed for selection of objects if enforce_equal_object_count
            is True.
        coords: Image coordinates to use. If None, the whole well will be used.
                Format: [y_start, y_end, x_start, x_end].
        level: pyramid level to use.
        output_table_name: Name of output table.
        output_label_name: Name of the output label image which will map the
                multiplexed pixel clusters
        overwrite: If True, overwrite existing output table.
    """

    # filter the image list to only include images in wells that were specified
    zarr_urls = [zarr_url for zarr_url in zarr_urls
                    if zarr_url.rsplit("/", 3)[1] + zarr_url.rsplit("/", 2)[1] in well_names]

    well_dict = group_by_well(zarr_urls)
    mpps_list = []

    # if coords is None, use the whole well:
    if coords is None:
        coords = [0, None, 0, None]

    for well_name, well_list in well_dict.items():

        well_id = well_name.rsplit("/", 2)[1] + well_name.rsplit("/", 1)[1]

        # get image of all channels of the specified region in the well
        img, channel_names = get_image_from_zarr(well_list=well_list,
                                                 well_name=well_id,
                                                 level=level,
                                                 coords=coords)

        # get the label image
        label_zarr_url = get_label_zarr_url(well_list, label_image_name)
        label = da.from_zarr(f"{label_zarr_url}/labels/"
                             f"{label_image_name}/{level}")[
                0,
                coords[0]:coords[1],
                coords[2]:coords[3]].compute()

        # calculate multiplexed pixel profiles (label is cast to uint16
        # to save memory)
        mpps_list.append(get_mpps(img, label.astype('uint16'),
                                  channel_names=channel_names,
                                  well_name=well_id))

    mpps = pd.concat(mpps_list)

    # add well level coordinates
    new_index = list(mpps.index.names)
    new_index.extend(['well_x', 'well_y'])

    mpps.set_index(mpps.reset_index()['y'].values + coords[0],
                   inplace=True, append=True)
    mpps.set_index(mpps.reset_index()['x'].values + coords[2],
                   inplace=True, append=True)
    mpps.index.rename(names=new_index, inplace=True)

    if enforce_equal_object_count:
        # set numpy.random seed
        if seed is not None:
            np.random.seed(seed=seed)
            logger.info(f"Using seed {seed} for random selection of objects.")

        # enforce that the same number of cells are used per well
        n_cells = mpps.reset_index().groupby(['well'])['label'].nunique()
        min_cells = n_cells.min()
        mpps = mpps.reset_index().groupby('well', as_index=False).apply(
            lambda x: x.loc[x['label'].isin(
                np.random.choice(x['label'].unique(),
                                 min_cells,
                                 replace=False))])
        mpps.set_index(new_index, inplace=True)
        logger.info(f"Using {min_cells} randomly sampled objects per well.")

    # quantile filter mpps
    mpps = filter_mpps(mpps)

    if channels_to_use is not None:
        mpps_filtered = mpps[channels_to_use]
    elif channels_to_exclude is not None:
        mpps_filtered = mpps.drop(columns=channels_to_exclude)
    elif channels_to_use is None and channels_to_exclude is None:
        mpps_filtered = mpps

    # scale mpps
    mpps_scaled = scale_mpps(mpps_filtered)

    # train SOM
    logger.info("Training SOM")
    som_data = np.array(mpps_scaled)
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
    pheno_labels, graph, Q = cluster(mpps_scaled_agg,
                                     k=phenograph_neighbours,
                                     n_jobs=8)

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
    #var = mpps_filtered.columns.to_frame(name='columns')
    used_for_clustering = [True if c in mpps_filtered.columns
                           else False for c in mpps.columns]

    varm = {'used_for_clustering': np.array(used_for_clustering)}

    mpps_ad = ad.AnnData(X=mpps.reset_index(drop=True),
                         obs=obs,
                         varm=varm,
                         layers={
                           'z-scored': np.array(mpps_normalized.reset_index(
                               drop=True))
                         },
                         dtype='uint16')


    # save results
    plate_path = zarr_urls[0].rsplit("/", 3)[0]
    image_group = zarr.group(plate_path)
    write_table(
        image_group,
        output_table_name,
        mpps_ad,
        overwrite=overwrite,
        table_attrs={"type": "feature_table",
                     "region": {
                         "path": f"../../{label_zarr_url.split('/')[-1]}/"
                                 f"labels/{output_label_name}"},
                     "instance_key": "label"}
    )

    # save MCU label map to labels
    # prepare label image

    data_zyx = da.from_zarr(f"{label_zarr_url}/labels/"
                            f"{label_image_name}/{level}")

    ngff_image_meta = load_NgffImageMeta(label_zarr_url)
    num_levels = ngff_image_meta.num_levels
    coarsening_xy = ngff_image_meta.coarsening_xy
    full_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=0)
    actual_res_pxl_sizes_zyx = ngff_image_meta.get_pixel_sizes_zyx(level=level)

    # Rescale datasets (only relevant for level>0)
    if ngff_image_meta.axes_names[0] != "c":
        raise ValueError(
            "Cannot set `remove_channel_axis=True` for multiscale "
            f"metadata with axes={ngff_image_meta.axes_names}. "
            'First axis should have name "c".'
        )
    new_datasets = rescale_datasets(
        datasets=[ds.dict() for ds in ngff_image_meta.datasets],
        coarsening_xy=coarsening_xy,
        reference_level=level,
        remove_channel_axis=True,
    )

    label_attrs = {
        "image-label": {
            "version": __OME_NGFF_VERSION__,
            "source": {"image": "../../"},
        },
        "multiscales": [
            {
                "name": output_label_name,
                "version": __OME_NGFF_VERSION__,
                "axes": [
                    ax.dict()
                    for ax in ngff_image_meta.multiscale.axes
                    if ax.type != "channel"
                ],
                "datasets": new_datasets,
            }
        ],
    }

    label_dtype = np.uint32

    shape = data_zyx.shape
    if len(shape) == 2:
        shape = (1, *shape)
    chunks = data_zyx.chunksize
    if len(chunks) == 2:
        chunks = (1, *chunks)

    # Compute and store 0-th level to disk
    for well_name, well_list in well_dict.items():
        well_id = well_name.rsplit("/", 2)[1] + well_name.rsplit("/", 1)[1]

        # get the label map for multiplexed units
        mcu_labels = get_image_from_mpps(mpps,
                                         well_name=well_id,
                                         shape=shape,
                                         column='pheno_cluster')

        label_zarr_url = get_label_zarr_url(well_list, label_image_name)

        image_group = zarr.group(label_zarr_url)

        label_group = prepare_label_group(
            image_group,
            output_label_name,
            overwrite=overwrite,
            label_attrs=label_attrs,
            logger=logger,
        )

        logger.info(
            f"Helper function `prepare_label_group` returned {label_group=}"
        )

        out = f"{label_zarr_url}/labels/{output_label_name}/0"
        logger.info(f"Output label path: {out}")
        store = zarr.storage.FSStore(str(out))

        label_zarr = zarr.create(
            shape=shape,
            chunks=chunks,
            dtype=label_dtype,
            store=store,
            overwrite=False,
            dimension_separator="/",
        )

        logger.info(
            f"label will have shape {data_zyx.shape} "
            f"and chunks {data_zyx.chunks}"
        )

        da.array(mcu_labels).to_zarr(
            url=label_zarr,
            compute=True,
        )

        logger.info(
            f"Saved Multiplexed Pixel Map for {out}."
            "now building pyramids."
        )

        # Starting from on-disk highest-resolution data, build and write to
        # disk a pyramid of coarser levels
        build_pyramid(
            zarrurl=out.rsplit("/", 1)[0],
            overwrite=overwrite,
            num_levels=num_levels,
            coarsening_xy=coarsening_xy,
        )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=multiplexed_pixel_clustering,
        logger_name=logger.name,
    )