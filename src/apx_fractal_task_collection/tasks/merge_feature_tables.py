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
import fractal_tasks_core
import pandas as pd
import anndata as ad
from pydantic import validate_call
from typing import Optional
import zarr

from fractal_tasks_core.tables import write_table

__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__

logger = logging.getLogger(__name__)


@validate_call
def merge_feature_tables(  # noqa: C901
        *,
        # Default arguments for fractal tasks:
        zarr_url: str,
        # Task-specific arguments:
        feature_table_1: str,
        feature_table_2: str,
        left_on: str = "well_name",
        right_on: Optional[str] = None,
        how: str = 'inner',
        suffixes: Optional[list] = None,
        ignore_duplicate_columns: bool = True,
        new_feature_table_name: Optional[str] = None,
) -> None:
    """
    Merge two Fractal feature tables (for example cells and nuclei).

    Takes two Fractal feature tables and merges them based on a common column.
    The resulting table will contain all columns from both tables. The new
    table will inherit its attributes from the first table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        feature_table_1: Name of the first feature table to be merged.
        feature_table_2: Name of the second feature table to be merged.
        left_on: Column name in feature table 1 to merge on.
        right_on: Column name in feature table 2 to merge on.
            If None, it defaults to left_on.
        how: Type of merge to be performed. Default is 'inner'.
        suffixes: Suffixes to be added to overlapping column names. If
            None are provided, the name of the object to which the feature
            table relates will be added as a suffix.
        ignore_duplicate_columns: If True, duplicate columns will be taken
            from the first table. If False, duplicate columns be merged
            with a suffix.
        new_feature_table_name: Name of the new feature table to be created.
            If None, it defaults to the original feature table name.

    """

    logger.info(f"Loading feature table {feature_table_1} and"
                f" {feature_table_2} from {zarr_url}")

    # read feature tables
    ft_1 = ad.read_zarr(f"{zarr_url}/tables/{feature_table_1}")
    ft_2 = ad.read_zarr(f"{zarr_url}/tables/{feature_table_2}")

    # for each feature table, find info about the object it relates to
    table_group = zarr.open(f"{zarr_url}/tables/{feature_table_1}", mode="a")
    object_1 = table_group.attrs["region"]['path'].split("/")[-1]

    table_group = zarr.open(f"{zarr_url}/tables/{feature_table_2}", mode="a")
    object_2 = table_group.attrs["region"]['path'].split("/")[-1]

    df_1 = ft_1.to_df()
    df_1[ft_1.obs.columns.to_list()] = ft_1.obs

    df_2 = ft_2.to_df()
    df_2[ft_2.obs.columns.to_list()] = ft_2.obs


    if suffixes is None:
        suffixes = ['', f"_{object_2}"]

    # get columns to use for merge
    if ignore_duplicate_columns:
        logger.info("Ignoring duplicate columns.")
        cols_to_use = df_2.columns.difference(df_1.columns).to_list()
        cols_to_use.append("label")
    else:
        cols_to_use = df_2.columns

    out_df = pd.merge(df_1, df_2[cols_to_use],
                      left_on=left_on,
                      right_on=right_on,
                      how=how,
                      suffixes=tuple(suffixes),
                      )

    # get all obs columns across two dataframes
    if ignore_duplicate_columns:
        obs_cols = list(set(ft_1.obs.columns.to_list()).union(
            set(ft_2.obs.columns.to_list())))
    else:
        # find obs that are uniquely in ft_2
        unique_obs_ft2 = ft_2.obs.columns.difference(ft_1.obs.columns).to_list()

        ft2_obs = ft_2.obs.copy()
        ft2_obs_columns = [f"{col}{suffixes[1]}" if col not in unique_obs_ft2
                           else col for col in ft2_obs.columns]
        obs_cols = list(set(ft_1.obs.columns.to_list()).union(
            set(ft2_obs_columns)))

    obs_cols.remove("label")
    obs_cols.insert(0, f"label{suffixes[1]}")
    obs_cols.insert(0, f"label{suffixes[0]}")

    feature_cols = out_df.columns.difference(obs_cols).to_list()

    out_df = out_df[obs_cols + feature_cols]


    merged_feature_table = ad.AnnData(X=out_df[feature_cols],
                                   obs=out_df[obs_cols],
                                   dtype='float32')

    # get original table attributes
    table_group = zarr.open_group(
        f"{zarr_url}/tables/{feature_table_1}",
        mode='r')
    orig_attrs = table_group.attrs.asdict()

    if new_feature_table_name is None:
        new_feature_table_name = feature_table_1

    out_group = zarr.open_group(zarr_url, mode="r+")

    # Write to zarr group
    write_table(
        out_group,
        new_feature_table_name,
        merged_feature_table,
        overwrite=True,
        table_attrs=orig_attrs,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=merge_feature_tables,
        logger_name=logger.name,
    )