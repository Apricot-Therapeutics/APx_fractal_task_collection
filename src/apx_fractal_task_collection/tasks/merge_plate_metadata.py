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
def merge_plate_metadata(  # noqa: C901
        *,
        # Default arguments for fractal tasks:
        zarr_url: str,
        # Task-specific arguments:
        feature_table_name: str,
        metadata_path: str,
        left_on: str = "well_name",
        right_on: Optional[str] = None,
        new_feature_table_name: Optional[str] = None,
) -> None:
    """
    Merge a metadata csv file with a Fractal feature table.

    Takes a csv file containing metadata and merges it into a Fractal
    feature table. The metadata columns will appear in the obs of the
    feature table.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        feature_table_name: Name of the feature table to be merged with the metadata.
        metadata_path: Path to the metadata csv file.
        left_on: Column name in the feature table to merge on.
        right_on: Column name in the metadata to merge on.
            If None, it defaults to left_on.
        new_feature_table_name: Name of the new feature table to be created.
            If None, it defaults to the original feature table name.
    """

    # ngff_image = NgffImage(zarr_url, "r")
    # # read the feature table
    # feature_table = ngff_image.tables.get_table(feature_table_name)
    # # read the plate metadata
    # metadata = pd.read_csv(metadata_path)
    #
    # # merge feature table with metadata
    # if right_on is None:
    #     right_on = left_on
    # new_feature_table = pd.merge(feature_table.table.reset_index(),
    #                              metadata,
    #                              left_on=left_on,
    #                              right_on=right_on)
    #
    # # create a new feature table in the zarr
    # feat_table = ngff_image.tables.new(
    #     name=new_feature_table_name,
    #     label_image=f'../{feature_table.source_label()}',
    #     table_type='feature_table',
    #     overwrite=True
    # )
    #
    # feat_table.set_table(new_feature_table)
    # feat_table.consolidate()

    logger.info(f"Loading feature table {feature_table_name} from {zarr_url}")
    feature_table = ad.read_zarr(f"{zarr_url}/tables/{feature_table_name}")

    logger.info(f"Loading metadata from {metadata_path}")
    metadata = pd.read_csv(metadata_path)

    # merge feature table with metadata
    if right_on is None:
        right_on = left_on

    new_obs = pd.merge(feature_table.obs,
                       metadata,
                       left_on=left_on,
                       right_on=right_on)

    # replace missing values with string NA (np.nan or pd.NA does not work
    # for some reason when saving the anndata table)
    new_obs = new_obs.where(pd.notnull(new_obs), "NA")

    # drop the right_on column if it is not the same as left_on
    if left_on != right_on:
        new_obs = new_obs.drop(columns=right_on)

    merged_feature_table = ad.AnnData(X=feature_table.to_df(),
                                   obs=new_obs,
                                   dtype='float32')

    # get original table attributes
    table_group = zarr.open_group(
        f"{zarr_url}/tables/{feature_table_name}",
        mode='r')
    orig_attrs = table_group.attrs.asdict()

    if new_feature_table_name is None:
        new_feature_table_name = feature_table_name

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
        task_function=merge_plate_metadata,
        logger_name=logger.name,
    )