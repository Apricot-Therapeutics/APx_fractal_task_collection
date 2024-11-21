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
from pydantic import validate_call
from ngio.core import NgffImage
from typing import Optional

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

    ngff_image = NgffImage(zarr_url, "r")
    # read the feature table
    feature_table = ngff_image.tables.get_table(feature_table_name)
    # read the plate metadata
    metadata = pd.read_csv(metadata_path)

    # merge feature table with metadata
    if right_on is None:
        right_on = left_on
    new_feature_table = pd.merge(feature_table.table.reset_index(),
                                 metadata,
                                 left_on=left_on,
                                 right_on=right_on)

    # create a new feature table in the zarr
    feat_table = ngff_image.tables.new(
        name=new_feature_table_name,
        label_image=f'../{feature_table.source_label()}',
        table_type='feature_table',
        overwrite=True
    )

    feat_table.set_table(new_feature_table)
    feat_table.consolidate()


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=merge_plate_metadata,
        logger_name=logger.name,
    )