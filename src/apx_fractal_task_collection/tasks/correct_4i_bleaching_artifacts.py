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

import anndata as ad
import fractal_tasks_core
import pandas as pd
import zarr

from apx_fractal_task_collection.io_models import InitArgsCorrect4iBleachingArtifacts
from fractal_tasks_core.tables import write_table

from pydantic import validate_call


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)


@validate_call
def normalize_feature_table(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsCorrect4iBleachingArtifacts,
    # Task-specific arguments:
    output_table_name_suffix: str = "_bleaching_corrected",
):
    """
    Correct bleaching aritfact in the feature table with the selected control 
    condition.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_correct_4i_bleaching_artifacts`.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_table_name_suffix: Suffix to be added to the output table name.
    """

    # Load the feature table
    feature_table = ad.read_zarr(
        f"{zarr_url}/tables/{init_args.feature_table_name}")

    feature_df = feature_table.to_df()
    features_to_correct = init_args.current_scale_factors.columns

    # Correct bleaching artifacts
    corrected_feature_df = feature_df.copy()
    corrected_feature_df[features_to_correct] = corrected_feature_df[
        features_to_correct].div(init_args.current_scale_factors.values)
    
    # get original table attributes
    table_group = zarr.open_group(
        f"{zarr_url}/tables/{init_args.feature_table_name}",
        mode='r')
    orig_attrs = table_group.attrs.asdict()

    out_group = zarr.open_group(zarr_url, mode="r+")

    # Write to zarr group
    write_table(
        out_group,
        init_args.feature_table_name + output_table_name_suffix,
        normalized_feature_table,
        overwrite=True,
        table_attrs=orig_attrs,
    )

if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=correct_4i_bleaching_artifacts,
        logger_name=logger.name,
    )
