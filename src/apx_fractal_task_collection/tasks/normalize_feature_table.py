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
from enum import Enum
import numpy as np

from apx_fractal_task_collection.io_models import InitArgsNormalizeFeatureTable
from fractal_tasks_core.tables import write_table

from pydantic import validate_call


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

class NormalizationMethod(Enum):
    """
    Enum for the normalization method options.

    Attributes:
        z_score: z-score normalization [(x - mean) / std]
        robust_z_score: robust z-score normalization [(x - median) / MAD]
    """

    z_score = "z-score"
    robust_z_score = "robust z-score"

    def normalize(self,
                  data: pd.DataFrame,
                  ctrl_data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the data based on the selected method.

        Args:
            data: Data to be normalized.
            ctrl_data: Data of the control condition.

        Returns:
            Normalized data.
        """

        if self == NormalizationMethod.z_score:
            return (data - ctrl_data.mean()) / ctrl_data.std()
        elif self == NormalizationMethod.robust_z_score:
            return (0.6745 * (data - ctrl_data.median()) /
                    ((ctrl_data - ctrl_data.median()).abs().median()))



@validate_call
def normalize_feature_table(  # noqa: C901
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    init_args: InitArgsNormalizeFeatureTable,
    # Task-specific arguments:
    normalization_method: NormalizationMethod = NormalizationMethod.robust_z_score,
    log_transform_before_normalization: bool = False,
    output_table_name_suffix: str = "_normalized",
):
    """
    Normalize measurements in the feature table with selected method
    and the selected control condition.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        init_args: Intialization arguments provided by
            `init_segment_secondary_objects`.
            (standard argument for Fractal tasks, managed by Fractal server).
        normalization_method: Method to be used for normalization. Choices are:
            - z_score: z-score normalization [(X - mean) / std]
            - robust_z_score: robust z-score normalization [(X - median) / MAD]
        log_transform_before_normalization: Whether to log-transform the data before normalization.
        output_table_name_suffix: Suffix to be added to the output table name.
    """

    # Load the feature table
    feature_table = ad.read_zarr(
        f"{zarr_url}/tables/{init_args.feature_table_name}")
    feature_table_df = feature_table.to_df()

    # Load the feature tables of the control wells
    ctrl_feature_tables = [
        ad.read_zarr(
            f"{z_url}/tables/{init_args.feature_table_name}"
        ) for z_url in init_args.ctrl_zarr_urls
    ]

    # concatenate the control feature tables
    ctrl_df = pd.concat([f.to_df() for f in ctrl_feature_tables])

    if log_transform_before_normalization:
        feature_table_df = np.log1p(feature_table_df)
        ctrl_df = np.log1p(ctrl_df)

    # Normalize the measurements
    normalized_feature_table = normalization_method.normalize(
        data=feature_table_df,
        ctrl_data=ctrl_df)

    # convert to anndata table
    normalized_feature_table = ad.AnnData(X=normalized_feature_table,
                                   obs=feature_table.obs,
                                   dtype='float32')

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
        task_function=normalize_feature_table,
        logger_name=logger.name,
    )
