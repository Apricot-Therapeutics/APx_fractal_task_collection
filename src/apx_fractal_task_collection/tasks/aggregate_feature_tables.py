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

from apx_fractal_task_collection.io_models import (
    InitArgsAggregateFeatureTables)

import fractal_tasks_core
import zarr
import anndata as ad
from fractal_tasks_core.tables import write_table
from pydantic import validate_call


__OME_NGFF_VERSION__ = fractal_tasks_core.__OME_NGFF_VERSION__


logger = logging.getLogger(__name__)

def concat_features(feature_tables):

    well_table = ad.concat(feature_tables, axis=1, merge="same")
    well_table.var_names_make_unique(join="trash")

    vars_to_keep = [v for v in well_table.var_names if "trash" not in v]
    morphology = [v for v in vars_to_keep if "Morphology" in v]
    intensity = [v for v in vars_to_keep if "Intensity" in v]
    texture = [v for v in vars_to_keep if "Texture" in v]
    population = [v for v in vars_to_keep if "Population" in v]

    vars_to_keep = morphology + intensity + texture + population

    feature_table = well_table[:, vars_to_keep]

    return feature_table

@validate_call
def aggregate_feature_tables(  # noqa: C901
        *,
        # Default arguments for fractal tasks:
        zarr_url: str,
        init_args: InitArgsAggregateFeatureTables,
        # Task-specific arguments:
        input_table_name: str,
        output_table_name: str,
        output_image: str = '0',
        overwrite: bool = True

) -> None:
    """
    Aggregate feature tables that were calculated per zarr-image to one
    Anndata table containing feature measurements across all zarr-images.

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
        input_table_name: Name of the feature table.
        output_table_name: Name of the aggregated feature table. If this is the
            same as the input_table_name, the input table will be overwritten.
        tables_to_merge: List of feature tables to merge into the main
            feature table. For example, if the input feature table is the table
            for cells, tables to merge could include nuclei and cytoplasm.
            Only use this option if you ran Label Assignment by Overlap first.
        output_image: In which zarr-image to store the aggregated feature
            table. By default, it is saved in the first image of the zarr. If
            output_table_name is the same as input_table_name, the table will
            be overwritten in the same image.
        overwrite: If True, overwrite existing feature table.
    """

    logger.info(
        f"Aggregating features from feature table"
        f" {input_table_name} to well level.")

    feature_tables = [
        ad.read_zarr(
            f"{z_url}/tables/{input_table_name}"
        ) for z_url in init_args.zarr_urls
    ]

    well_table = concat_features(feature_tables)
    out_zarr_path = f"{Path(zarr_url).parent}/{output_image}"


    out_group = zarr.open_group(out_zarr_path, mode="r+")

    # get original table attributes
    table_group = zarr.open_group(
        f"{zarr_url}/tables/{input_table_name}",
        mode='r')
    orig_attrs = table_group.attrs.asdict()

    # Write to zarr group
    write_table(
        out_group,
        output_table_name,
        well_table,
        overwrite=overwrite,
        table_attrs=orig_attrs,
    )


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=aggregate_feature_tables,
        logger_name=logger.name,
    )

